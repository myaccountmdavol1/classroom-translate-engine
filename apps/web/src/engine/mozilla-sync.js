// Mozilla's translation models (Bergamot, the WebAssembly build of Marian
// that Firefox translates with) for the browser engine, called
// SYNCHRONOUSLY from Python in the same worker.
//
// The engine's translate stage calls a provider and waits for the text, in
// the middle of a translation job. A synchronous call into the runtime's
// WebAssembly is the only way to give it that from inside the worker's one
// Python thread (ct_browser/local_mt.py calls translate() through
// `js.ctMoz`). Loading is asynchronous and happens between Python entries,
// before a job that needs it: ensure(code) downloads the runtime and the
// language's model.
//
// The runtime is @browsermt/bergamot-translator 0.4.9 (MPL-2.0); the calls
// into it (AlignedMemory, TranslationModel, BlockingService.translate) and
// the model settings follow its worker/translator-worker.js. The models are
// Mozilla's (MPL-2.0), from the registry in scripts/build-mt-models.mjs.
//
// Copied as-is into public/engine/<hash>/ by scripts/build-engine.mjs.

// What the runtime's own worker sets, with the overrides it always applies.
// alignment: soft is what HTML mode needs to put markup back.
const MODEL_CONFIG = [
  "beam-size: 1",
  "normalize: 1.0",
  "word-penalty: 0",
  "cpu-threads: 0",
  "gemm-precision: int8shiftAlphaAll",
  "skip-cost: true",
  "alignment: soft",
  "quiet: true",
  "quiet-translation: true",
  "max-length-break: 128",
  "mini-batch-words: 1024",
  "workspace: 128",
  "max-length-factor: 2.0",
].join("\n");

// The runtime imports its matrix kernels under these names; its own build
// carries them as the "Fallback" exports.
const GEMM = {
  int8_prepare_a: "int8PrepareAFallback",
  int8_prepare_b: "int8PrepareBFallback",
  int8_prepare_b_from_transposed: "int8PrepareBFromTransposedFallback",
  int8_prepare_b_from_quantized_transposed: "int8PrepareBFromQuantizedTransposedFallback",
  int8_prepare_bias: "int8PrepareBiasFallback",
  int8_multiply_and_add_bias: "int8MultiplyAndAddBiasFallback",
  int8_select_columns_of_b: "int8SelectColumnsOfBFallback",
};

/**
 * @param {object} opts
 * @param {(rel: string) => Promise<Uint8Array>} opts.asset  a checked, cached
 *   file from this engine build (relative to its folder)
 * @param {(file: object) => Promise<Uint8Array>} opts.modelFile  a checked,
 *   cached, decompressed model file from manifest.mt.models
 * @param {object} opts.manifest  the build's manifest.json
 */
export function createLocalTranslator({ asset, modelFile, manifest }) {
  return {
    ready: false,
    langs: [],
    mod: null,
    service: null,
    model: null,
    memory: null,

    async ensureRuntime() {
      if (this.mod) return;
      const rt = manifest.mt.runtime;
      const [jsBytes, wasm] = await Promise.all([asset(rt.js), asset(rt.wasm)]);
      let instance = null;
      const gemm = Object.fromEntries(
        Object.entries(GEMM).map(([name, fallback]) => [name, (...args) => instance.exports[fallback](...args)]),
      );
      let failed;
      const instantiateFailed = new Promise((_, reject) => { failed = reject; });
      const started = new Promise((resolve) => {
        const Module = {
          print: () => {},
          printErr: () => {},
          instantiateWasm: (imports, receive) => {
            WebAssembly.instantiate(wasm, { ...imports, wasm_gemm: gemm }).then((out) => {
              instance = out.instance;
              const all = [...Object.values(instance.exports), ...Object.values(imports.env || {})];
              this.memory = all.find((x) => x instanceof WebAssembly.Memory) || null;
              receive(out.instance, out.module);
            }).catch((e) => failed(new Error("the translator could not start: " + (e && e.message ? e.message : e))));
            return {};
          },
          onRuntimeInitialized: () => resolve(Module),
        };
        // The runtime is a classic script (var Module = typeof Module !=
        // "undefined" ? Module : {}); a module worker has no importScripts,
        // so it is evaluated here with our Module in its scope.
        new Function("Module", new TextDecoder().decode(jsBytes))(Module);
      });
      this.mod = await Promise.race([started, instantiateFailed]);
      this.service = new this.mod.BlockingService({ cacheSize: 0 });
    },

    aligned(bytes, alignment) {
      const memory = new this.mod.AlignedMemory(bytes.byteLength, alignment);
      memory.getByteArrayView().set(bytes);
      return memory;
    },

    /** Load the model for `code` (a key of manifest.mt.models, such as "es").
     * One model at a time: loading another frees the last. */
    async ensure(code) {
      const entry = manifest.mt.models[code];
      if (!entry) throw new Error(`no translation model for ${code} in this build`);
      await this.ensureRuntime();
      if (this.langs.includes(code)) return code;
      const files = await Promise.all(entry.files.map((f) => modelFile(f).then((bytes) => [f.kind, bytes])));
      const of = (kind) => files.filter(([k]) => k === kind).map(([, b]) => b);
      const vocabBytes = of("vocab").length ? of("vocab") : [...of("srcvocab"), ...of("trgvocab")];
      const vocabs = new this.mod.AlignedMemoryList();
      for (const v of vocabBytes) vocabs.push_back(this.aligned(v, 64));
      const model = new this.mod.TranslationModel(
        MODEL_CONFIG,
        this.aligned(of("model")[0], 256),
        this.aligned(of("lex")[0], 64),
        vocabs,
        null,
      );
      if (this.model) this.model.delete();
      this.model = model;
      this.langs = [code];
      this.ready = true;
      return code;
    },

    has(code) {
      return this.langs.includes(code);
    },

    /** Fully synchronous: called from Python in the middle of a job.
     * `texts` is a list of strings; returns their translations in order. */
    translate(texts, code, html) {
      if (!this.langs.includes(code)) throw new Error(`language ${code} is not loaded`);
      const input = new this.mod.VectorString();
      const options = new this.mod.VectorResponseOptions();
      for (const t of texts) {
        input.push_back(String(t));
        options.push_back({ qualityScores: false, alignment: Boolean(html), html: Boolean(html) });
      }
      let responses = null;
      try {
        responses = this.service.translate(this.model, input, options);
        const out = [];
        for (let i = 0; i < responses.size(); i++) out.push(responses.get(i).getTranslatedText());
        return out;
      } finally {
        input.delete();
        options.delete();
        if (responses) responses.delete();
      }
    },

    heapBytes() {
      return this.memory?.buffer?.byteLength || 0;
    },
  };
}
