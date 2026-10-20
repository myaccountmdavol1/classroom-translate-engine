// The text reader (Tesseract 5, as tesseract.js-core 7.0.0) for the browser
// engine, called SYNCHRONOUSLY from Python in the same worker.
//
// The engine's OCR stage calls a provider and waits for the words, in the
// middle of a translation job. A synchronous call into the core's
// WebAssembly is the only way to give it that from inside the worker's one
// Python thread (ct_browser/ocr.py calls recognize() through `js.ctTess`).
// Loading is asynchronous and happens between Python entries, before a job
// that needs it: ensure(lang) downloads the core and the language pack.
//
// The calls into the core (TessBaseAPI: Init, SetVariable, SetImageFile,
// Recognize, GetTSVText) follow tesseract.js's worker script (Apache-2.0,
// https://github.com/naptha/tesseract.js); the core ships with its LICENSE
// in tess/LICENSE.
//
// Copied as-is into public/engine/<hash>/ by scripts/build-engine.mjs.

// Tiny modules that validate only with relaxed SIMD, and with plain SIMD.
const RELAXED_SIMD = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 15, 1, 13, 0, 65, 1, 253, 15, 65, 2, 253, 15, 253, 128, 2, 11]);
const SIMD = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);

/**
 * @param {object} opts
 * @param {(rel: string) => Promise<Uint8Array>} opts.asset  a checked, cached
 *   file from this engine build (relative to its folder)
 * @param {object} opts.manifest  the build's manifest.json
 */
export function createTextReader({ asset, manifest }) {
  return {
    ready: false,
    langs: [],
    variant: null,
    mod: null,
    api: null,
    memory: null,

    async ensureCore() {
      if (this.mod) return;
      const variant = WebAssembly.validate(RELAXED_SIMD) ? "relaxedsimd-lstm"
        : WebAssembly.validate(SIMD) ? "simd-lstm" : null;
      if (!variant) throw new Error("this browser has no WebAssembly SIMD, which the text reader needs");
      const base = manifest.tess.variants[variant];
      const [jsBytes, wasm] = await Promise.all([asset(base + ".js"), asset(base + ".wasm")]);
      // The core is a classic script (var TesseractCore = ...); a module
      // worker has no importScripts, so it is evaluated here and its
      // factory returned.
      const factory = new Function(new TextDecoder().decode(jsBytes) + "\n;return TesseractCore;")();
      // If instantiation fails, `receive` is never called and the factory's
      // promise never settles: race it against the failure.
      let failed;
      const instantiateFailed = new Promise((_, reject) => { failed = reject; });
      this.mod = await Promise.race([factory({
        print: () => {},
        printErr: () => {},
        // instantiateWasm (rather than wasmBinary) keeps a handle on the
        // core's memory, for the memory report: the build exports no HEAPU8.
        instantiateWasm: (imports, receive) => {
          WebAssembly.instantiate(wasm, imports).then(({ instance, module }) => {
            const all = [...Object.values(instance.exports), ...Object.values(imports.env || {}), ...Object.values(imports.a || {})];
            this.memory = all.find((x) => x instanceof WebAssembly.Memory) || null;
            receive(instance, module);
          }).catch((e) => failed(new Error("the text reader could not start: " + (e && e.message ? e.message : e))));
          return {};
        },
      }), instantiateFailed]);
      this.variant = variant;
    },

    /** Load `lang` (a Tesseract code such as "eng"); English when this build
     * has no pack for it, which the provider also falls back to. */
    async ensure(lang) {
      const code = manifest.tess.langs[lang] ? lang : "eng";
      await this.ensureCore();
      if (this.langs.includes(code)) return code;
      const data = await asset(manifest.tess.langs[code]);
      this.mod.FS.writeFile(`/${code}.traineddata`, data);
      this.mod.FS.chdir("/");
      // One pack at a time: Init with a new language replaces the old one.
      if (this.api) this.api.End();
      this.api = new this.mod.TessBaseAPI();
      if (this.api.Init(null, code, 1) === -1) throw new Error("the text reader could not start");
      this.langs = [code];
      this.ready = true;
      return code;
    },

    has(lang) {
      return this.langs.includes(manifest.tess.langs[lang] ? lang : "eng");
    },

    /** Fully synchronous: called from Python in the middle of a job.
     * Returns Tesseract's TSV rows (no header line). */
    recognize(png, lang, psm) {
      if (!this.langs.includes(lang)) throw new Error(`language ${lang} is not loaded`);
      this.mod.FS.writeFile("/input", png);
      // Every call sets it: the core keeps the last value it was given.
      this.api.SetVariable("tessedit_pageseg_mode", String(psm));
      if (this.api.SetImageFile(1, 0) === 1) throw new Error("the picture could not be opened");
      this.api.Recognize(null);
      const tsv = this.api.GetTSVText();
      this.api.Clear();
      return tsv;
    },

    heapBytes() {
      if (!this.mod) return 0;
      return this.memory?.buffer?.byteLength || this.mod.HEAPU8?.length || 0;
    },
  };
}
