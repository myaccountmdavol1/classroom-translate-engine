// The browser engine: the unchanged ct_engine in Pyodide 0.29.3, in a
// dedicated module worker, so Python never runs on the page's main thread.
// Started and owned by the engine host (src/engine/host.ts) in one tab.
//
// Every Python call is one "entry", and exactly one runs at a time. A
// request is one entry; a job is one entry, run after the requests waiting
// for Python have been answered. Jobs are never run inside the request that
// queued them (ct_browser/executor.py says why).
//
// Before a translation job that will read scanned pages or words in
// pictures, the text reader (tesseract-sync.js) is loaded between entries,
// so requests keep being answered while it downloads.
//
// Every downloaded file is checked before use, from the network and from
// Cache Storage alike: Pyodide's core files against pins in manifest.json,
// its packages against the sha256 in its pinned lock file, and this build's
// own files against manifest.json. A web filter's "blocked" page (HTML,
// often a 200) is never cached or run. A download that stops sending data
// gives up with a plain message.
//
// Documents live in Python's store in memory (MEMFS, under /ct). Every write
// the engine makes there is also posted to the host, in order, the moment it
// is made ("journal": ct_browser/journal.py tells, this reads the bytes from
// MEMFS and hands the buffer over without copying), and the host keeps the
// lasting copy in OPFS. The host puts a document back ("hydrate") before the
// first request that names it, and takes it out of memory again ("evict")
// once it has been idle a while. Both go through the same queue as
// requests, so they never happen in the middle of a Python entry.
//
// Messages in:   {type:"boot", origin, recycleBytes}
//                {type:"request", id, method, path, headers, body}
//                {type:"abort", id}
//                {type:"hydrate", id, files:[[key, ArrayBuffer]...], resume}
//                {type:"evict", id}
//                {type:"resume", ids, stops, code, quiet}
//                {type:"resync", id}         write the whole document out again
//                                            (after the disk refused a write)
//                {type:"limit", recycleBytes}   a new size limit (0: never)
//                {type:"halt"}               finish the entry in progress, then
//                                            run no more Python ("halted")
//                {type:"unhalt"}             carry on after all
// Messages out:  {type:"progress", phase:"engine"|"reader", loaded, total}
//                {type:"ready", summary} | {type:"boot-failed", message}
//                {type:"response", id, status, headers, body}
//                {type:"dropped", id}
//                {type:"snapshot", raw}      a JobOut or DocumentOut (JSON)
//                {type:"journal", seq, op, key, data}   a store write, in order
//                {type:"journal-failed", seq, key, message}  a write that could
//                                            not be read back to be told
//                {type:"resync-start", id} {type:"resync-file", id, key, data}
//                {type:"resync-end", id, keys}
//                {type:"hydrated", id, bytes} | {type:"evicted", id}
//                {type:"resumed", ids, resumed, quiet}
//                {type:"job", state:"start"|"end", documentId, jobId, translation,
//                 error?, memory?, pending?, stopChecks?}
//                {type:"recycle", reason, memory}   this worker has grown too
//                                            big (or ran out of memory): it has
//                                            stopped, between entries, and the
//                                            host starts a fresh one
//                {type:"halted"}
//                {type:"fatal", message}     Pyodide can no longer be used: the
//                                            host replaces this worker
//
// A running job's checkpoints ask the host whether she pressed Stop
// (ctCancelRequested below, used by ct_browser/cancel.py): a synchronous
// request, which the Service Worker hands to the host's free main thread.
//
// Copied as-is into public/engine/<hash>/ by scripts/build-engine.mjs.

import { createTextReader } from "./tesseract-sync.js";

const OWN_BASE = new URL("./", self.location.href).href;
const RESPONSE_TIMEOUT_MS = 240000; // no answer at all for this long: give up
const IDLE_TIMEOUT_MS = 90000; // no new bytes for this long: give up
const nativeFetch = self.fetch.bind(self);
const post = (msg, transfer) => self.postMessage(msg, transfer || []);

let manifest = null;
let pyodideBase = null;
let pyodideCache = null;
let ownCache = null;
let lockSha = null; // Pyodide package file name -> sha256, from its pinned lock

// ---- Download progress ----------------------------------------------------------
const progress = { phase: "engine", loaded: 0, total: 0, lastPost: 0 };
function addBytes(n) {
  progress.loaded += n;
  const now = Date.now();
  if (now - progress.lastPost > 150) {
    progress.lastPost = now;
    post({ type: "progress", phase: progress.phase, loaded: progress.loaded, total: progress.total });
  }
}
function startProgress(phase, total) {
  progress.phase = phase;
  progress.loaded = 0;
  progress.total = total;
  progress.lastPost = 0;
  post({ type: "progress", phase, loaded: 0, total });
}

// ---- Cache-first, checked fetches (lifted from the Phase 0 probe) -----------------
class CheckFailed extends Error {}

const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
async function sha256Hex(bytes) { return hex(await crypto.subtle.digest("SHA-256", bytes)); }

function bucketFor(abs) {
  if (abs.startsWith(pyodideBase)) return pyodideCache;
  if (abs.startsWith(OWN_BASE)) return ownCache;
  if (abs.startsWith(self.location.origin + "/fonts/v1/")) return ownCache;
  return null;
}

// The sha256 a file must have, or null when there is no pin for it.
function expectedSha(abs) {
  if (abs.startsWith(pyodideBase)) {
    const name = abs.slice(pyodideBase.length).split("?")[0];
    return manifest.pyodide.core[name] || (lockSha && lockSha[name]) || null;
  }
  if (abs.startsWith(OWN_BASE)) {
    return manifest.files[abs.slice(OWN_BASE.length).split("?")[0]]?.sha256 || null;
  }
  const font = manifest.fonts.find((f) => abs === self.location.origin + f.url);
  return font ? font.sha256 : null;
}

async function readWithProgress(resp, idle) {
  const reader = resp.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    idle.reset();
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    addBytes(value.length);
  }
  idle.clear();
  const out = new Uint8Array(got);
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.length; }
  return out;
}

// A fetch that gives up, with a plain message, when the network goes quiet.
async function patientFetch(abs, init, label) {
  const ac = new AbortController();
  let timer = null;
  let why = "";
  const idle = {
    reset(ms = IDLE_TIMEOUT_MS) {
      clearTimeout(timer);
      timer = setTimeout(() => { why = `no data for ${ms / 1000} s`; ac.abort(); }, ms);
    },
    clear() { clearTimeout(timer); },
  };
  idle.reset(RESPONSE_TIMEOUT_MS);
  try {
    const resp = await nativeFetch(abs, { ...init, signal: ac.signal });
    return { resp, ac, idle };
  } catch (e) {
    idle.clear();
    if (why) throw new Error(`The download of ${label} stopped (${why}). The network may be blocking it or be very slow.`);
    throw e;
  }
}

async function cachedFetch(input, init) {
  const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  const abs = new URL(url, self.location.href).href;
  const bucket = bucketFor(abs);
  if (!bucket || (init && init.method && init.method !== "GET")) return nativeFetch(input, init);
  const label = abs.split("/").pop();
  const want = expectedSha(abs);
  // Pyodide passes `integrity` for its packages; our own check of the same
  // sha256 replaces it, covers cache hits too, and keeps byte progress.
  const rest = { ...(init || {}) };
  delete rest.integrity;
  delete rest.signal;
  let cache = null;
  try { cache = await caches.open(bucket); } catch { /* Cache Storage blocked */ }
  if (cache) {
    const hit = await cache.match(abs);
    if (hit) {
      const body = new Uint8Array(await hit.arrayBuffer());
      if (want && (await sha256Hex(body)) !== want) {
        try { await cache.delete(abs); } catch { /* fetched again below */ }
      } else {
        addBytes(body.length);
        return new Response(body, { status: 200, headers: hit.headers });
      }
    }
  }
  const { resp, ac, idle } = await patientFetch(abs, { ...rest, cache: rest.cache || "default" }, label);
  if (!resp.ok) { idle.clear(); return resp; }
  const ctype = (resp.headers.get("content-type") || "").toLowerCase();
  let body;
  try {
    body = await readWithProgress(resp, idle);
  } catch (e) {
    idle.clear();
    if (ac.signal.aborted) throw new Error(`The download of ${label} stopped. The network may be blocking it or be very slow.`);
    throw e;
  }
  if (ctype.includes("text/html")) {
    throw new CheckFailed(`${label} came back as a web page, so a web filter may be blocking ${new URL(abs).host}.`);
  }
  if (want && (await sha256Hex(body)) !== want) {
    throw new CheckFailed(`${label} did not match its checksum, so something on the network changed it.`);
  }
  const headers = new Headers(resp.headers);
  headers.set("content-length", String(body.length));
  const fresh = new Response(body, { status: resp.status, statusText: resp.statusText, headers });
  if (cache && want) {
    try { await cache.put(abs, fresh.clone()); } catch { /* storage full: still usable now */ }
  }
  return fresh;
}
self.fetch = cachedFetch;

async function asset(rel) {
  const resp = await cachedFetch(new URL(rel, OWN_BASE).href);
  if (!resp.ok) throw new Error(`could not load ${rel} (${resp.status})`);
  return new Uint8Array(await resp.arrayBuffer());
}

// Pyodide's JS files are loaded with import(), which would skip the checks
// and Cache Storage: they are fetched (checked, cached) and imported from a Blob.
async function importChecked(name) {
  const bytes = await (await cachedFetch(pyodideBase + name)).arrayBuffer();
  const url = URL.createObjectURL(new Blob([bytes], { type: "text/javascript" }));
  try { return await import(url); } finally { URL.revokeObjectURL(url); }
}

// ---- Boot ---------------------------------------------------------------------------
let py = null;
let glue = null;
const reader = { tess: null, failed: new Set() };

async function boot(origin) {
  const t0 = performance.now();
  manifest = await (await nativeFetch(new URL("manifest.json", OWN_BASE), { cache: "no-cache" })).json();
  pyodideBase = manifest.pyodide.indexURL;
  pyodideCache = `ct-pyodide-${manifest.pyodide.version}`;
  ownCache = `ct-engine-${manifest.hash}`;
  // Builds other than this one and the one before it are no longer served.
  try {
    for (const k of await caches.keys()) {
      if (k.startsWith("ct-engine-") && k !== ownCache) await caches.delete(k);
    }
  } catch { /* Cache Storage blocked: everything comes from the network */ }
  startProgress("engine", manifest.bootBytes);

  const lockBytes = await (await cachedFetch(pyodideBase + "pyodide-lock.json")).arrayBuffer();
  const lock = JSON.parse(new TextDecoder().decode(lockBytes));
  lockSha = {};
  for (const p of Object.values(lock.packages)) lockSha[p.file_name] = p.sha256;
  const { loadPyodide } = await importChecked("pyodide.mjs");
  if (typeof globalThis._createPyodideModule !== "function") await importChecked("pyodide.asm.js");
  const stderr = [];
  py = await loadPyodide({
    indexURL: pyodideBase,
    lockFileContents: lock,
    packageBaseUrl: pyodideBase,
    stdout: () => {},
    stderr: (line) => { stderr.push(String(line).slice(0, 300)); if (stderr.length > 30) stderr.shift(); },
  });
  await py.loadPackage(manifest.pyodide.packages, { messageCallback: () => {}, errorCallback: (m) => stderr.push(String(m)) });

  const site = py.runPython("import site; site.getsitepackages()[0]");
  for (const rel of manifest.wheels) py.unpackArchive(await asset(rel), "wheel", { extractDir: site });
  py.FS.mkdirTree("/engine");
  py.unpackArchive(await asset("engine.zip"), "zip", { extractDir: "/engine" });
  // The engine's own font faces, identical to the files the editor shows.
  const fonts = await Promise.all(manifest.fonts.map(async (f) => {
    const resp = await cachedFetch(self.location.origin + f.url);
    if (!resp.ok) throw new Error(`could not load ${f.url} (${resp.status})`);
    return [f.dest, new Uint8Array(await resp.arrayBuffer())];
  }));
  for (const [dest, bytes] of fonts) {
    const path = `/engine/ct_engine/fonts/faces/${dest}`;
    py.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
    py.FS.writeFile(path, bytes);
  }
  py.runPython("import sys, importlib; sys.path.insert(0, '/engine'); importlib.invalidate_caches()");

  glue = py.pyimport("ct_browser.boot");
  const env = { ...manifest.env, CT_RELAY_URL: `${origin}/api/mt` };
  storeRoot = String(env.CT_STORAGE_ROOT || "/ct").replace(/\/+$/, "");
  const summary = JSON.parse(glue.start(
    JSON.stringify(env),
    (raw) => post({ type: "snapshot", raw }),
    journal,
  ));
  summary.seconds = +((performance.now() - t0) / 1000).toFixed(2);
  summary.bytes = progress.loaded;
  post({ type: "progress", phase: "engine", loaded: progress.total, total: progress.total });
  return summary;
}

// ---- The store journal and the documents in memory --------------------------------------
let storeRoot = "/ct";

// Called by Python (ct_browser/journal.py) right after each store write,
// synchronously and in order. The bytes are read from MEMFS here, into a
// buffer of their own that is handed to the host, not copied again.
function journal(seq, op, key, path) {
  let data = null;
  if (op === "lost") {
    post({ type: "journal-failed", seq, key: String(key), message: "the engine could not find the file it wrote" });
    return;
  }
  if (op === "put") {
    try {
      data = py.FS.readFile(String(path));
    } catch (e) {
      post({ type: "journal-failed", seq, key: String(key), message: String(e && e.message || e) });
      return;
    }
  }
  const buf = data ? (data.byteOffset === 0 && data.byteLength === data.buffer.byteLength ? data.buffer : data.slice().buffer) : null;
  post({ type: "journal", seq, op: String(op), key: String(key), data: buf }, buf ? [buf] : []);
}

function mkdirs(path) {
  py.FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
}

// A document's files, from the host's lasting copy, into MEMFS.
function hydrate(id, files) {
  let bytes = 0;
  for (const [key, buf] of files) {
    const path = `${storeRoot}/${key}`;
    mkdirs(path);
    const data = new Uint8Array(buf);
    py.FS.writeFile(path, data);
    bytes += data.byteLength;
  }
  return bytes;
}

function removeTree(path) {
  let st;
  try { st = py.FS.stat(path); } catch { return; }
  if (!py.FS.isDir(st.mode)) { py.FS.unlink(path); return; }
  for (const name of py.FS.readdir(path)) {
    if (name !== "." && name !== "..") removeTree(`${path}/${name}`);
  }
  py.FS.rmdir(path);
}

// Every file of a document in MEMFS, as store keys, the job and manifest
// files last: a tab that dies part-way leaves the old manifest pointing at
// files that are all there.
function documentKeys(id) {
  const base = `${storeRoot}/documents/${id}`;
  const out = [];
  const walk = (path) => {
    let names;
    try { names = py.FS.readdir(path); } catch { return; }
    for (const name of names) {
      if (name === "." || name === "..") continue;
      const p = `${path}/${name}`;
      const st = py.FS.stat(p);
      if (py.FS.isDir(st.mode)) walk(p);
      else out.push(p.slice(storeRoot.length + 1));
    }
  };
  walk(base);
  const last = (k) => (/\/manifest\.json$/.test(k) ? 2 : /\/jobs\//.test(k) ? 1 : 0);
  return out.sort((a, b) => last(a) - last(b) || (a < b ? -1 : a > b ? 1 : 0));
}

// The whole document told to the host again, in one go between Python
// entries (so nothing writes to it meanwhile).
function resync(id) {
  const keys = documentKeys(id);
  post({ type: "resync-start", id });
  for (const key of keys) {
    const data = py.FS.readFile(`${storeRoot}/${key}`);
    const buf = data.byteOffset === 0 && data.byteLength === data.buffer.byteLength ? data.buffer : data.slice().buffer;
    post({ type: "resync-file", id, key, data: buf }, [buf]);
  }
  post({ type: "resync-end", id, keys });
}

// Bytes of every file under the store: MEMFS keeps them in JS memory,
// outside Python's heap, and they count toward recycling.
function memfsBytes(path = storeRoot) {
  let total = 0;
  let names;
  try { names = py.FS.readdir(path); } catch { return 0; }
  for (const name of names) {
    if (name === "." || name === "..") continue;
    const p = `${path}/${name}`;
    try {
      const st = py.FS.stat(p);
      total += py.FS.isDir(st.mode) ? memfsBytes(p) : st.size;
    } catch { /* gone meanwhile */ }
  }
  return total;
}

function memory() {
  let heap = 0;
  try { heap = py._module.HEAP8.length; } catch { /* not exposed */ }
  return { heap, tess: reader.tess ? reader.tess.heapBytes() : 0, memfs: memfsBytes() };
}

// ---- Stop, asked from inside a running job ------------------------------------------
// Python calls this between pages (ct_browser/cancel.py). This thread is
// busy with the job, so the question goes out as a synchronous request; the
// Service Worker routes it to the host, which remembers what was stopped.
// Any trouble answers "no": the job then finishes as it would have.
//
// The job's thread waits for each answer, so a host that cannot answer must
// never cost more than a moment: an answer that fails or is slow stops the
// asking for a while (5 s, doubling to a minute), and the counts go to the
// host for the diagnostic.
const STOP_TIMEOUT_MS = 800;
const STOP_SLOW_MS = 400;
const stopChecks = { asks: 0, failed: 0, slow: 0, skipped: 0, totalMs: 0, maxMs: 0 };
let stopQuietUntil = 0;
let stopBackoffMs = 0;
function stopTrouble() {
  stopBackoffMs = stopBackoffMs ? Math.min(stopBackoffMs * 2, 60000) : 5000;
  stopQuietUntil = Date.now() + stopBackoffMs;
}
self.ctCancelRequested = (jobId) => {
  if (Date.now() < stopQuietUntil) {
    stopChecks.skipped += 1;
    return false;
  }
  stopChecks.asks += 1;
  const t0 = performance.now();
  let ok = false;
  let stop = false;
  try {
    const xhr = new XMLHttpRequest();
    xhr.open("GET", `${self.location.origin}/engine/api/__stop/${encodeURIComponent(String(jobId))}`, false);
    xhr.timeout = STOP_TIMEOUT_MS;
    xhr.send();
    ok = xhr.status === 200;
    stop = ok && xhr.responseText.trim() === "1";
  } catch {
    ok = false;
  }
  const ms = performance.now() - t0;
  stopChecks.totalMs += ms;
  stopChecks.maxMs = Math.max(stopChecks.maxMs, ms);
  if (!ok) stopChecks.failed += 1;
  else if (ms > STOP_SLOW_MS) stopChecks.slow += 1;
  if (!ok || ms > STOP_SLOW_MS) stopTrouble();
  else stopBackoffMs = 0;
  return stop;
};

// ---- The one-entry queue ---------------------------------------------------------------
const requests = []; // waiting for Python, oldest first (requests, hydrate, evict, resume)
let running = false; // a Python entry is in progress
let jobWaiting = false; // the engine's job pool has work
let preparing = null; // the text reader loading before a job
let dead = false; // Pyodide failed fatally: nothing more runs here
let halted = false; // asked to stop (or stopped itself to be replaced)
let haltSaid = false;
let recycled = false; // stopped itself to be replaced: never carries on
let recycleBytes = 0; // past this (Python + text reader + MEMFS), a fresh worker

// Pyodide marks an error it cannot recover from (a WebAssembly trap, memory
// running out); after one, every call throws "already fatally failed". The
// host is told, and replaces this worker: what it held is gone, as after a
// restart, and the pages put their documents back from their backups.
function isFatal(e) {
  return Boolean(
    e && (e.pyodide_fatal_error || e instanceof WebAssembly.RuntimeError ||
      /fatally failed/i.test(String(e.message || e))),
  );
}
function die(e) {
  if (dead) return;
  dead = true;
  requests.length = 0;
  post({ type: "fatal", message: String((e && e.message) || e).split("\n").slice(-3).join(" ").slice(0, 500) });
}

async function runRequest(req) {
  try {
    const body = req.body ? new Uint8Array(req.body) : null;
    const result = await glue.handle(req.method, req.path, JSON.stringify(req.headers || {}), body);
    const [status, headersJson, content] = result;
    // A copy of exactly its bytes, so the transfer moves nothing else.
    const out = content.byteOffset === 0 && content.byteLength === content.buffer.byteLength
      ? content.buffer : content.slice().buffer;
    post({ type: "response", id: req.id, status, headers: JSON.parse(headersJson), body: out }, [out]);
  } catch (e) {
    if (isFatal(e)) { die(e); return; }
    const message = "Something went wrong on this computer while doing that. Please try again.";
    const out = new TextEncoder().encode(JSON.stringify({ error: message, detail: message, diagnostic: String(e && e.message || e).split("\n").slice(-4).join("\n") })).buffer;
    post({ type: "response", id: req.id, status: 500, headers: [["content-type", "application/json"]], body: out }, [out]);
  }
  try { jobWaiting = glue.has_pending_job(); } catch (e) { jobWaiting = false; if (isFatal(e)) die(e); }
}

// Hydrate, evict and resume: queued like requests, so they run between
// Python entries, in the order the host sent them.
function runHousekeeping(item) {
  try {
    if (item.type === "hydrate") {
      const bytes = hydrate(item.id, item.files || []);
      item.files = null;
      let resumed = [];
      if (item.resume) resumed = JSON.parse(glue.resume(JSON.stringify({ ids: [item.id], stops: item.stops || [], code: item.code || null }))).resumed;
      post({ type: "hydrated", id: item.id, bytes, resumed });
    } else if (item.type === "evict") {
      removeTree(`${storeRoot}/documents/${item.id}`);
      post({ type: "evicted", id: item.id });
    } else if (item.type === "resync") {
      resync(item.id);
    } else if (item.type === "resume") {
      const out = JSON.parse(glue.resume(JSON.stringify({ ids: item.ids ?? null, stops: item.stops || [], code: item.code || null })));
      post({ type: "resumed", ids: item.ids ?? null, resumed: out.resumed, quiet: Boolean(item.quiet) });
    }
  } catch (e) {
    if (isFatal(e)) { die(e); return; }
    post({ type: "housekeeping-failed", item: item.type, id: item.id || null, message: String(e && e.message || e).split("\n").slice(-3).join(" ").slice(0, 500) });
  }
  try { jobWaiting = glue.has_pending_job(); } catch (e) { jobWaiting = false; if (isFatal(e)) die(e); }
}

function sayHalted() {
  if (haltSaid) return;
  haltSaid = true;
  post({ type: "halted" });
}

async function prepareReader(lang) {
  if (!reader.tess) reader.tess = createTextReader({ asset, manifest });
  self.ctTess = reader.tess;
  startProgress("reader", reader.tess.mod ? 0 : manifest.readerBytes);
  await reader.tess.ensure(lang);
  post({ type: "progress", phase: "reader", loaded: progress.total, total: progress.total });
}

function runJob() {
  let info = null;
  try { info = JSON.parse(glue.next_job() || "null"); } catch (e) { info = null; if (isFatal(e)) { die(e); return; } }
  if (!info) { jobWaiting = false; return; }
  const lang = info.ocr_language;
  if (lang && !(reader.tess && reader.tess.has(lang)) && !reader.failed.has(info.job_id)) {
    // Loaded between entries: requests are answered meanwhile. If it cannot
    // load, the job still runs, and each page it could not read says so.
    preparing = prepareReader(lang)
      .catch(() => { reader.failed.add(info.job_id); })
      .finally(() => { preparing = null; pump(); });
    return;
  }
  for (const k of Object.keys(stopChecks)) stopChecks[k] = 0;
  stopQuietUntil = 0;
  stopBackoffMs = 0;
  post({ type: "job", state: "start", documentId: info.document_id, jobId: info.job_id, translation: info.translation });
  let out = {};
  try {
    out = JSON.parse(glue.run_next_job());
  } catch (e) {
    if (isFatal(e)) { die(e); return; }
    out = { error: String(e && e.message || e).slice(-1500) };
  }
  try { jobWaiting = glue.has_pending_job(); } catch (e) { jobWaiting = false; if (isFatal(e)) { die(e); return; } }
  const mem = memory();
  post({ type: "job", state: "end", documentId: info.document_id, jobId: info.job_id, translation: info.translation, error: out.error || null, memory: mem, pending: jobWaiting, stopChecks: { ...stopChecks, totalMs: Math.round(stopChecks.totalMs), maxMs: Math.round(stopChecks.maxMs) } });
  // Between jobs, never during one: a worker that ran out of memory, or has
  // grown past the limit (the wasm heap never shrinks), stops here and the
  // host starts a fresh one, which picks up any waiting job from the store.
  const total = mem.heap + mem.tess + mem.memfs;
  if (out.recycle || (recycleBytes > 0 && total > recycleBytes)) {
    halted = true;
    haltSaid = true;
    recycled = true;
    post({ type: "recycle", reason: out.recycle ? "memory-error" : "size", memory: mem, limit: recycleBytes });
  }
}

async function pump() {
  if (running || !glue || dead) return;
  running = true;
  try {
    for (;;) {
      if (dead) break;
      if (halted) { sayHalted(); break; }
      const req = requests.shift();
      if (req) {
        if (req.type === "request") await runRequest(req);
        else runHousekeeping(req);
        continue;
      }
      if (!jobWaiting || preparing) break;
      runJob();
      // Let messages that arrived during the job (new requests, aborts)
      // land before deciding what runs next.
      await new Promise((r) => setTimeout(r, 0));
    }
  } finally {
    running = false;
  }
}

// ---- Messages ------------------------------------------------------------------------------
let booting = null;
self.onmessage = (ev) => {
  const msg = ev.data || {};
  if (msg.type === "boot") {
    recycleBytes = Number(msg.recycleBytes) || 0;
    if (!booting) {
      booting = boot(msg.origin).then(
        (summary) => { post({ type: "ready", summary }); pump(); },
        (e) => post({
          type: "boot-failed",
          message: e instanceof CheckFailed ? e.message : String(e && e.message || e).split("\n").slice(-3).join(" "),
          check: e instanceof CheckFailed,
        }),
      );
    }
  } else if (msg.type === "request" || msg.type === "hydrate" || msg.type === "evict" || msg.type === "resume" || msg.type === "resync") {
    if (dead || halted) return; // the host has been told, and answers it
    requests.push(msg);
    pump();
  } else if (msg.type === "limit") {
    recycleBytes = Number(msg.recycleBytes) || 0;
  } else if (msg.type === "halt") {
    halted = true;
    // Always answered, even by a worker that already stopped itself to be
    // replaced: the host may be waiting for exactly this.
    if (!running) { haltSaid = true; post({ type: "halted" }); }
  } else if (msg.type === "unhalt") {
    // The host decided not to hand over after all (she started something).
    if (recycled) return;
    halted = false;
    haltSaid = false;
    pump();
  } else if (msg.type === "abort") {
    // Dropped only while it waits: once Python has it, it is answered.
    const i = requests.findIndex((r) => r.type === "request" && r.id === msg.id);
    if (i >= 0) {
      requests.splice(i, 1);
      post({ type: "dropped", id: msg.id });
    }
  }
};
