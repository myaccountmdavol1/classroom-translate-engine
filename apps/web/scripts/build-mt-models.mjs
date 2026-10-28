#!/usr/bin/env node
// Prepares Mozilla's translation models (Firefox Translations, MPL-2.0) for
// the browser engine's 'mozilla' provider (services/engine/ct_browser/
// local_mt.py, src/engine/mozilla-sync.js).
//
// Mozilla's buckets send no CORS headers, so a page cannot download the
// models from them: they are copied, once, to a host of our own. This
// script downloads each language's released English -> X model from
// Mozilla's registry, checks it, and writes
//
//   apps/web/mt-models/<code>/<sha256 prefix>-<file>.gz    (not committed:
//       upload the folder to the models host as it is)
//   apps/web/scripts/mt-models.lock.json                    (committed: the
//       files, their sha256 and sizes; build-engine.mjs puts it in the
//       engine manifest, and the worker checks every byte against it)
//
// Each file's name carries its own sha256, so a model Mozilla updates is a
// new file at a new address and a browser never mixes old and new.
//
// The lock pins the models: run again and every download must match it.
// To take Mozilla's newer models on purpose: --update.
//
// Usage (from apps/web):  node scripts/build-mt-models.mjs [--update]

import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, rmSync, existsSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, "..");
const OUT = join(WEB, "mt-models");
const LOCK = join(HERE, "mt-models.lock.json");
const REGISTRY =
  "https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/db/models.json";

// Mozilla's codes for the engine's languages (ct_browser/local_mt.py
// MOZILLA_CODES; a test there checks the two lists agree).
const CODES = ["es", "fr", "ar", "bn", "ur", "ru", "ko", "ja", "vi", "pl", "he", "uk", "pt", "zh", "zh_hant"];

const update = process.argv.includes("--update");
const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");
const mb = (n) => `${(n / 1e6).toFixed(1)} MB`;

async function download(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${url}: ${resp.status}`);
  return Buffer.from(await resp.arrayBuffer());
}

function kindOf(name) {
  for (const k of ["model", "lex", "srcvocab", "trgvocab", "vocab"]) if (name.startsWith(k + ".")) return k;
  throw new Error(`unexpected model file ${name}`);
}

const registry = JSON.parse((await download(REGISTRY)).toString("utf8"));
const old = existsSync(LOCK) ? JSON.parse(readFileSync(LOCK, "utf8")) : null;
const lock = { generatedBy: "apps/web/scripts/build-mt-models.mjs", registry: REGISTRY, license: "MPL-2.0", models: {} };

rmSync(OUT, { recursive: true, force: true });
let total = 0;
for (const code of CODES) {
  const pair = `en-${code}`;
  const entry = (registry.models[pair] || []).find((m) => String(m.releaseStatus || "").startsWith("Release"));
  if (!entry) throw new Error(`Mozilla has no released model for ${pair}`);
  const files = [];
  for (const [, f] of Object.entries(entry.files)) {
    const raw = zlib.gunzipSync(await download(registry.baseUrl.replace(/\/+$/, "") + "/" + f.path));
    const name = f.path.split("/").pop().replace(/\.gz$/, "");
    if (f.uncompressedHash && sha256(raw) !== f.uncompressedHash) {
      throw new Error(`${pair} ${name} does not match Mozilla's own checksum`);
    }
    // Node's gzip writes no name and mtime 0: the same bytes every run.
    const gz = zlib.gzipSync(raw, { level: 9 });
    const sha = sha256(gz);
    const path = `${code}/${sha.slice(0, 16)}-${name}.gz`;
    mkdirSync(join(OUT, code), { recursive: true });
    writeFileSync(join(OUT, path), gz);
    files.push({ kind: kindOf(name), path, sha256: sha, size: gz.length, rawSha256: sha256(raw), rawSize: raw.length, gzip: true });
  }
  files.sort((a, b) => a.path.localeCompare(b.path));
  const was = old?.models?.[code];
  if (was && !update) {
    const same = JSON.stringify(was.files.map((f) => f.rawSha256).sort()) === JSON.stringify(files.map((f) => f.rawSha256).sort());
    if (!same) throw new Error(`${pair} changed upstream since the lock was written; run with --update to take it`);
  }
  const bytes = files.reduce((n, f) => n + f.size, 0);
  total += bytes;
  lock.models[code] = { pair, architecture: entry.architecture, bytes, files };
  console.log(`${pair}: ${files.length} files, ${mb(bytes)}`);
}
writeFileSync(LOCK, JSON.stringify(lock, null, 2) + "\n");
console.log(`all models: ${mb(total)} in ${OUT}`);
