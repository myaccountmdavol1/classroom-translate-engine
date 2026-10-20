#!/usr/bin/env node
// Builds the browser engine: everything the teacher's browser downloads to
// run the unchanged ct_engine in a Web Worker (see
// docs/superpowers/plans/2026-10-19-browser-engine-pdf.md, Phases 2-3).
//
// Output, committed so a deploy needs no Python or network:
//   public/engine/<hash>/
//     engine.zip            ct_engine + ct_browser, deterministic. The font
//                           faces' .ttf files are left out: the worker takes
//                           the identical files from /fonts/v1 (checked
//                           here), which the editor loads anyway.
//     wheels/*.whl          pure-Python wheels Pyodide 0.29.3 does not ship,
//                           the whole closure, sha256-pinned (school filters
//                           may block PyPI, so they come from our origin)
//     tess/*                tesseract.js-core 7.0.0 (relaxedsimd + simd) and
//                           tessdata_fast eng, loaded only before a
//                           translation that needs them
//     engine.worker.js      the worker, and tesseract-sync.js, copied from
//     tesseract-sync.js     src/engine (plain JS, not bundled by Next)
//     manifest.json         every file's size and sha256, Pyodide's pins and
//                           the byte total of a first visit
//   src/engine/manifest.generated.ts   the hash, and the answers the host
//                           gives without Python (languages, output modes,
//                           the engine's limits)
//
// The directory is named by a hash of its contents, so it is served as
// immutable and a new build never mixes with an old one. The last two are
// kept, so a tab still running the previous build finishes its job.
// Pyodide itself loads from jsDelivr (pinned by sha256 in the worker).
//
// Usage (from apps/web):  node scripts/build-engine.mjs
// Python used for pip:     $CT_ENGINE_PYTHON, else services/engine/.venv/bin/python

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import {
  copyFileSync, cpSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync,
  renameSync, rmSync, statSync, writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import zlib from "node:zlib";

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = resolve(HERE, "..");
const REPO = resolve(WEB, "..", "..");
const ENGINE_ROOT = join(REPO, "services", "engine");
const PACKAGES = ["ct_engine", "ct_browser"];
const OUT_ROOT = join(WEB, "public", "engine");
const FONTS_DIR = join(WEB, "public", "fonts", "v1");
const SRC_ENGINE = join(WEB, "src", "engine");
const GENERATED = join(SRC_ENGINE, "manifest.generated.ts");
const VENV_PY = join(ENGINE_ROOT, ".venv", "bin", "python");
const PY = process.env.CT_ENGINE_PYTHON || (existsSync(VENV_PY) ? VENV_PY : "python3");
// Everything Vercel uploads from apps/web/public. A deploy has a size limit,
// and two engine builds plus fonts must stay well inside it.
const PUBLIC_LIMIT_MB = 80;
const KEEP_BUILDS = 2;

// ---- Pins --------------------------------------------------------------------
const PYODIDE_VERSION = "0.29.3";
const PYODIDE_BASE = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const PYODIDE_LOCK_SHA256 = "3256ffc76388de0e37f4b34d42ab484268d1afc675179ff97b2a5bb14f84ccac";
// Pyodide's own core files; the worker checks each against these, from the
// network and from Cache Storage alike.
const PYODIDE_CORE = {
  "pyodide.mjs": "23213d27e9dacbda6f25f955fb38dabfc40ada082c04ee253047ea5dddcaa3b5",
  "pyodide.asm.js": "1263f02b5b26099b96112378156f242dd98b39a8201ba7765e5fe3d455c5ce91",
  "pyodide.asm.wasm": "e2f4ee75b325e35eb31bfb8c613d4dd5098f5502c156a97847686875b5025480",
  "python_stdlib.zip": "4298b6ee445cb724c3973437da47789752b9e6ff4e26619026b283ec801fc46b",
  "pyodide-lock.json": PYODIDE_LOCK_SHA256,
};
// Pyodide's packages the engine imports (their dependencies come along).
const PYODIDE_PACKAGES = [
  "pymupdf", "lxml", "numpy", "pillow", "pydantic", "fastapi", "httpx",
  "packaging", "six", "typing-extensions", "typing-inspection",
];

// The native engine's versions (services/engine/requirements.txt), except
// python-bidi: 0.6 is a Rust extension, 0.4.2 is pure Python.
const WHEELS = [
  { req: "python-dotenv==1.2.3", file: "python_dotenv-1.2.3-py3-none-any.whl", sha256: "904552145e8bfed22162c09dab1c2b9b54fefa7b23ba780f4f26ca0316b0f0d9" },
  { req: "pydantic-settings==2.7.0", file: "pydantic_settings-2.7.0-py3-none-any.whl", sha256: "e00c05d5fa6cbbb227c84bd7487c5c1065084119b750df7c8c1a554aed236eb5" },
  { req: "python-multipart==0.0.20", file: "python_multipart-0.0.20-py3-none-any.whl", sha256: "8a62d3a8335e06589fe01f2a3e178cdcc632f3fbe0d492ad9ee0ec35aab1f104" },
  { req: "pypdf==5.1.0", file: "pypdf-5.1.0-py3-none-any.whl", sha256: "3bd4f503f4ebc58bae40d81e81a9176c400cbbac2ba2d877367595fb524dfdfc" },
  { req: "python-docx==1.1.2", file: "python_docx-1.1.2-py3-none-any.whl", sha256: "08c20d6058916fb19853fcf080f7f42b6270d89eac9fa5f8c15f691c0017fabe" },
  { req: "XlsxWriter==3.2.9", file: "xlsxwriter-3.2.9-py3-none-any.whl", sha256: "9a5db42bc5dff014806c58a20b9eae7322a134abb6fce3c92c181bfb275ec5b3" },
  { req: "python-pptx==1.0.2", file: "python_pptx-1.0.2-py3-none-any.whl", sha256: "160838e0b8565a8b1f67947675886e9fea18aa5e795db7ae531606d68e785cba" },
  { req: "arabic-reshaper==3.0.1", file: "arabic_reshaper-3.0.1-py3-none-any.whl", sha256: "41c5adc2420f85758eada7e880251c4b6a2adbd83377bd27e5d4eba71f648bc7" },
  { req: "python-bidi==0.4.2", file: "python_bidi-0.4.2-py2.py3-none-any.whl", sha256: "50eef6f6a0bbdd685f9e8c207f3c9050f5b578d0a46e37c76a9c4baea2cc2e13" },
  { req: "pytesseract==0.3.13", file: "pytesseract-0.3.13-py3-none-any.whl", sha256: "7a99c6c2ac598360693d83a416e36e0b33a67638bb9d77fdcac094a3589d4b34" },
];
// langdetect ships only as a source archive; its wheel is built here with
// SOURCE_DATE_EPOCH, which makes it reproducible (same setuptools, same
// bytes). CT_ENGINE_ALLOW_LANGDETECT_DRIFT=1 accepts another setuptools' hash.
const LANGDETECT = {
  req: "langdetect==1.0.9",
  sdist: "langdetect-1.0.9.tar.gz",
  sdistSha256: "cbc1fef89f8d062739774bd51eda3da3274006b3661d199c2655f6b3f6d605a0",
  file: "langdetect-1.0.9-py3-none-any.whl",
  sha256: "df33a36e4e6f940eb7966403d3346fc58ea981f1da60e6f19213bee08d06121d",
  epoch: "1620345600",
};
// Unpacked in this order in the worker (dependencies first; no resolver there).
const INSTALL_ORDER = [
  "python_dotenv", "pydantic_settings", "python_multipart", "pypdf", "python_docx",
  "xlsxwriter", "python_pptx", "arabic_reshaper", "python_bidi", "pytesseract", "langdetect",
];

const TESS_CORE = {
  url: "https://registry.npmjs.org/tesseract.js-core/-/tesseract.js-core-7.0.0.tgz",
  sha256: "ba584355515eaff877552022853c0e71f2cb70466e759d1d6940484929718ee0",
  variants: ["relaxedsimd-lstm", "simd-lstm"],
};
const TESSDATA = {
  eng: {
    url: "https://cdn.jsdelivr.net/gh/tesseract-ocr/tessdata_fast@4.1.0/eng.traineddata",
    sha256: "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
  },
};

// What the worker sets before ct_engine is first imported. CT_RELAY_URL is
// added there (it needs the page's origin).
const ENGINE_ENV = {
  CT_PAGE_CONCURRENCY: "1",
  CT_MAX_JOB_CONCURRENCY: "1",
  CT_STORAGE_ROOT: "/ct",
  CT_OCR_PROVIDER: "browser",
  CT_TRANSLATION_PROVIDER: "deepl,google",
  CT_LOG_LEVEL: "WARNING",
  // Production: errors carry the teacher's sentence, never a traceback.
  CT_ENVIRONMENT: "production",
};

// ---- Helpers -------------------------------------------------------------------
const sha256 = (buf) => createHash("sha256").update(buf).digest("hex");
const mb = (n) => (n / 1e6).toFixed(2) + " MB";
function check(label, buf, want) {
  const got = sha256(buf);
  if (want && got !== want) throw new Error(`sha256 mismatch for ${label}\n  want ${want}\n  got  ${got}`);
  return got;
}
async function download(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`GET ${url} -> ${r.status}`);
  return Buffer.from(await r.arrayBuffer());
}
function pip(args, env = {}) {
  execFileSync(PY, ["-m", "pip", "--disable-pip-version-check", "-q", ...args], {
    stdio: ["ignore", "inherit", "inherit"],
    env: { ...process.env, PIP_NO_INPUT: "1", ...env },
  });
}
function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir).sort()) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) out.push(...walk(p));
    else out.push({ path: p, size: st.size });
  }
  return out;
}

// Deterministic zip: sorted entries, fixed 1980-01-01 timestamps, deflate 9.
function deterministicZip(entries) {
  const chunks = [];
  const central = [];
  let offset = 0;
  for (const { name, data } of entries) {
    const nameBuf = Buffer.from(name, "utf8");
    const deflated = zlib.deflateRawSync(data, { level: 9 });
    const stored = deflated.length >= data.length;
    const body = stored ? data : deflated;
    const method = stored ? 0 : 8;
    const crc = zlib.crc32(data) >>> 0;
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0x0800, 6); // UTF-8 names
    local.writeUInt16LE(method, 8);
    local.writeUInt16LE(0, 10); // time 00:00
    local.writeUInt16LE(0x21, 12); // date 1980-01-01
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(body.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(nameBuf.length, 26);
    local.writeUInt16LE(0, 28);
    chunks.push(local, nameBuf, body);
    const cen = Buffer.alloc(46);
    cen.writeUInt32LE(0x02014b50, 0);
    cen.writeUInt16LE(0x0314, 4); // made by: unix, 2.0
    cen.writeUInt16LE(20, 6);
    cen.writeUInt16LE(0x0800, 8);
    cen.writeUInt16LE(method, 10);
    cen.writeUInt16LE(0, 12);
    cen.writeUInt16LE(0x21, 14);
    cen.writeUInt32LE(crc, 16);
    cen.writeUInt32LE(body.length, 20);
    cen.writeUInt32LE(data.length, 24);
    cen.writeUInt16LE(nameBuf.length, 28);
    cen.writeUInt32LE((0o100644 << 16) >>> 0, 38); // -rw-r--r--
    cen.writeUInt32LE(offset, 42);
    central.push(cen, nameBuf);
    offset += 30 + nameBuf.length + body.length;
  }
  const cenBuf = Buffer.concat(central);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(cenBuf.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...chunks, cenBuf, end]);
}

// ---- Build ---------------------------------------------------------------------
const work = mkdtempSync(join(tmpdir(), "ct-engine-build-"));
const stage = join(work, "out");
const pins = [];
try {
  console.log(`engine build\n  python: ${PY}\n  out:    ${OUT_ROOT}/<hash>\n  work:   ${work}`);
  mkdirSync(join(stage, "wheels"), { recursive: true });
  mkdirSync(join(stage, "tess"), { recursive: true });

  // 1. Pure wheels, no resolver: the closure is checked in step 3.
  const wdir = join(work, "wheels");
  pip(["download", "--no-deps", "--only-binary=:all:", "--python-version", "3.13",
    "--implementation", "py", "--abi", "none", "--platform", "any", "-d", wdir,
    ...WHEELS.map((w) => w.req)]);
  for (const w of WHEELS) {
    const buf = readFileSync(join(wdir, w.file));
    check(w.file, buf, w.sha256);
    copyFileSync(join(wdir, w.file), join(stage, "wheels", w.file));
    pins.push([w.file, w.sha256, buf.length]);
  }

  // 2. langdetect, built from its pinned source archive.
  const sdir = join(work, "sdist");
  pip(["download", "--no-deps", "--no-binary=:all:", "-d", sdir, LANGDETECT.req]);
  check(LANGDETECT.sdist, readFileSync(join(sdir, LANGDETECT.sdist)), LANGDETECT.sdistSha256);
  const bdir = join(work, "built");
  pip(["wheel", "--no-deps", "--no-build-isolation", "-w", bdir, join(sdir, LANGDETECT.sdist)],
    { SOURCE_DATE_EPOCH: LANGDETECT.epoch });
  const ld = readFileSync(join(bdir, LANGDETECT.file));
  const ldSha = sha256(ld);
  if (ldSha !== LANGDETECT.sha256) {
    if (process.env.CT_ENGINE_ALLOW_LANGDETECT_DRIFT !== "1") check(LANGDETECT.file, ld, LANGDETECT.sha256);
    console.warn(`WARNING langdetect wheel hash drifted: ${ldSha}`);
  }
  copyFileSync(join(bdir, LANGDETECT.file), join(stage, "wheels", LANGDETECT.file));
  pins.push([LANGDETECT.file + " (built)", ldSha, ld.length]);

  // 3. Dependency closure: every Requires-Dist of every wheel we ship is met
  //    by another shipped wheel or by a package in Pyodide's own lock.
  const lockBuf = await download(PYODIDE_BASE + "pyodide-lock.json");
  check("pyodide-lock.json", lockBuf, PYODIDE_LOCK_SHA256);
  const lockPath = join(work, "pyodide-lock.json");
  writeFileSync(lockPath, lockBuf);
  const closurePy = join(work, "closure.py");
  writeFileSync(closurePy, `
import json, sys, zipfile, email.parser, glob, os
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version
lock = json.load(open(sys.argv[1]))["packages"]
have = {canonicalize_name(v["name"]): v["version"] for v in lock.values()}
src = {}
for whl in sorted(glob.glob(os.path.join(sys.argv[2], "*.whl"))):
    with zipfile.ZipFile(whl) as z:
        meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        m = email.parser.Parser().parsestr(z.read(meta).decode())
    name = canonicalize_name(m["Name"]); have[name] = m["Version"]; src[name] = m.get_all("Requires-Dist") or []
env = {"python_version": "3.13", "python_full_version": "3.13.2", "sys_platform": "emscripten",
       "platform_system": "Emscripten", "os_name": "posix", "platform_machine": "wasm32",
       "implementation_name": "cpython", "platform_python_implementation": "CPython", "extra": ""}
bad = []
for name, reqs in sorted(src.items()):
    for r in reqs:
        req = Requirement(r)
        if req.marker and not req.marker.evaluate(env):
            continue
        got = have.get(canonicalize_name(req.name))
        ok = got is not None and (not req.specifier or req.specifier.contains(Version(got), prereleases=True))
        if not ok: bad.append(f"{name}: {req} (have {got})")
if bad:
    print("UNMET:", bad); sys.exit(1)
print(f"dependency closure: {len(src)} shipped wheels, every requirement met")
`);
  execFileSync(PY, ["-I", closurePy, lockPath, join(stage, "wheels")], { stdio: "inherit" });

  // 3b. Pyodide's core files and the packages the worker loads: downloaded
  //     here only to prove the pins and to know a first visit's byte count
  //     (the progress bar's "of 35 MB"). Not copied.
  const lock = JSON.parse(lockBuf.toString("utf8")).packages;
  let pyodideBytes = lockBuf.length;
  for (const [f, want] of Object.entries(PYODIDE_CORE)) {
    if (f === "pyodide-lock.json") continue;
    const buf = await download(PYODIDE_BASE + f);
    check(f, buf, want);
    pyodideBytes += buf.length;
  }
  const closure = new Set();
  const visit = (name) => {
    const key = Object.keys(lock).find((k) => k.toLowerCase() === name.toLowerCase()
      || lock[k].name.toLowerCase() === name.toLowerCase());
    if (!key) throw new Error(`Pyodide ${PYODIDE_VERSION} has no package ${name}`);
    if (closure.has(key)) return;
    closure.add(key);
    for (const dep of lock[key].depends || []) visit(dep);
  };
  PYODIDE_PACKAGES.forEach(visit);
  for (const key of [...closure].sort()) {
    const p = lock[key];
    const buf = await download(PYODIDE_BASE + p.file_name);
    check(p.file_name, buf, p.sha256);
    pyodideBytes += buf.length;
  }
  console.log(`Pyodide ${PYODIDE_VERSION}: ${closure.size} packages + core, ${mb(pyodideBytes)} (all pins verified)`);

  // 3c. The engine's own pins, so differences from Pyodide's versions are on record.
  const enginePins = {};
  for (const line of readFileSync(join(ENGINE_ROOT, "requirements.txt"), "utf8").split("\n")) {
    const m = line.trim().match(/^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==([^\s#;]+)/);
    if (m) enginePins[m[1]] = m[2];
  }

  // 4. tesseract.js-core 7.0.0 and the language packs.
  const tgz = await download(TESS_CORE.url);
  check("tesseract.js-core-7.0.0.tgz", tgz, TESS_CORE.sha256);
  const tdir = join(work, "tess");
  mkdirSync(tdir);
  writeFileSync(join(tdir, "core.tgz"), tgz);
  execFileSync("tar", ["-xzf", join(tdir, "core.tgz"), "-C", tdir]);
  for (const v of TESS_CORE.variants) {
    for (const ext of [".js", ".wasm"]) {
      copyFileSync(join(tdir, "package", `tesseract-core-${v}${ext}`), join(stage, "tess", `tesseract-core-${v}${ext}`));
    }
  }
  copyFileSync(join(tdir, "package", "LICENSE"), join(stage, "tess", "LICENSE"));
  for (const [lang, t] of Object.entries(TESSDATA)) {
    const buf = await download(t.url);
    check(`${lang}.traineddata`, buf, t.sha256);
    writeFileSync(join(stage, "tess", `${lang}.traineddata`), buf);
  }

  // 5. Fonts: the engine's faces must be byte-identical to /fonts/v1, which
  //    the worker fetches instead of shipping them twice.
  const facesDir = join(ENGINE_ROOT, "ct_engine", "fonts", "faces");
  const fonts = [];
  for (const { path } of walk(facesDir)) {
    if (!/\.(ttf|otf|ttc)$/i.test(path)) continue;
    const rel = relative(facesDir, path).split(sep).join("/");
    const web = join(FONTS_DIR, ...rel.split("/"));
    if (!existsSync(web)) throw new Error(`font ${rel} is in the engine but not in public/fonts/v1`);
    const a = readFileSync(path);
    if (sha256(a) !== sha256(readFileSync(web))) throw new Error(`font ${rel} differs between the engine and public/fonts/v1`);
    fonts.push({ url: `/fonts/v1/${rel}`, dest: rel, bytes: a.length, sha256: sha256(a) });
  }

  // 6. Engine source. Fonts' .ttf files, caches, tests, corpus and data
  //    are left out; catalog.json stays. Any .env stops the build.
  const SKIP_DIRS = new Set(["__pycache__", "tests", "corpus", "data", ".pytest_cache", "evaluation"]);
  const files = [];
  const collect = (dir) => {
    for (const name of readdirSync(dir).sort()) {
      const p = join(dir, name);
      const st = statSync(p);
      if (st.isDirectory()) { if (!SKIP_DIRS.has(name)) collect(p); continue; }
      if (name.startsWith(".env")) throw new Error(`refusing to package ${p}`);
      if (name === ".DS_Store" || name.endsWith(".pyc")) continue;
      if (/\.(ttf|otf|ttc)$/i.test(name)) continue;
      files.push(p);
    }
  };
  for (const pkg of PACKAGES) collect(join(ENGINE_ROOT, pkg));
  const entries = files
    .map((p) => ({ name: relative(ENGINE_ROOT, p).split(sep).join("/"), data: readFileSync(p) }))
    .sort((a, b) => (a.name < b.name ? -1 : 1));
  for (const must of ["ct_engine/fonts/faces/catalog.json", "ct_browser/boot.py", "ct_browser/ocr.py", "ct_browser/executor.py"]) {
    if (!entries.some((e) => e.name === must)) throw new Error(`${must} missing from engine.zip`);
  }
  if (entries.some((e) => /(^|\/)(tests|corpus|data)\//.test(e.name))) throw new Error("tests, corpus or data would be packaged");
  // The evaluation package is a developer tool the API never imports; left
  // out on purpose, and checked so a future import fails here, not in a tab.
  const imports = entries.filter((e) => e.name.endsWith(".py") && /(from|import)\s+[.\w]*evaluation/.test(e.data.toString("utf8")));
  if (imports.length) throw new Error(`these import ct_engine.evaluation: ${imports.map((e) => e.name).join(", ")}`);
  const zip = deterministicZip(entries);
  writeFileSync(join(stage, "engine.zip"), zip);
  console.log(`engine.zip: ${entries.length} files, ${mb(zip.length)}`);

  // 7. The worker and the text reader's bridge.
  for (const f of ["engine.worker.js", "tesseract-sync.js"]) copyFileSync(join(SRC_ENGINE, f), join(stage, f));

  // 8. What the host answers without Python, from the engine's own code
  //    (the same lists and limits its routes return), with the browser's env.
  const answersPy = join(work, "answers.py");
  writeFileSync(answersPy, `
import json, os, sys
sys.path.insert(0, sys.argv[1])
os.environ.update(json.loads(sys.argv[2]))
from ct_engine.config import get_settings
from ct_engine.api.routes import languages, output_modes
from ct_engine.ingest.registry import SUPPORTED_EXTENSIONS
s = get_settings()
health = {
    "status": "ok", "environment": s.environment, "storage_backend": s.storage_backend,
    "translation_provider": s.translation_provider, "ocr_provider": s.ocr_provider,
    "retention_hours": s.retention_hours, "max_page_count": s.max_page_count,
    "max_upload_mb": s.max_upload_bytes // (1024 * 1024), "access_required": False,
    "supported_formats": sorted(SUPPORTED_EXTENSIONS | {"png", "jpg", "jpeg"}), "memory": None,
}
print(json.dumps({"health": health, "languages": languages(), "output_modes": output_modes()}))
`);
  const answers = JSON.parse(execFileSync(PY, ["-I", answersPy, ENGINE_ROOT,
    JSON.stringify({ ...ENGINE_ENV, CT_STORAGE_ROOT: join(work, "ct") })], { encoding: "utf8" }));

  // 9. Manifest and hash. The hash covers every file the worker loads.
  const wheelFiles = INSTALL_ORDER.map((stem) => {
    const f = readdirSync(join(stage, "wheels")).find((x) => x.toLowerCase().startsWith(stem + "-"));
    if (!f) throw new Error(`no wheel for ${stem}`);
    return "wheels/" + f;
  });
  const payload = ["engine.zip", "engine.worker.js", "tesseract-sync.js", ...wheelFiles,
    ...TESS_CORE.variants.flatMap((v) => [`tess/tesseract-core-${v}.js`, `tess/tesseract-core-${v}.wasm`]),
    ...Object.keys(TESSDATA).map((l) => `tess/${l}.traineddata`)];
  const filesMeta = {};
  for (const p of payload) {
    const buf = readFileSync(join(stage, p));
    filesMeta[p] = { bytes: buf.length, sha256: sha256(buf) };
  }
  const hash = sha256(Buffer.from(JSON.stringify({ filesMeta, fonts, PYODIDE_CORE, ENGINE_ENV }))).slice(0, 12);
  const sum = (ps) => ps.reduce((n, p) => n + filesMeta[p].bytes, 0);
  const engineBytes = sum(["engine.zip", ...wheelFiles]) + fonts.reduce((n, f) => n + f.bytes, 0);
  const tessVariantBytes = Math.max(...TESS_CORE.variants.map((v) => sum([`tess/tesseract-core-${v}.js`, `tess/tesseract-core-${v}.wasm`])));
  const manifest = {
    generatedBy: "apps/web/scripts/build-engine.mjs",
    hash,
    pyodide: {
      version: PYODIDE_VERSION, indexURL: PYODIDE_BASE, core: PYODIDE_CORE,
      packages: PYODIDE_PACKAGES, bytes: pyodideBytes,
    },
    env: ENGINE_ENV,
    enginePins,
    wheels: wheelFiles,
    fonts,
    tess: {
      variants: Object.fromEntries(TESS_CORE.variants.map((v) => [v, `tess/tesseract-core-${v}`])),
      langs: Object.fromEntries(Object.keys(TESSDATA).map((l) => [l, `tess/${l}.traineddata`])),
      coreBytes: tessVariantBytes,
    },
    // What a first visit downloads before the engine answers, and what the
    // text reader adds (core + English) the first time it is needed.
    bootBytes: pyodideBytes + engineBytes,
    readerBytes: tessVariantBytes + filesMeta["tess/eng.traineddata"].bytes,
    files: filesMeta,
  };
  writeFileSync(join(stage, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");

  // 10. Into place: public/engine/<hash>/, keeping the newest builds only.
  mkdirSync(OUT_ROOT, { recursive: true });
  const dest = join(OUT_ROOT, hash);
  rmSync(dest, { recursive: true, force: true });
  try {
    renameSync(stage, dest);
  } catch (e) {
    if (e.code !== "EXDEV") throw e;
    cpSync(stage, dest, { recursive: true });
  }
  const builds = readdirSync(OUT_ROOT)
    .filter((d) => /^[0-9a-f]{12}$/.test(d) && statSync(join(OUT_ROOT, d)).isDirectory())
    .map((d) => ({ d, t: d === hash ? Infinity : statSync(join(OUT_ROOT, d, "manifest.json")).mtimeMs }))
    .sort((a, b) => b.t - a.t);
  for (const { d } of builds.slice(KEEP_BUILDS)) {
    rmSync(join(OUT_ROOT, d), { recursive: true, force: true });
    console.log(`removed old build ${d}`);
  }

  const generated = `// GENERATED by apps/web/scripts/build-engine.mjs -- do not edit.
// The browser engine build this app uses, and what the engine host answers
// without starting Python (the engine's own lists and limits, with the
// browser's settings).

export const ENGINE_HASH = ${JSON.stringify(hash)};
export const ENGINE_BASE = ${JSON.stringify(`/engine/${hash}/`)};
export const ENGINE_BOOT_BYTES = ${manifest.bootBytes};
export const ENGINE_READER_BYTES = ${manifest.readerBytes};

export const ENGINE_HEALTH: Record<string, unknown> = ${JSON.stringify(answers.health, null, 2)};

export const ENGINE_LANGUAGES = ${JSON.stringify(answers.languages)};

export const ENGINE_OUTPUT_MODES = ${JSON.stringify(answers.output_modes)};
`;
  writeFileSync(GENERATED, generated);

  console.log("\npins verified:");
  for (const [n, h, b] of pins) console.log(`  ${h.slice(0, 16)}…  ${n}  (${mb(b)})`);
  console.log(`\nbuild ${hash}: first visit ${mb(manifest.bootBytes)} (Pyodide ${mb(pyodideBytes)}), text reader ${mb(manifest.readerBytes)}`);
  const total = walk(join(WEB, "public")).reduce((s, f) => s + f.size, 0);
  console.log(`public/ total: ${mb(total)} (limit ${PUBLIC_LIMIT_MB} MB)`);
  if (total > PUBLIC_LIMIT_MB * 1e6) throw new Error("public/ is over the size limit");
} finally {
  rmSync(work, { recursive: true, force: true });
}
