# Classroom Translate document engine

This is the document engine of Classroom Translate, a tool that helps a
teacher translate worksheets, packets, slide decks and Word files for
students and families while keeping the page layout. The engine reads a
document, finds its text (including text in pictures and scanned pages),
sends the text for translation, and rebuilds the pages with the translated
text in place. It also runs the page editor's changes.

The engine runs in two ways:

- **In the teacher's browser.** The unchanged Python engine runs in
  [Pyodide](https://pyodide.org) 0.29.3 inside a Web Worker, with
  [PyMuPDF](https://github.com/pymupdf/PyMuPDF) 1.26.3 (MuPDF) for PDF work
  and tesseract.js-core 7.0.0 (Tesseract 5) for reading text in pictures.
- **Natively**, as a FastAPI service in a container (the `Dockerfile`).

The Classroom Translate website itself (the Next.js app) is a separate,
private program. It talks to this engine through messages (in the browser)
or HTTP (natively). This repository holds the engine, the browser glue that
starts and feeds it, and the script that builds the browser bundle.

## Licence and source offer

This engine is licensed under the **GNU Affero General Public License,
version 3** (see `LICENSE`), because it is combined with PyMuPDF/MuPDF,
which are AGPL-3.0. Third-party components keep their own licences; see
`THIRD_PARTY_NOTICES.md`.

**Source offer.** If you use Classroom Translate, the browser downloads a
build of this engine. Each build is named by a content hash (for example
`3c60062ce00b`, served from `/engine/3c60062ce00b/`), and this repository
has a git tag `engine-<hash>` for every build deployed to users. That tag is
the Corresponding Source for that build: check it out and follow "Building
the browser bundle" below to reproduce it. The exact upstream sources of
the third-party components are linked from `THIRD_PARTY_NOTICES.md`. If you
cannot find the tag for a build you were served, open an issue on this
repository and the source for it will be provided.

`builds/<hash>.manifest.json` is the manifest of each deployed build: the
size and sha256 of every file the browser loads, and the versions of
everything pinned.

## What is here

```
services/engine/
  ct_engine/            the engine (Python package): reading, OCR, translation,
                        rebuilding pages, the editor's operations, the API
  ct_engine/fonts/faces the bundled fonts (all SIL OFL 1.1; licence in each folder)
  ct_browser/           what the engine needs in the browser: boot, the job
                        executor, the OPFS journal, OCR through tesseract.js,
                        translation through the website's relay, cancelling
  tests/                the test suite (pytest)
  corpus/               generators for the synthetic test documents
  scripts/              run_packet_via_relay.py, a native check of the relay path
  requirements*.txt     native dependencies
  Dockerfile, render.yaml, fly.toml   native deployment
apps/web/
  scripts/build-engine.mjs   builds the browser bundle (all pins live here)
  src/engine/engine.worker.js  the Web Worker that runs Pyodide and the engine
  src/engine/tesseract-sync.js the synchronous bridge to tesseract.js-core
  src/engine/host.ts, protocol.ts, opfs-store.ts, index-db.ts
                             the page-side host: starts the worker, answers the
                             website's API calls with it, keeps documents in OPFS
apps/web/src/lib/shapes.golden.json  shape outlines shared with the editor's tests
builds/                      manifests of deployed browser builds
```

The paths match the private repository, so `build-engine.mjs` runs here
unchanged. `host.ts` imports four modules of the private website
(`@/lib/api`, `@/lib/backup`, `@/lib/docsChannel`, `@/lib/leaveGuard`):
the website's API client and its tab bookkeeping. They are not part of the
engine and are not included, so `host.ts` is published for reading and is
not compiled on its own here.

## Running natively

Python 3.12. From `services/engine`:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                     # the test suite
CT_TRANSLATION_PROVIDER=deepl CT_DEEPL_API_KEY=... \
  .venv/bin/python -m uvicorn ct_engine.main:app --port 8000
```

Settings are environment variables starting with `CT_` (see
`ct_engine/config.py`); `render.yaml` and `fly.toml` show two working
deployments. Reading scanned pages needs the Tesseract binary
(`brew install tesseract tesseract-lang`, or `apt-get install tesseract-ocr`);
the `Dockerfile` installs it, the Noto fonts for scripts the bundled faces do
not cover, and LibreOffice for laying out Word and PowerPoint files.
Offline translation is optional: `pip install -r requirements-local-translation.txt`
and `CT_TRANSLATION_PROVIDER=local`.

The tests build a synthetic corpus of PDF, Word and PowerPoint files into
`corpus/out/` the first time they run (`corpus/build_*.py`). Six tests
skip here because what they compare against is not published: three in
`tests/test_tilted_text.py` (picture crops from a real deck), two in
`tests/test_browser_relay.py` (the website's translation relay) and one in
`tests/test_font_catalog.py` (the website's copy of the font catalogue).
`apps/web/src/lib/shapes.golden.json` is the shape outlines the engine and
the website's editor are both checked against
(`tests/test_shapes_geometry.py`).

## Building the browser bundle

Needs Node 22 or later, network access (PyPI, npm, jsDelivr) and the native
virtualenv above (the script uses its pip, and imports the engine once to
record its language list and limits).

```bash
# The worker loads the engine's fonts from the website's /fonts/v1 copy,
# and the build checks that copy matches the engine's faces byte for byte.
mkdir -p apps/web/public/fonts
cp -R services/engine/ct_engine/fonts/faces apps/web/public/fonts/v1

cd apps/web
node scripts/build-engine.mjs
```

The output is `apps/web/public/engine/<hash>/` (engine.zip, the wheels
Pyodide does not ship, tesseract.js-core and its LICENSE, tessdata_fast
English, the worker, manifest.json) and `apps/web/src/engine/manifest.generated.ts`.
Every downloaded file is checked against a sha256 pinned in the script, and
engine.zip is deterministic, so the same source gives the same hash. The
langdetect wheel is built from its source archive with `SOURCE_DATE_EPOCH`
(the recipe is in the script); a different setuptools may give a different
wheel hash, which `CT_ENGINE_ALLOW_LANGDETECT_DRIFT=1` accepts. Pyodide
itself and its packages (PyMuPDF among them) are loaded by the worker from
jsDelivr and checked against Pyodide's pinned lock file.

To serve it, a page must start `engine.worker.js` from that folder as a
module worker and speak the protocol in `src/engine/protocol.ts`; see the
comments at the top of `engine.worker.js`.

## Contributing

Issues and pull requests are welcome. By contributing you agree that your
contribution is licensed under the AGPL-3.0.
