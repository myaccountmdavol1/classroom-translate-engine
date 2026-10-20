# Third-party notices

The Classroom Translate document engine (this repository) is licensed under
the GNU Affero General Public License v3 (`LICENSE`). It is distributed
together with, or loads at run time, the third-party components below. Each
keeps its own licence. Versions are the ones pinned for the browser build
in `apps/web/scripts/build-engine.mjs` and recorded in
`builds/<hash>.manifest.json`; the native build uses
`services/engine/requirements.txt`.

## In the browser build

### Loaded from jsDelivr by the worker (checked against Pyodide's lock file)

| Component | Version | Licence | Source |
|---|---|---|---|
| Pyodide (CPython 3.13 for WebAssembly) | 0.29.3 | MPL-2.0 (CPython: PSF License) | https://github.com/pyodide/pyodide/tree/0.29.3 , release archive https://github.com/pyodide/pyodide/releases/tag/0.29.3 |
| PyMuPDF | 1.26.3 | AGPL-3.0 (or Artifex commercial licence) | https://github.com/pymupdf/PyMuPDF/tree/1.26.3 , sdist https://pypi.org/project/PyMuPDF/1.26.3/#files |
| MuPDF (built into PyMuPDF 1.26.3) | 1.26.x as pinned by PyMuPDF 1.26.3 | AGPL-3.0 (or Artifex commercial licence) | https://mupdf.com/releases , https://github.com/ArtifexSoftware/mupdf ; the Pyodide build recipe is at https://github.com/pyodide/pyodide/tree/0.29.3/packages/pymupdf |
| lxml (with libxml2, libxslt) | 6.0.2 | BSD-3-Clause (libxml2, libxslt: MIT) | https://github.com/lxml/lxml |
| NumPy | 2.2.5 | BSD-3-Clause | https://github.com/numpy/numpy |
| Pillow | 11.3.0 | MIT-CMU (HPND) | https://github.com/python-pillow/Pillow |
| pydantic | 2.12.5 | MIT | https://github.com/pydantic/pydantic |
| pydantic-core | 2.41.5 | MIT | https://github.com/pydantic/pydantic-core |
| FastAPI | 0.116.1 | MIT | https://github.com/fastapi/fastapi |
| Starlette | 0.47.2 | BSD-3-Clause | https://github.com/encode/starlette |
| HTTPX | 0.28.1 | BSD-3-Clause | https://github.com/encode/httpx |
| AnyIO | 4.9.0 | MIT | https://github.com/agronholm/anyio |
| sniffio | 1.3.1 | MIT or Apache-2.0 | https://github.com/python-trio/sniffio |
| Jinja2 | 3.1.6 | BSD-3-Clause | https://github.com/pallets/jinja |
| MarkupSafe | 3.0.2 | BSD-3-Clause | https://github.com/pallets/markupsafe |
| annotated-types | 0.7.0 | MIT | https://github.com/annotated-types/annotated-types |
| packaging | 24.2 | Apache-2.0 or BSD-2-Clause | https://github.com/pypa/packaging |
| six | 1.17.0 | MIT | https://github.com/benjaminp/six |
| typing-extensions | 4.15.0 | PSF-2.0 | https://github.com/python/typing_extensions |
| typing-inspection | 0.4.2 | MIT | https://github.com/pydantic/typing-inspection |
| OpenSSL (libopenssl, ssl module) | 1.1.1w | OpenSSL / SSLeay License | https://www.openssl.org/source/old/1.1.1/ |

MuPDF includes third-party libraries (among them FreeType, HarfBuzz,
jbig2dec, lcms2, libjpeg, OpenJPEG, zlib, Gumbo, Extract and MuJS) under
their own permissive licences; see the `thirdparty/` directory of the MuPDF
source.

### Served with each build (`/engine/<hash>/wheels/`), pure-Python wheels

Each wheel carries its licence file in its `.dist-info` folder.

| Component | Version | Licence | Source |
|---|---|---|---|
| python-bidi | 0.4.2 | **LGPL-3.0** (GNU Lesser General Public License v3; `COPYING` and `COPYING.LESSER` in the wheel) | https://github.com/MeirKriheli/python-bidi/tree/v0.4.2 , https://pypi.org/project/python-bidi/0.4.2/#files |
| arabic-reshaper | 3.0.1 | MIT | https://github.com/mpcabd/python-arabic-reshaper |
| langdetect | 1.0.9 | Apache-2.0 (with the NOTICE of Cybozu Labs' language-detection, shipped in the wheel) | https://github.com/Mimino666/langdetect , https://pypi.org/project/langdetect/1.0.9/#files |
| pypdf | 5.1.0 | BSD-3-Clause | https://github.com/py-pdf/pypdf |
| python-docx | 1.1.2 | MIT | https://github.com/python-openxml/python-docx |
| python-pptx | 1.0.2 | MIT | https://github.com/scanny/python-pptx |
| XlsxWriter | 3.2.9 | BSD-2-Clause | https://github.com/jmcnamara/XlsxWriter |
| pydantic-settings | 2.7.0 | MIT | https://github.com/pydantic/pydantic-settings |
| python-dotenv | 1.2.3 | BSD-3-Clause | https://github.com/theskumar/python-dotenv |
| python-multipart | 0.0.20 | Apache-2.0 | https://github.com/Kludex/python-multipart |
| pytesseract | 0.3.13 | Apache-2.0 | https://github.com/madmaze/pytesseract |

python-bidi is used unmodified and loaded as a separate wheel, so it can be
replaced with another build of the same version: swap the file in
`wheels/` and its sha256 in `build-engine.mjs`.

langdetect's NOTICE:

> language-detection license
>
> Copyright (c) 2010-2014 Cybozu Labs, Inc. All rights reserved.
>
> Licensed under the Apache License, Version 2.0.

### Served with each build (`/engine/<hash>/tess/`), the text reader

| Component | Version | Licence | Source |
|---|---|---|---|
| tesseract.js-core (Tesseract 5 and Leptonica compiled to WebAssembly) | 7.0.0 | Apache-2.0; the package's `LICENSE` is served next to it as `tess/LICENSE` | https://github.com/naptha/tesseract.js-core/tree/v7.0.0 , https://registry.npmjs.org/tesseract.js-core/-/tesseract.js-core-7.0.0.tgz |
| Tesseract OCR | 5.x as built by tesseract.js-core 7.0.0 | Apache-2.0 | https://github.com/tesseract-ocr/tesseract |
| Leptonica | as built by tesseract.js-core 7.0.0 | BSD-2-Clause style (Leptonica licence) | https://github.com/DanBloomberg/leptonica |
| tessdata_fast `eng.traineddata` | 4.1.0 | Apache-2.0 | https://github.com/tesseract-ocr/tessdata_fast/tree/4.1.0 |

The calls into the Tesseract core in `apps/web/src/engine/tesseract-sync.js`
follow tesseract.js's worker script, Copyright Project Naptha and the
tesseract.js contributors, Apache-2.0 (https://github.com/naptha/tesseract.js).

tesseract.js-core 7.0.0 ships no separate NOTICE file; its Apache-2.0
`LICENSE` is the notice.

## Fonts (in this repository and in both builds)

Bundled in `services/engine/ct_engine/fonts/faces/` (and served to the
browser from the website's `/fonts/v1/`), each folder with its `OFL.txt`.
All are licensed under the SIL Open Font License, Version 1.1.

| Folder | Fonts | Copyright |
|---|---|---|
| `andika/` | Andika | Copyright (c) 2004-2022 SIL International |
| `comic-neue/` | Comic Neue | Copyright 2014 The Comic Neue Project Authors (https://github.com/crozynski/comicneue) |
| `lexend/` | Lexend | Copyright 2018 The Lexend Project Authors (https://github.com/googlefonts/lexend), Reserved Font Name "RevReading Lexend" |
| `marks/` | Noto Sans Symbols 2 | Copyright 2022 The Noto Project Authors (https://github.com/notofonts/symbols) |
| `noto-sans/` | Noto Sans | Copyright 2022 The Noto Project Authors (https://github.com/notofonts/latin-greek-cyrillic) |
| `noto-serif/` | Noto Serif | Copyright 2022 The Noto Project Authors (https://github.com/notofonts/latin-greek-cyrillic) |
| `open-sans/` | Open Sans | Copyright 2020 The Open Sans Project Authors (https://github.com/googlefonts/opensans) |

## Native build only

`services/engine/requirements.txt` additionally pins, among others,
PyMuPDF 1.25.1 (AGPL-3.0), pdfplumber (MIT), uvicorn (BSD-3-Clause),
reportlab (BSD, tests only) and pytest (MIT, tests only); the optional
`requirements-local-translation.txt` adds Argos Translate (MIT) and
python-bidi 0.6.11 (LGPL-3.0). The `Dockerfile` installs the Tesseract
binary and language data (Apache-2.0), the Noto and DejaVu fonts (OFL /
Bitstream Vera licence) and LibreOffice (MPL-2.0) from Debian packages;
they run as separate programs and are not part of this repository.
