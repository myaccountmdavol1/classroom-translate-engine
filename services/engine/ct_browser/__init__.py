"""Glue that runs the unchanged ct_engine somewhere other than Render.

ct_engine never imports this package, and the Render image never contains
it (its Dockerfile copies only ct_engine), so nothing here can change what
teachers get from Render. Everything is applied from outside, by an explicit
call, never as a side effect of importing:

* relay.py -- DeepL and Google through the web app's translation relay
  (/api/mt/*), so no key is needed where the engine runs. ``relay.install()``
  puts them in place of the engine's own DeepL and Google providers.
* executor.py -- the job pool for an engine with no threads: submit()
  records the work, and the browser worker's queue runs it later.
* ocr.py -- Tesseract (tesseract.js-core) called synchronously in the
  worker; ``ocr.install()`` registers it as the 'browser' OCR provider.
* cancel.py -- lets Stop reach a running job: its checkpoints also ask
  the page, since the request cannot reach Python while the job runs.
* journal.py -- tells the page every write to the engine's store, in order,
  so the page can keep the lasting copy in the browser's own storage (OPFS).
* boot.py -- ``start()`` applies every browser shim (and calls the two
  install()s), then serves requests (``handle``), runs jobs
  (``run_next_job``) and picks interrupted jobs up again (``resume``), one
  Python entry at a time.

The browser worker's boot (Phase 2 of docs/superpowers/plans/
2026-10-19-browser-engine-pdf.md) calls these; so does the native harness
scripts/run_packet_via_relay.py.
"""
