# Document engine. Deploy to a container host (Fly.io, Render, Railway, Cloud
# Run, ECS) -- NOT to a serverless function: PyMuPDF plus a translation model
# exceeds serverless size limits, and a 100-page job outruns the timeout.
# The distro is pinned deliberately. `python:3.12-slim` follows Debian
# stable, and when that rolled from bookworm to trixie the font package
# list below stopped resolving and the build died -- a base image moving
# under us, with nothing in this repo having changed.
FROM python:3.12-slim-trixie

# Three system dependencies:
#   * Tesseract, for reading scanned pages. It is a binary, not a wheel, so
#     pip cannot supply it; without it scanned pages are carried through
#     unchanged and flagged.
#   * Fonts for scripts MuPDF does not bundle. Without these, Arabic, Hebrew
#     and Bengali cannot be rendered, and those pages are refused rather than
#     shipped with missing-character boxes. fonts-noto-core is the one that
#     matters: every face _PREFERRED names for those scripts, and for
#     Devanagari, lives in it.
#   * LibreOffice Writer and Impress, headless, to lay a Word file or a
#     slide deck out as pages so it opens in the page editor. ~240 MB at peak for a few seconds per
#     upload, one at a time; without it a Word file opens as a text list.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-spa \
        tesseract-ocr-fra \
        tesseract-ocr-por \
        tesseract-ocr-chi-sim \
        tesseract-ocr-chi-tra \
        tesseract-ocr-ara \
        tesseract-ocr-rus \
        tesseract-ocr-kor \
        tesseract-ocr-jpn \
        tesseract-ocr-ben \
        tesseract-ocr-urd \
        fonts-dejavu-core \
        fonts-noto-core \
        fonts-noto-cjk \
        fonts-beng \
        libreoffice-writer-nogui \
        libreoffice-impress-nogui \
    && apt-get clean \
    && find /var/lib/apt/lists -type f -delete

WORKDIR /app

COPY requirements.txt requirements-local-translation.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Offline translation is optional and large (~3 GB with torch). It is left
# out by default because the hosted route translates at a provider and has
# to fit a 512 MB free instance -- and because Render's Blueprints have no
# build-argument field at all, so a default is the only way to reach that
# build. Ask for it explicitly with
#   --build-arg LOCAL_TRANSLATION=1
# which is what fly.toml does, for the route where no text leaves the box.
ARG LOCAL_TRANSLATION=0
RUN if [ "$LOCAL_TRANSLATION" = "1" ]; then \
        pip install --no-cache-dir -r requirements-local-translation.txt; \
    fi

# The fonts a teacher can choose by name (Noto Sans and Serif, Open Sans,
# Andika, Lexend, Comic Neue, and Noto Sans Symbols 2 for list bullets) are
# in the package, under ct_engine/fonts/faces/, all OFL -- about 8 MB, read
# only when chosen. They have no Arabic, Bengali or CJK letters: text in
# those is still drawn in fonts-noto-core and fonts-noto-cjk above.
COPY ct_engine ./ct_engine

# Uploaded documents live here. Mount a volume so a restart does not discard
# in-flight jobs -- the pipeline is resumable, but only if state survives.
ENV CT_STORAGE_ROOT=/data
RUN mkdir -p /data

# Run as a non-root user: this service handles student work.
RUN useradd --create-home --uid 10001 engine && chown -R engine /data /app
USER engine

ENV CT_ENVIRONMENT=production \
    PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+p+'/api/health', timeout=4).status==200 else 1)"

# Bind whatever port the host asks for. Render assigns one through PORT and
# routes to that; Fly sets nothing and gets 8000, which is what fly.toml's
# internal_port expects. `exec` keeps uvicorn as PID 1 so a shutdown signal
# still reaches it and in-flight pages are written out.
CMD ["sh", "-c", "exec python -m uvicorn ct_engine.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
