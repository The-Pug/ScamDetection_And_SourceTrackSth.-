# TraceLens

AI media forensics web app — evidence intake, deepfake screening, source tracing,
and a tamper-evident chain of custody. Flask backend, plain HTML/CSS/JS frontend.

## Run it (Docker — recommended for the demo)

```
cd TraceLens
docker build -t tracelens .
docker run -p 8000:8000 -v "$(pwd)/instance:/app/instance" tracelens
```

Open http://127.0.0.1:8000. The AI model is baked into the image at build time, so
the container works offline at runtime — no download during the actual demo. The
`-v` volume mount is what makes case data, evidence files, and the custody log
survive container restarts/rebuilds — don't drop it. Runs on gunicorn with 4 threads,
so multiple people can use it at once without blocking each other. Any laptop with
Docker installed can run this identically — copy the repo over (or `git clone`) and
run the two commands above.

If `docker compose` is available, `docker compose up --build` does the same thing.

## Run it (without Docker)

```
cd TraceLens
uv venv --python 3.12          # first time only
uv pip install -r requirements.txt --python .venv/bin/python   # first time only
.venv/bin/python app.py
```

Open http://127.0.0.1:5000 — first AI analysis on a fresh machine will download the
classifier model from Hugging Face (~1-2 min), then it's cached locally and fast.
This runs Flask's single-threaded dev server, not gunicorn — fine for solo use,
but two people clicking around at once will block each other. Use Docker for the
actual demo.

## What's real vs. what's a prototype

- **Real**: SHA-256 fingerprinting, EXIF extraction, perceptual hashing, Error Level
  Analysis, a pretrained HF deepfake classifier (`prithivMLmods/deepfake-detector-model-v1`),
  OpenCV Haar face detection, PDF report export, and a SQLite-backed chain of custody
  where each event's hash incorporates the previous event's hash (tamper-evident —
  editing any past entry breaks the chain, checked live on the dashboard).
- **Real, and the actual differentiator**: perceptual-hash recurrence tracking — every
  uploaded image is checked against every other image ever seen (across cases), so a
  recycled/reposted image gets flagged automatically instead of requiring a manual
  hash lookup.
- **Real: whole-folder ingest.** Evidence intake takes a single file, a multi-select,
  or an entire folder (via "Select Folder" or literally dragging a folder from the
  OS file manager onto the page — it's walked recursively client-side). Every file
  in it runs the full pipeline automatically: hash, EXIF, ELA, AI screening, face
  detection, recurrence check, chain-of-custody logging — no per-file clicking. Files
  it can't handle (non-media junk that folders always contain) are skipped and shown
  as such, not silently dropped. Lands on a batch summary page for >1 file.
- **Manual entry, not automated**: Source Tracing is a form — you record a claimed
  source URL/ID/hash, it doesn't crawl or reverse-image-search the web. That's the
  honest scope for this pass.

AI output is an advisory screening signal, not forensic proof — every AI-related page
says this on purpose; keep that framing in the pitch.

## Layout

- `app.py` — Flask routes
- `forensics.py` — the actual detection/analysis logic (hashing, EXIF, ELA, AI model, PDF)
- `db.py` — SQLite persistence + chain-of-custody hashing + recurrence lookup
- `templates/`, `static/` — frontend
