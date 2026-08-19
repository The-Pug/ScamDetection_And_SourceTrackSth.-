# TraceLens

AI media forensics web app — evidence intake, deepfake screening, source tracing,
and a tamper-evident chain of custody. Built for PS4: AI Media Detection & Source
Tracing. Flask backend, SQLite, plain HTML/CSS/JS frontend.

## The problem this is actually solving

A pretrained deepfake classifier alone is a commodity — every rushed team has one,
and it's blind to anything outside its training domain (ours is tuned for
photographic human faces; point it at a screenshot or an animal and the score is
noise). So TraceLens treats that classifier as one weak, clearly-labeled signal
among several, and puts real weight on two things a classifier can't fake its way
around: **does this image already exist somewhere else** (reverse image search),
and **is a claimed time/place physically consistent with the photo** (real sun
position astronomy, not a guess). Every AI-derived number on every page carries an
explicit "advisory signal, not proof" caveat — that's not boilerplate, it's load-bearing.

## How it works

**Integrity & chain of custody** — SHA-256 fingerprint on intake, and a SQLite-backed
custody log where each event's hash incorporates the previous event's hash. Edit any
past entry and the chain breaks visibly (checked live on the dashboard) — this is
what makes the log tamper-evident rather than just a list of timestamps.

**Pixel forensics** — Error Level Analysis (recompression artifacts), a
region-by-region noise-consistency check (splits the image into a grid, flags
regions whose local noise level is a sharp outlier vs. the image's own median —
catches spliced-in or regenerated regions; scored, weak/circumstantial on its own),
and a frequency-domain spectral check (looks for the periodic peaks classic GAN
upsampling leaves in an image's FFT — deliberately *unscored*, because it targets an
older GAN-specific artifact and doesn't reliably fire on modern diffusion output;
tested against real Gemini-generated images and confirmed it doesn't overclaim there).
EXIF is extracted but *not* scored for being absent — WhatsApp/Instagram/Telegram
strip it from almost everything regardless of authenticity, so treating that as
suspicious was a real bug in an earlier version of this idea, fixed here.

**AI screening** — a pretrained HF classifier (`prithivMLmods/deepfake-detector-model-v1`).
Explicitly caveated everywhere it appears, and the app fires its own warning when zero
faces are detected in an image, since the model is only meaningfully trained on
photographic human faces. A real 3-way alternative model was evaluated mid-project
and rejected — it called a genuine photo "93.5% artificial," more confidently wrong
than what's shipped. That's exactly why this stays a minor signal, not the headline.

**Faces & video** — OpenCV Haar cascade face detection (tuned to `minNeighbors=10`,
`minSize=60` after the defaults threw 5 false "faces" on a photo of flowers). For
video, per-frame classifier scores are still computed, but the real addition is
**frame-to-frame consistency**: short contiguous bursts of frames (start/middle/end
of the clip) get their face region tracked and perceptual-hashed frame-to-frame,
flagging sudden discontinuities a face-swap's blend seam tends to produce — something
a per-frame average structurally can never see, since it never compares one frame to
the next. Thresholds are a first-pass heuristic (median-based, chosen after mean/std
was proven to self-mask the exact anomaly it was meant to catch) — not yet validated
against real deepfake footage, only synthetic test cases.

**Source tracing** (the actual differentiator) —
- *Reverse image search*: checks whether the exact image already exists elsewhere on
  the web (Google Vision Web Detection). Live-tested against a real, widely-published
  photo — correctly found 10 exact matches, the real Wikipedia/Commons pages hosting
  it, and an accurate best-guess label.
- *Landmark detection*: identifies recognizable places in a photo (Vision API).
  Live-tested against a real landmark photo — correctly named it with 81.5%
  confidence and coordinates accurate to within meters. Only works for genuinely
  famous places.
- *Sun/shadow physics*: given a claimed (or EXIF-derived) latitude/longitude/time,
  computes the sun's real position (`astral` — actual solar astronomy, zero AI,
  zero API key, works fully offline) and translates it into where a shadow should
  fall *in the photo's own frame* if the camera's EXIF compass heading is present.
  Validated against known astronomy (equinox sunrise at 90.1° vs. the true 90°) before
  ever touching real data.
- *Auto-extraction*: when a photo's own EXIF carries GPS coordinates, timestamp, and
  camera heading (survives on unedited originals; stripped by most messaging apps),
  the sun/shadow check runs automatically — no manual claim needed at all.
- *AI scene labels*: a vision model (Gemini) describes what's visually obvious in a
  photo — lighting quality, sky conditions, rough shadow direction *in frame* — for a
  human to read alongside the physics above. Deliberately never asked for a precise
  angle (a known weak spot for vision models); it says "unclear" instead of guessing.
  This describes, it does not judge — the human still makes the actual call.

**Reporting** — one-click PDF case report (evidence table, findings, chain of custody,
source records) via `reportlab`.

## What's tested live vs. what's a first-pass heuristic

Everything above has been run against real data at least once — this isn't a
"should work" list. The parts still genuinely open: Direction-1 video thresholds have
only been validated against a synthetic test clip, never real deepfake footage
(FaceForensics++/Celeb-DF would be the next step); and the whole app has only been
run via the dev server this session, not rebuilt through Docker with today's changes.

## API keys (optional — the app works without them)

Two features need a key; everything else runs with zero configuration.

1. **`GOOGLE_VISION_API_KEY`** — reverse image search + landmark detection. Get one at
   [console.cloud.google.com](https://console.cloud.google.com): new project → enable
   "Cloud Vision API" → Credentials → Create Credentials → API key. Needs a billing
   account attached (Google's requirement, not this app's), but has a real free tier
   (1,000 units/month per feature).
2. **`GEMINI_API_KEY`** — AI scene labeling. Get one at
   [aistudio.google.com](https://aistudio.google.com) → "Get API key". No card, no
   billing account, no Cloud project needed for the free tier.

Copy `.env.example` to `.env` and paste both in:

```
cp .env.example .env
# then edit .env and fill in the two keys
```

`.env` is gitignored — never commit real keys.

## Run it locally

### Without Docker (fastest for development)

```
cd TraceLens
uv venv --python 3.12          # first time only
uv pip install -r requirements.txt --python .venv/bin/python   # first time only
.venv/bin/python app.py
```

Open http://127.0.0.1:5000. First AI analysis on a fresh machine downloads the
classifier model from Hugging Face (~1-2 min), then it's cached and fast. This runs
Flask's single-threaded dev server — fine solo, but use Docker if more than one
person needs to click around at once.

### With Docker (recommended for a demo)

```
cd TraceLens
docker build -t tracelens .
docker run -p 8000:8000 -v "$(pwd)/instance:/app/instance" --env-file .env tracelens
```

Open http://127.0.0.1:8000. The AI model is baked into the image at build time, so
the container works offline at runtime. The `-v` mount is what makes case data,
evidence files, and the custody log survive restarts/rebuilds — don't drop it.
`--env-file .env` passes the API keys through without ever baking them into the
image itself. Runs on gunicorn with 4 threads, so multiple people can use it without
blocking each other.

If `docker compose` is available, `docker compose up --build` does the same thing
(add `env_file: .env` under the service if you want the keys picked up that way too).

## Layout

- `app.py` — Flask routes
- `forensics.py` — the actual detection/analysis logic: hashing, EXIF, ELA, the
  pixel-forensics signals, face/video analysis, the AI classifier, source tracing
  (reverse search, landmarks, sun physics), AI scene labels, PDF generation
- `db.py` — SQLite persistence, chain-of-custody hashing, recurrence lookup
- `templates/`, `static/` — frontend

AI output is an advisory screening signal, not forensic proof — every AI-related page
says this on purpose. Keep that framing in the pitch.
