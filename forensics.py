import io
import os
import hashlib
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlparse

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ExifTags, ImageStat, ImageDraw

MODEL_ID = "prithivMLmods/deepfake-detector-model-v1"

_model = None


def calculate_sha256(data):
    return hashlib.sha256(data).hexdigest()


def calculate_phash(image):
    image = image.convert("L").resize((16, 16))
    pixels = np.asarray(image, dtype=np.uint8)
    average = float(pixels.mean())
    bits = "".join("1" if pixel >= average else "0" for pixel in pixels.flatten())
    return f"{int(bits, 2):064x}"


def hamming_distance(hash_a, hash_b):
    try:
        return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()
    except Exception:
        return None


def extract_metadata(image):
    metadata = {}
    try:
        exif = image.getexif()
        for key, value in exif.items():
            tag = ExifTags.TAGS.get(key, str(key))
            metadata[tag] = str(value)
    except Exception:
        pass
    return metadata


def extract_gps_datetime(image):
    """Reads GPS coordinates + UTC timestamp directly out of a photo's own
    EXIF data, when present — the camera's own record of where and when it
    was taken. This only survives on largely-unmodified originals; WhatsApp,
    Instagram, and Telegram strip EXIF from nearly everything they touch, so
    this will come up empty for most forwarded/reposted images. When it IS
    present, it needs no manual entry and no claim to check against — it IS
    the claim, straight from the device.

    Returns {"lat", "lon", "when_utc"} or None if no usable GPS EXIF exists.
    """
    try:
        exif = image.getexif()
        gps = exif.get_ifd(0x8825)
    except Exception:
        return None
    if not gps:
        return None

    def dms_to_decimal(dms, ref):
        try:
            degrees, minutes, seconds = (float(v) for v in dms)
        except (TypeError, ValueError):
            return None
        value = degrees + minutes / 60 + seconds / 3600
        return -value if ref in ("S", "W") else value

    lat = dms_to_decimal(gps.get(2), gps.get(1))
    lon = dms_to_decimal(gps.get(4), gps.get(3))
    if lat is None or lon is None:
        return None

    when_utc = None
    date_stamp, time_stamp = gps.get(29), gps.get(7)
    if date_stamp and time_stamp:
        try:
            hour, minute, second = (int(float(v)) for v in time_stamp)
            year, month, day = (int(v) for v in date_stamp.split(":"))
            when_utc = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        except (TypeError, ValueError):
            when_utc = None

    heading = None
    if gps.get(17) is not None:
        try:
            heading = float(gps[17])  # compass direction the camera was pointing, degrees
        except (TypeError, ValueError):
            heading = None

    return {"lat": round(lat, 6), "lon": round(lon, 6), "when_utc": when_utc, "camera_heading": heading}


def perform_ela(image, quality=90):
    original = image.convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    difference = ImageChops.difference(original, recompressed)
    score = sum(ImageStat.Stat(difference).mean) / 3
    scale = max(1, min(30, 10 / (score + 0.1)))
    visualization = ImageEnhance.Brightness(difference).enhance(scale)
    return visualization, score


def _laplacian_response(gray):
    padded = np.pad(gray, 1, mode="edge")
    return (-4 * gray + padded[:-2, 1:-1] + padded[2:, 1:-1]
            + padded[1:-1, :-2] + padded[1:-1, 2:])


def analyze_noise_consistency(image, grid=8):
    """Splits the image into a grid x grid block grid and measures local
    Laplacian-response variance per block — a standard local sharpness/noise
    estimate. A real photo from one sensor tends to have fairly consistent
    noise texture across the frame; a block whose noise level is wildly
    different from the image's own median can indicate a spliced-in region
    or locally regenerated content (also happens naturally at sharp
    texture/depth-of-field boundaries, so this is circumstantial on its own).

    Uses each image's own median as the baseline (not a fixed number) since
    raw noise level depends heavily on subject matter, not authenticity —
    a close-up portrait and a busy garden scene have very different natural
    noise levels regardless of which one is real.

    Returns None if the image is too small to grid meaningfully.
    """
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    h, w = gray.shape
    if h < grid * 8 or w < grid * 8:
        return None
    lap = _laplacian_response(gray)
    bh, bw = h // grid, w // grid
    values = []
    for r in range(grid):
        for c in range(grid):
            block = lap[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw]
            values.append(float(np.var(block)))
    median = max(float(np.median(values)), 1e-6)
    ratios = [v / median for v in values]
    flagged = [x for x in ratios if x > 4.0 or x < 0.25]
    return {
        "median_variance": round(median, 2),
        "flagged_fraction": round(len(flagged) / len(ratios), 3),
        "max_ratio": round(max(ratios), 2),
    }


def analyze_frequency_artifacts(image, size=256):
    """Classic GANs (transposed-convolution upsampling) leave periodic
    checkerboard-style peaks in an image's frequency spectrum. This computes
    the azimuthally-averaged radial power spectrum and measures how far the
    sharpest peak sticks out above its own local (smoothed) baseline — a
    natural photo's spectrum falls off smoothly; a strong, narrow peak is
    the kind of thing periodic upsampling artifacts produce.

    NOT scored, deliberately: this targets a specific artifact from older
    GAN architectures. Modern diffusion generators (Stable Diffusion,
    Midjourney, DALL-E, Gemini-style image generation) use a different
    mechanism and don't reliably leave this signature — tested against two
    real Gemini-generated images and got no meaningfully higher spike_ratio
    than a real photo. A low number here proves nothing either way; kept as
    a diagnostic in case it's ever run against actual GAN output.
    """
    gray = np.asarray(image.convert("L").resize((size, size)), dtype=np.float64)
    f = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(f))
    cy, cx = size // 2, size // 2
    y, x = np.indices((size, size))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    max_r = min(cx, cy)
    radial_mean = np.array([
        magnitude[r == radius].mean() if (r == radius).any() else 0
        for radius in range(max_r)
    ])
    window = 5
    kernel = np.ones(window) / window
    smoothed = np.convolve(radial_mean, kernel, mode="same")
    residual = radial_mean - smoothed
    residual[:5] = 0
    residual[-5:] = 0  # moving-average edge bias inflates the last few bins regardless of content
    std = float(np.std(residual))
    peak_idx = int(np.argmax(residual))
    spike_ratio = float(residual[peak_idx]) / std if std > 1e-9 else 0.0
    return {"peak_radius": peak_idx, "spike_ratio": round(spike_ratio, 2)}


def forensic_indicators(image, metadata):
    """Returns (score, indicators). Each indicator is {"text", "risk"} — only
    signals that are actually discriminative move the score; near-universal
    conditions (e.g. missing EXIF, which almost every image that's passed
    through WhatsApp/Instagram/Telegram will show regardless of authenticity)
    are surfaced as context, not scored as suspicious."""
    score = 0
    indicators = []

    if metadata:
        indicators.append({"text": "EXIF metadata is present.", "risk": False})
    else:
        indicators.append({
            "text": "No EXIF metadata found. Not scored as suspicious on its own — "
                    "most messaging and social apps (WhatsApp, Instagram, Telegram) "
                    "strip EXIF from nearly everything they touch, genuine or not.",
            "risk": False,
        })

    software = str(metadata.get("Software", "")).lower()
    editing_tools = ["photoshop", "gimp", "canva", "lightroom", "snapseed"]
    matched_tool = next((tool for tool in editing_tools if tool in software), None)
    if matched_tool:
        score += 35
        indicators.append({
            "text": f"Editing software metadata detected: {metadata.get('Software')}. "
                    "The strongest signal this function produces — genuine capture "
                    "devices don't normally stamp this.",
            "risk": True,
        })

    if image.width < 512 or image.height < 512:
        score += 10
        indicators.append({
            "text": "Low resolution source — weak, circumstantial signal (consistent "
                    "with repeated re-saving, but also just true of small originals).",
            "risk": True,
        })

    noise = analyze_noise_consistency(image)
    if noise is not None:
        if noise["flagged_fraction"] > 0.20:
            score += 15
            indicators.append({
                "text": f"Localized noise-texture inconsistency: {noise['flagged_fraction']*100:.0f}% of "
                        f"image regions have a sharply different local noise level than the rest "
                        f"(worst outlier {noise['max_ratio']:.1f}x the image's own median). Consistent "
                        f"with a spliced-in region or regenerated content, but can also occur naturally "
                        f"at sharp texture/depth-of-field boundaries — weak, circumstantial signal. "
                        f"20% threshold was set from a 3-image test sample, not independently validated.",
                "risk": True,
            })
        else:
            indicators.append({
                "text": f"Noise texture is consistent across the image "
                        f"({noise['flagged_fraction']*100:.0f}% of regions flagged, below the 20% "
                        f"threshold). Not scored as suspicious.",
                "risk": False,
            })

    freq = analyze_frequency_artifacts(image)
    indicators.append({
        "text": f"Frequency-spectrum check: sharpest periodic peak is {freq['spike_ratio']:.1f}x the local "
                f"background. Not scored — this specifically targets classic GAN upsampling artifacts, "
                f"and modern diffusion generators (Midjourney/DALL-E/Stable Diffusion/Gemini-style) don't "
                f"reliably leave this signature, so this number doesn't prove anything either way.",
        "risk": False,
    })

    return min(score, 100), indicators


_COMPASS_16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
               "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def _to_compass(degrees):
    return _COMPASS_16[round(degrees / 22.5) % 16]


def sun_position_at(lat, lon, when_utc):
    """Given a claimed latitude/longitude and a UTC datetime, computes exactly
    where the sun actually was using real solar-position astronomy (the
    `astral` library) — not an estimate, not a lookup, real physics. Lets an
    investigator check whether a claimed photo's shadow direction/length and
    light quality are physically consistent with its claimed place and time,
    instead of eyeballing it with no reference at all.

    Deliberately does NOT try to measure shadows in the photo itself or
    render a verdict — that's a hard, unsolved-in-general computer vision
    problem. This hands the investigator the real fact so they can compare
    it against what they actually see, same as everything else on this page.

    `when_utc` must already be in UTC — this does no timezone lookup of its
    own; converting a claimed local time to UTC is the caller's job.
    Returns None on invalid lat/lon.
    """
    from astral import Observer
    from astral.sun import azimuth, elevation
    try:
        observer = Observer(latitude=float(lat), longitude=float(lon))
    except (TypeError, ValueError):
        return None

    az = azimuth(observer, when_utc)
    el = elevation(observer, when_utc)
    shadow_bearing = (az + 180) % 360

    if el < -6:
        light = "sun well below horizon — no direct sunlight or cast shadows should be visible at all"
    elif el < 0:
        light = "civil twilight — very dim, indirect light; shadows should be faint or absent"
    elif el < 15:
        light = "low sun — shadows should be long, light warm-toned (sunrise/sunset conditions)"
    elif el < 40:
        light = "low-to-mid sun — shadows should be moderately long"
    else:
        light = "high sun — shadows should be short"

    return {
        "sun_azimuth": round(az, 1),
        "sun_elevation": round(el, 1),
        "shadow_bearing": round(shadow_bearing, 1),
        "shadow_compass": _to_compass(shadow_bearing),
        "description": (
            f"Sun at {_to_compass(az)} ({az:.0f}°), {el:.0f}° above horizon — shadows "
            f"should fall toward {_to_compass(shadow_bearing)} ({shadow_bearing:.0f}°). {light}."
        ),
    }


_FRAME_RELATIVE_8 = [
    "into the background of the frame, away from camera",
    "toward the background-right of frame",
    "off to the right side of frame",
    "toward the foreground-right, angling toward camera",
    "toward the camera — foreground, growing toward the viewer",
    "toward the foreground-left, angling toward camera",
    "off to the left side of frame",
    "toward the background-left of frame",
]


def relative_shadow_direction(shadow_bearing, camera_heading):
    """Translates an abstract compass bearing ("shadows fall toward 11°")
    into where that shadow should actually appear IN THE PHOTO FRAME, given
    which way the camera itself was pointing (EXIF GPSImgDirection, when a
    device recorded it). This is the piece that makes the sun-position fact
    directly checkable against a photo — a compass bearing alone can't be
    compared to an image unless you also know which way the camera faced.

    Assumes a level, non-tilted shot (typical handheld photo); doesn't
    account for camera roll or a steep up/down pitch.
    """
    relative = (shadow_bearing - camera_heading) % 360
    index = round(relative / 45) % 8
    return _FRAME_RELATIVE_8[index]


def detect_landmark(image_bytes, api_key):
    """Calls Google Cloud Vision's Landmark Detection feature — identifies a
    recognizable place in a photo (monuments, well-known natural features,
    notable buildings) and returns its name, confidence, and coordinates.

    Returns None cleanly if no api_key is configured (GOOGLE_VISION_API_KEY
    unset) — the feature just doesn't run rather than breaking the app — and
    also returns None if nothing was recognized. Only works for genuinely
    famous/notable places; a random street or unremarkable room won't match.

    UNTESTED against a live key as of 2026-08-19 — no Vision API key exists
    on this account yet. The request/response shape follows Google's
    documented REST format for images:annotate; verify against a real key
    and a real recognizable-landmark photo before trusting it in a demo.

    NOTE: this identifies the place. Pulling actual reference photos of that
    place to compare lighting/shadows against needs a *different* API (a
    text-driven image search, e.g. Google's Custom Search JSON API with
    image search enabled) — Vision API's Web Detection only searches by
    image, not by place name, so it can't do that half of the job. That
    piece needs its own separate setup and hasn't been built.
    """
    if not api_key:
        return None
    import base64
    import requests

    encoded = base64.b64encode(image_bytes).decode("ascii")
    payload = {"requests": [{
        "image": {"content": encoded},
        "features": [{"type": "LANDMARK_DETECTION", "maxResults": 3}],
    }]}
    try:
        response = requests.post(
            f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
            json=payload, timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as error:
        return {"error": str(error)}

    annotations = (data.get("responses") or [{}])[0].get("landmarkAnnotations", [])
    if not annotations:
        return None
    results = []
    for annotation in annotations:
        location = (annotation.get("locations") or [{}])[0].get("latLng", {})
        results.append({
            "name": annotation.get("description"),
            "confidence": round(annotation.get("score", 0) * 100, 1),
            "lat": location.get("latitude"),
            "lon": location.get("longitude"),
        })
    return results


def label_scene(image_bytes, api_key, model="gemini-flash-latest"):
    """Asks a vision-capable model (Gemini, free tier) to describe
    qualitative, visually-obvious scene properties — NOT a verdict, and
    deliberately NOT a precise angle. Vision models are known to be
    unreliable at precise geometric estimation, which is exactly why the
    real shadow-bearing math stays in sun_position_at()/
    relative_shadow_direction() above, both pure deterministic astronomy.
    This only does the tedious "describe what's visible" step so a human
    reviewer isn't starting from a blank page — the human still makes the
    actual comparison against the physics and the actual call. The prompt
    explicitly tells the model to say "unclear" rather than guess.

    Returns None if no api_key is configured. Returns {"error": ...} on
    request/parse failure rather than raising, matching detect_landmark's
    pattern, so one flaky call can't take the upload down.

    UNTESTED against a live key as of 2026-08-19 — no Gemini key exists on
    this account yet. Verify against a real key and a real photo (including
    one with obvious strong shadows) before trusting it in a demo.
    """
    if not api_key:
        return None
    import base64
    import json as _json
    import requests

    prompt = (
        "You are assisting a human forensic investigator, not replacing their "
        "judgment. Look at this photo and describe ONLY what is visually "
        "obvious. Do NOT estimate precise angles, compass directions, or "
        "degrees — say \"unclear\" instead of guessing at a number. Respond "
        "as JSON with exactly these fields: "
        '{"shadow_direction_in_frame": one of '
        '["left","right","toward camera","away from camera","multiple/ambiguous","not visible/unclear"], '
        '"light_on_subject_upper": apparent light direction on the upper/top part of the main subject '
        '(e.g. a face or head) — one of '
        '["left","right","above","below","toward camera","away from camera","multiple/ambiguous","not visible/unclear","no clear subject"], '
        '"light_on_subject_lower": same categories as above, but for the lower/body part of the main subject, '
        '"light_on_background": same categories as above, but for the background/environment behind the subject, '
        '"lighting_quality": brief phrase (e.g. "warm/golden", "cool/blue", "flat/overcast", "artificial/mixed"), '
        '"sky_or_background_conditions": brief phrase, '
        '"notable_observations": one sentence on anything else visually relevant, or "" if nothing stands out}'
    )
    encoded = base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "contents": [{"role": "user", "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": encoded}},
        ]}],
        "generationConfig": {"response_mime_type": "application/json"},
    }
    import time
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    last_error = None
    for attempt in range(2):  # one retry — this endpoint has shown real transient 503s under load
        try:
            response = requests.post(
                url, headers={"x-goog-api-key": api_key}, json=payload, timeout=60,
            )
            if response.status_code == 503 and attempt == 0:
                time.sleep(3)
                continue
            response.raise_for_status()
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return _json.loads(text)
        except (requests.RequestException, KeyError, IndexError, ValueError) as error:
            last_error = error
    return {"error": str(last_error)}


_LIGHT_DIRECTIONS = {"left", "right", "above", "below", "toward camera", "away from camera"}


def light_direction_consistency(scene_labels):
    """Real light comes from one dominant source. label_scene() above only
    describes what it sees, region by region — it's never asked whether
    that's consistent, on purpose. This is the function that decides: it
    takes the three region readings (upper/lower subject, background) and
    checks whether they agree. A face lit from the left and a body lit from
    the right, in the same photo, is a physical contradiction, not a matter
    of opinion — same logic a VFX artist uses to spot a bad composite.

    Excluded (returns None) unless at least two regions gave a clear,
    comparable reading — "unclear"/"multiple/ambiguous"/"no clear subject"
    don't count either way, same treatment every other signal in this app
    gets when it can't run.

    Brand new and UNVALIDATED as of 2026-08-19 — never tested against a
    real forged/composite photo, only wired up and reasoned through. Kept
    at a modest weight in compute_priority() for exactly that reason.
    """
    if not scene_labels:
        return None
    fields = {
        "upper (face/head)": scene_labels.get("light_on_subject_upper"),
        "lower (body)": scene_labels.get("light_on_subject_lower"),
        "background": scene_labels.get("light_on_background"),
    }
    readable = {label: v for label, v in fields.items() if v in _LIGHT_DIRECTIONS}
    if len(readable) < 2:
        return None

    values = list(readable.values())
    pairs = [(a, b) for i, a in enumerate(values) for b in values[i + 1:]]
    mismatches = sum(1 for a, b in pairs if a != b)
    return {
        "points": round(100 * mismatches / len(pairs)),
        "readings": readable,
        "consistent": mismatches == 0,
    }


def reverse_image_search(image_bytes, api_key):
    """Calls Google Cloud Vision's Web Detection feature — checks whether
    this exact (or a near-duplicate) image already appears anywhere else on
    the web. This is the original core of Direction 2: if the same image is
    found elsewhere, already dated/captioned differently, that's a checkable
    fact — not an inference — and stronger evidence than any pixel-level
    signal in this file. Answers the standing critique that "Source Tracing"
    was just a manual form (validate_source_url below), which it still is
    for anything the investigator types by hand; this is what actually
    verifies something automatically.

    Returns None cleanly if no api_key is configured, or if the API found
    nothing at all (no matches, no pages, no guess, no entities). Returns
    {"error": ...} on a request failure rather than raising.

    UNTESTED against a live key as of 2026-08-19 — code follows Google's
    documented REST response shape for webDetection, but verify against a
    real key and a real recycled/reposted image before trusting it in a
    demo, same caveat as detect_landmark above.
    """
    if not api_key:
        return None
    import base64
    import requests

    encoded = base64.b64encode(image_bytes).decode("ascii")
    payload = {"requests": [{
        "image": {"content": encoded},
        "features": [{"type": "WEB_DETECTION", "maxResults": 10}],
    }]}
    try:
        response = requests.post(
            f"https://vision.googleapis.com/v1/images:annotate?key={api_key}",
            json=payload, timeout=20,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as error:
        return {"error": str(error)}

    web = (data.get("responses") or [{}])[0].get("webDetection", {})
    full_matches = [{"url": m.get("url")} for m in web.get("fullMatchingImages", [])]
    partial_matches = [{"url": m.get("url")} for m in web.get("partialMatchingImages", [])]
    pages = [
        {"url": p.get("url"), "title": p.get("pageTitle") or "(untitled)"}
        for p in web.get("pagesWithMatchingImages", [])
    ]
    best_guess = ", ".join(g.get("label", "") for g in web.get("bestGuessLabels", []) if g.get("label"))
    entities = [e.get("description") for e in web.get("webEntities", []) if e.get("description")][:5]

    if not (full_matches or partial_matches or pages or best_guess or entities):
        return None

    return {
        "full_matches": full_matches,
        "partial_matches": partial_matches,
        "pages": pages,
        "best_guess": best_guess,
        "entities": entities,
    }


def validate_source_url(url):
    if not url:
        return "No source URL supplied."
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ["http", "https"]:
            return "Invalid URL."
        if not parsed.netloc:
            return "Invalid domain."
        return "Valid source format • Domain: " + parsed.netloc
    except Exception:
        return "Unable to parse URL"


def detect_faces(image):
    """NOTE: minNeighbors=10 / minSize=60 (tightened 2026-08-19 from 5/40) —
    the looser defaults were firing false positives on busy textured
    backgrounds (flowers/foliage in a generated image scored 5-6 "faces").
    Confirmed clean (0 false positives) against that case plus two other real
    uploads. NOT yet confirmed against a real human face — this is a
    minNeighbors increase, which trades away some recall for precision, so
    verify it still finds actual faces in a real photo before trusting it."""
    import cv2
    array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, 1.1, 10, minSize=(60, 60))
    return [tuple(int(v) for v in f) for f in faces]


def create_face_overlay(image, faces):
    result = image.convert("RGBA").copy()
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x, y, w, h in faces:
        padding = int(max(w, h) * 0.2)
        draw.rectangle(
            (
                max(0, x - padding), max(0, y - padding),
                min(result.width, x + w + padding),
                min(result.height, y + h + padding),
            ),
            outline=(255, 80, 80, 230),
            width=4,
        )
    return Image.alpha_composite(result, overlay).convert("RGB")


def _select_device():
    """Auto-detect the best available torch device. Falls back to CPU.
    NOTE: the MPS (Apple Silicon) and CUDA (NVIDIA) paths are implemented per
    the documented torch/transformers APIs but have not been hardware-tested —
    this machine has neither. Verify on the actual target hardware before
    trusting them in a demo."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_ai_model():
    global _model
    if _model is None:
        from transformers import pipeline
        _model = pipeline("image-classification", model=MODEL_ID, device=_select_device())
    return _model


def _scores_from_results(results):
    fake_score = 0.0
    real_score = 0.0
    for result in results:
        label = str(result["label"]).lower()
        score = float(result["score"])
        if "fake" in label or "deepfake" in label or "synthetic" in label or label in ["0", "class_0"]:
            fake_score = max(fake_score, score)
        if "real" in label or "authentic" in label or label in ["1", "class_1"]:
            real_score = max(real_score, score)
    return fake_score, real_score


def run_ai_detection_batch(images, batch_size=8):
    """Runs AI classification on a list of PIL images in batches (one forward
    pass per batch instead of per image) — same math as one-at-a-time in eval
    mode, just faster. Returns a list of (fake_score, real_score, raw_results)
    in the same order as the input."""
    if not images:
        return []
    model = load_ai_model()
    all_results = model(list(images), batch_size=batch_size)
    output = []
    for results in all_results:
        fake_score, real_score = _scores_from_results(results)
        output.append((fake_score, real_score, results))
    return output


def run_ai_detection(image):
    return run_ai_detection_batch([image])[0]


def extract_video_frames(video_data, number_of_frames=10):
    import cv2
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(video_data)
    temp_file.close()
    video = cv2.VideoCapture(temp_file.name)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(video.get(cv2.CAP_PROP_FPS) or 0)
    frames = []
    if total_frames > 0:
        indexes = np.linspace(0, total_frames - 1, min(number_of_frames, total_frames)).astype(int)
        for index in indexes:
            video.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((int(index), Image.fromarray(frame)))
    video.release()
    try:
        os.unlink(temp_file.name)
    except OSError:
        pass
    return frames, fps, total_frames


def _largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def _crop_face(image, bbox, padding_ratio=0.2):
    x, y, w, h = bbox
    padding = int(max(w, h) * padding_ratio)
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image.width, x + w + padding)
    bottom = min(image.height, y + h + padding)
    return image.crop((left, top, right, bottom))


def _burst_frame_ranges(total_frames, fps, burst_seconds=1.5, num_bursts=3):
    """Picks up to `num_bursts` short, contiguous runs of frames spread across
    the video (start/middle/end) rather than one whole-video sparse sample —
    temporal consistency needs frames that are actually adjacent in time, not
    a second or more apart like extract_video_frames' 10-frame sample."""
    if total_frames <= 0:
        return []
    burst_len = max(4, int(round(burst_seconds * fps))) if fps else 8
    burst_len = min(burst_len, total_frames)
    if total_frames <= burst_len:
        return [(0, total_frames)]
    starts = sorted(set(np.linspace(0, total_frames - burst_len, num_bursts).astype(int).tolist()))
    return [(int(s), burst_len) for s in starts]


def _read_burst(video, start, length):
    import cv2
    video.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(length):
        success, frame = video.read()
        if not success:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


def analyze_temporal_consistency(video_data, burst_seconds=1.5, num_bursts=3):
    """Direction-1 detector: rather than scoring sampled frames independently
    (what the per-frame classifier above does), tracks the detected face crop
    across short contiguous bursts of frames and watches for sudden pHash
    jumps between consecutive frames. A face-swap's blend boundary tends to
    flicker frame-to-frame in a way a single-frame classifier never looks for,
    since it never compares one frame to the next.

    Returns None if no face is found in any sampled frame (nothing to track).
    Otherwise returns per-burst frame-to-frame pHash deltas plus which
    transitions were flagged as anomalous.

    NOTE: the flagging threshold (3x a burst's own median transition, floored
    at a minimum bit difference) is a first-pass heuristic, not empirically
    validated against real deepfake footage. Tune it against public sample
    sets (FaceForensics++, Celeb-DF) before trusting it in a demo — same
    caveat this file already carries for the AI classifier and MPS/CUDA path.
    """
    import cv2
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(video_data)
    temp_file.close()
    video = cv2.VideoCapture(temp_file.name)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(video.get(cv2.CAP_PROP_FPS) or 0) or 25.0

    bursts_out = []
    all_deltas = []
    for start, length in _burst_frame_ranges(total_frames, fps, burst_seconds, num_bursts):
        rgb_frames = _read_burst(video, start, length)
        crops = []
        for frame in rgb_frames:
            image = Image.fromarray(frame)
            face = _largest_face(detect_faces(image))
            crops.append(_crop_face(image, face) if face else None)

        hashes = [calculate_phash(c) if c is not None else None for c in crops]
        deltas = []
        for i in range(len(hashes) - 1):
            if hashes[i] is None or hashes[i + 1] is None:
                continue
            bits = hamming_distance(hashes[i], hashes[i + 1])
            if bits is not None:
                deltas.append({
                    "from_frame": start + i, "to_frame": start + i + 1,
                    "seconds": round((start + i + 1) / fps, 2), "bits": bits,
                })

        if len(deltas) >= 3:
            # Median rather than mean/std: a real discontinuity is expected to be
            # a minority of transitions, and a mean-based threshold gets dragged
            # up by the very outliers it's supposed to catch (self-masking) —
            # confirmed by a failing test before this was switched to median.
            median = float(np.median([d["bits"] for d in deltas]))
            threshold = max(20.0, median * 3)
            flagged = [d for d in deltas if d["bits"] > threshold]
        else:
            median = threshold = 0.0
            flagged = []

        bursts_out.append({
            "start_frame": start, "start_seconds": round(start / fps, 2),
            "frames_read": len(rgb_frames), "faces_found": sum(1 for c in crops if c is not None),
            "deltas": deltas, "median_bits": round(median, 1), "threshold_bits": round(threshold, 1),
            "flagged": flagged,
        })
        all_deltas.extend(deltas)

    video.release()
    try:
        os.unlink(temp_file.name)
    except OSError:
        pass

    if not any(b["faces_found"] for b in bursts_out):
        return None

    total_transitions = len(all_deltas)
    flagged_count = sum(len(b["flagged"]) for b in bursts_out)
    consistency_score = round(100 * (1 - flagged_count / total_transitions), 1) if total_transitions else None

    return {
        "bursts": bursts_out,
        "total_transitions": total_transitions,
        "flagged_count": flagged_count,
        "consistency_score": consistency_score,
    }


def compute_priority(item, matches, web_detection, video_temporal=None, scene_labels=None):
    """Combines every signal already computed elsewhere on this evidence item
    into one weighted "how much attention does this deserve" score — not a
    fake/real probability, deliberately. Blending a 0-100 pixel-forensics
    score with a binary "found on the web" flag with an AI classifier
    percentage into a single fake-vs-real number would be false precision;
    "how much should an investigator look closer" is a question all of those
    can honestly answer together.

    Weights come directly from what this project has actually measured, not
    a guess: pixel forensics (noise-consistency etc.) cleanly separated real
    from AI-generated on the one real test we ran it against, so it gets the
    heaviest weight. The AI classifier proved unreliable on the same test —
    it's weighted low, and EXCLUDED entirely (weight 0) on a faceless image,
    since it's been directly shown to be noise there, not just weak.

    Returns {"score", "level", "components"} where `components` is the full
    breakdown — every signal's raw points, its weight, and why it does or
    doesn't count — so the number is never a black box. Nothing here hides
    a signal; it just says how much each one was trusted to matter.
    """
    components = []

    # Lighting-direction consistency: commented out 2026-08-19, not deleted. Tested live
    # against the real pug photo + 2 Gemini fakes (3 runs each) and it doesn't separate
    # real from fake — the SAME real photo flipped between "consistent" and "inconsistent"
    # across its own two successful runs, more disagreement with itself than there was
    # between real and fake. That's the underlying vision model's non-determinism, not a
    # bug in light_direction_consistency() below, which is still intact if this gets
    # revisited (e.g. with majority-voting across multiple calls per image).
    #
    # if item.get("kind") == "image":
    #     light_check = light_direction_consistency(scene_labels)
    #     if light_check is not None:
    #         components.append({
    #             "label": "Lighting-direction consistency",
    #             "why": "Real light comes from one dominant source — different parts of the photo implying "
    #                    "different light directions is a physical contradiction, not a guess. AI only reports "
    #                    "what it sees per region; this page's own logic decides if it adds up. Brand new and "
    #                    "unvalidated — never tested against a real forged photo, weighted low for exactly that reason.",
    #             "points": light_check["points"], "weight": 0.15,
    #         })
    #     else:
    #         components.append({
    #             "label": "Lighting-direction consistency", "points": None, "weight": 0.0,
    #             "why": "Excluded — needs at least two of face/body/background to get a clear, comparable "
    #                    "reading, and this image didn't have enough of them.",
    #         })

    if item.get("kind") == "image":
        components.append({
            "label": "Editing signs (pixel forensics)",
            "why": "The one signal that's actually been shown to separate real from AI-generated on this project's own test data.",
            "points": item.get("forensic_score") or 0, "weight": 0.40,
        })

    if item.get("ai_fake_score") is not None:
        no_face_image = item.get("kind") == "image" and not item.get("faces_detected")
        if no_face_image:
            components.append({
                "label": "AI classifier opinion", "points": None, "weight": 0.0,
                "why": "Excluded — no face detected, and this classifier has been directly shown to be noise outside human faces, not just weak.",
            })
        else:
            components.append({
                "label": "AI classifier opinion",
                "why": "Weighted low on purpose — a single trained model that's shown itself to be confidently wrong before.",
                "points": round(item["ai_fake_score"] * 100, 1), "weight": 0.15,
            })

    best_distance = min((m["distance"] for m in matches), default=None) if matches else None
    components.append({
        "label": "Matches something already in this case",
        "why": "A near-identical match already logged is worth a second look, whatever the reason.",
        "points": 100 if (best_distance is not None and best_distance <= 6) else (45 if matches else 0),
        "weight": 0.15,
    })

    if web_detection and not web_detection.get("error"):
        found = bool(web_detection.get("full_matches") or web_detection.get("partial_matches") or web_detection.get("pages"))
        components.append({
            "label": "Already published elsewhere on the web",
            "why": "Not evidence of AI manipulation by itself — but a recycled/miscaptioned real photo is exactly the pattern this catches.",
            "points": 100 if found else 0, "weight": 0.15,
        })
    else:
        components.append({
            "label": "Already published elsewhere on the web", "points": None, "weight": 0.0,
            "why": "Not run — no Vision API key configured, or the lookup failed.",
        })

    if video_temporal:
        total = video_temporal.get("total_transitions") or 0
        ratio = (video_temporal.get("flagged_count", 0) / total) if total else 0
        components.append({
            "label": "Frame-to-frame video glitches",
            "why": "The one check built specifically to look at change over time in video, instead of scoring frames one at a time.",
            "points": round(ratio * 100, 1), "weight": 0.40,
        })

    active = [c for c in components if c["points"] is not None]
    total_weight = sum(c["weight"] for c in active) or 1
    score = sum(c["points"] * c["weight"] for c in active) / total_weight

    if score >= 50:
        level = "HIGH"
    elif score >= 20:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {"score": round(score, 1), "level": level, "components": components}


def generate_pdf_report(case_id, investigator, evidence_list, events, findings, sources):
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("TraceLens — Digital Media Forensic Report", styles["Title"]),
        Spacer(1, 10),
        Paragraph(f"Case ID: {case_id}", styles["BodyText"]),
        Paragraph(f"Investigator: {investigator}", styles["BodyText"]),
        Spacer(1, 15),
        Paragraph("Evidence", styles["Heading2"]),
    ]

    if evidence_list:
        evidence_rows = [["Filename", "SHA-256", "pHash", "ELA", "Forensic Score"]]
        for item in evidence_list:
            evidence_rows.append([
                str(item.get("filename", ""))[:40],
                str(item.get("sha256", ""))[:20] + "...",
                str(item.get("phash", ""))[:20] + ("..." if item.get("phash") else ""),
                f"{item.get('ela_score'):.2f}" if item.get("ela_score") is not None else "-",
                str(item.get("forensic_score", "-")),
            ])
        evidence_table = Table(evidence_rows, colWidths=[45 * mm, 40 * mm, 40 * mm, 20 * mm, 25 * mm])
        evidence_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
        ]))
        story.append(evidence_table)
    else:
        story.append(Paragraph("No evidence recorded.", styles["BodyText"]))

    story += [Spacer(1, 15), Paragraph("Findings", styles["Heading2"])]
    if findings:
        for finding in findings:
            story.append(Paragraph("• " + str(finding), styles["BodyText"]))
    else:
        story.append(Paragraph("No findings recorded.", styles["BodyText"]))

    story += [Spacer(1, 15), Paragraph("Source Tracing", styles["Heading2"])]
    if sources:
        source_rows = [["Source ID", "URL", "Known Hash", "Notes"]] + [
            [str(s.get("source_id", ""))[:25], str(s.get("url", ""))[:40],
             str(s.get("known_hash", ""))[:20], str(s.get("notes", ""))[:40]]
            for s in sources
        ]
        source_table = Table(source_rows, colWidths=[30 * mm, 50 * mm, 35 * mm, 40 * mm])
        source_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
        ]))
        story.append(source_table)
    else:
        story.append(Paragraph("No sources recorded.", styles["BodyText"]))

    story += [Spacer(1, 15), Paragraph("Chain of Custody", styles["Heading2"])]
    custody_rows = [["Time", "Action", "Details", "Event Hash"]] + [
        [event.get("time", ""), event.get("action", ""), str(event.get("details", ""))[:60],
         str(event.get("event_hash", ""))[:16] + "..."]
        for event in events
    ]
    custody_table = Table(custody_rows, colWidths=[30 * mm, 35 * mm, 60 * mm, 30 * mm])
    custody_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    story.append(custody_table)

    story += [
        Spacer(1, 15),
        Paragraph(
            "AI results are screening signals and require qualified human forensic review "
            "before evidentiary conclusions. Chain-of-custody entries are hash-chained "
            "(each event's hash incorporates the previous event's hash) so any retroactive "
            "edit to this log is detectable.",
            styles["BodyText"],
        ),
    ]
    document.build(story)
    return buffer.getvalue()
