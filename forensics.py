import io
import os
import hashlib
import tempfile
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

    return min(score, 100), indicators


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
    import cv2
    array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
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


def load_ai_model():
    global _model
    if _model is None:
        from transformers import pipeline
        _model = pipeline("image-classification", model=MODEL_ID)
    return _model


def run_ai_detection(image):
    model = load_ai_model()
    results = model(image)
    fake_score = 0.0
    real_score = 0.0
    for result in results:
        label = str(result["label"]).lower()
        score = float(result["score"])
        if "fake" in label or "deepfake" in label or "synthetic" in label or label in ["0", "class_0"]:
            fake_score = max(fake_score, score)
        if "real" in label or "authentic" in label or label in ["1", "class_1"]:
            real_score = max(real_score, score)
    return fake_score, real_score, results


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
