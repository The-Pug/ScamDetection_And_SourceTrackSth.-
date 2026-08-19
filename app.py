import io
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from flask import (
    Flask, render_template, request, redirect, url_for, session,
    send_file, send_from_directory, jsonify, flash,
)
from PIL import Image
from werkzeug.utils import secure_filename

import db
import forensics

load_dotenv()
GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

BASE_DIR = Path(__file__).parent
MEDIA_DIR = BASE_DIR / "instance" / "media"
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "webp"}
ALLOWED_VIDEO = {"mp4", "mov", "avi", "mkv"}

def _load_secret_key():
    env_secret = os.environ.get("TRACELENS_SECRET")
    if env_secret:
        return env_secret
    key_path = BASE_DIR / "instance" / "secret_key"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    if key_path.exists():
        return key_path.read_bytes()
    key = os.urandom(24)
    key_path.write_bytes(key)
    return key


app = Flask(__name__)
app.secret_key = _load_secret_key()
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB, to allow whole-folder ingests

db.init_db()


def ext_of(filename):
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def current_case():
    case_id = session.get("case_id", "CASE-2026-001")
    investigator = session.get("investigator", "Demo Investigator")
    db.ensure_case(case_id, investigator)
    return case_id, investigator


def case_media_dir(case_id):
    d = MEDIA_DIR / secure_filename(case_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


@app.context_processor
def inject_globals():
    case_id, investigator = current_case()
    return {
        "case_id": case_id,
        "investigator": investigator,
        "all_cases": db.list_cases(),
        "nav_counts": {
            "evidence": len(db.list_evidence(case_id)),
            "events": len(db.list_events(case_id)),
            "sources": len(db.list_sources(case_id)),
        },
    }


@app.route("/case/set", methods=["POST"])
def set_case():
    case_id = request.form.get("case_id", "").strip() or "CASE-2026-001"
    investigator = request.form.get("investigator", "").strip() or "Demo Investigator"
    session["case_id"] = case_id
    session["investigator"] = investigator
    db.ensure_case(case_id, investigator)
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/")
def root():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    case_id, investigator = current_case()
    evidence_list = db.list_evidence(case_id)
    events = db.list_events(case_id)
    findings = db.list_findings(case_id)
    sources = db.list_sources(case_id)
    chain_ok, broken_at = db.verify_chain(case_id)
    stats = {
        "evidence": len(evidence_list),
        "ai_analyses": sum(1 for e in events if "AI" in e["action"]),
        "sources": len(sources),
        "custody_events": len(events),
    }
    return render_template(
        "dashboard.html",
        stats=stats, findings=findings, chain_ok=chain_ok, broken_at=broken_at,
    )


@app.route("/evidence")
def evidence():
    case_id, _ = current_case()
    evidence_list = db.list_evidence(case_id)
    return render_template("evidence.html", evidence_list=evidence_list)


def _prepare_evidence_file(case_id, investigator, file, batch_id):
    """Fast path for one uploaded file: hash, EXIF/pHash/ELA (images), DB
    insert, custody log, recurrence check. Deliberately does NOT call the AI
    model — that's batched separately across the whole upload afterward, in
    _run_ai_for_batch, so N images cost one forward pass each instead of N.
    Never raises — failures/skips are recorded as a batch_items row immediately
    so one bad file can't sink a folder upload; returns None for those, or a
    dict describing what to hand to the AI phase."""
    raw_name = file.filename or "unnamed"
    filename = secure_filename(raw_name) or f"file_{uuid.uuid4().hex[:8]}"
    extension = ext_of(filename)

    if extension not in ALLOWED_IMAGE and extension not in ALLOWED_VIDEO:
        db.add_batch_item(batch_id, case_id, raw_name, "skipped")
        return None

    try:
        data = file.read()
        sha256 = forensics.calculate_sha256(data)
        media_dir = case_media_dir(case_id)

        if extension in ALLOWED_IMAGE:
            image = Image.open(io.BytesIO(data)).convert("RGB")
            metadata = forensics.extract_metadata(image)
            phash = forensics.calculate_phash(image)
            ela_image, ela_score = forensics.perform_ela(image)
            forensic_score, reasons = forensics.forensic_indicators(image, metadata)
            landmark = forensics.detect_landmark(data, GOOGLE_VISION_API_KEY)
            web_detection = forensics.reverse_image_search(data, GOOGLE_VISION_API_KEY)
            scene_labels = forensics.label_scene(data, GEMINI_API_KEY)
            gps = forensics.extract_gps_datetime(image)

            stored_name = f"orig_{sha256[:12]}.jpg"
            ela_name = f"ela_{sha256[:12]}.jpg"
            image.save(media_dir / stored_name, format="JPEG", quality=95)
            ela_image.save(media_dir / ela_name, format="JPEG", quality=90)

            evidence_id = db.insert_evidence(case_id, {
                "kind": "image",
                "filename": filename,
                "sha256": sha256,
                "mime": file.mimetype,
                "bytes": len(data),
                "width": image.width,
                "height": image.height,
                "phash": phash,
                "ela_score": round(ela_score, 3),
                "forensic_score": forensic_score,
                "forensic_reasons": json.dumps(reasons),
                "metadata_json": json.dumps(metadata),
                "landmark_json": json.dumps(landmark) if landmark else None,
                "web_detection_json": json.dumps(web_detection) if web_detection else None,
                "scene_labels_json": json.dumps(scene_labels) if scene_labels else None,
                "filepath": str(media_dir / stored_name),
                "ela_filepath": str(media_dir / ela_name),
                "received_at": db.current_time(),
                "logged": 1,
            })
            if gps and gps.get("when_utc"):
                db.add_source(
                    case_id, url="", source_id=filename, known_hash=sha256,
                    notes="Auto-extracted from this file's own EXIF GPS data — no manual claim needed.",
                    claimed_lat=gps["lat"], claimed_lon=gps["lon"],
                    claimed_datetime=gps["when_utc"].strftime("%Y-%m-%dT%H:%M"),
                    camera_heading=gps.get("camera_heading"),
                )
                db.add_finding(
                    case_id,
                    f"'{filename}' carries its own EXIF GPS data ({gps['lat']}, {gps['lon']}) — "
                    f"sun/shadow physics check auto-recorded in Source Tracing, no manual claim needed.",
                )

            matches = db.find_phash_matches(phash, evidence_id)
            if matches:
                best = matches[0]
                db.add_finding(
                    case_id,
                    f"Recurrence: '{filename}' perceptually matches {len(matches)} prior "
                    f"record(s) already in the system (closest: '{best['filename']}' in case "
                    f"{best['case_id']}, Hamming distance {best['distance']}).",
                )

            if web_detection and (web_detection.get("full_matches") or web_detection.get("partial_matches") or web_detection.get("pages")):
                db.add_finding(
                    case_id,
                    f"Reverse image search: '{filename}' already appears elsewhere on the web — "
                    f"{len(web_detection.get('full_matches', []))} exact match(es), "
                    f"{len(web_detection.get('pages', []))} page(s) hosting it. Check Evidence Detail "
                    f"for URLs; a genuinely new, unpublished image would show none of this.",
                )

            return {
                "kind": "image", "evidence_id": evidence_id, "filename": filename,
                "image": image, "media_dir": media_dir, "sha256": sha256,
                "recurrence": len(matches),
            }

        else:  # video
            stored_name = f"orig_{sha256[:12]}.{extension}"
            with open(media_dir / stored_name, "wb") as fh:
                fh.write(data)
            evidence_id = db.insert_evidence(case_id, {
                "kind": "video",
                "filename": filename,
                "sha256": sha256,
                "mime": file.mimetype,
                "bytes": len(data),
                "filepath": str(media_dir / stored_name),
                "received_at": db.current_time(),
                "logged": 1,
            })
            return {
                "kind": "video", "evidence_id": evidence_id, "filename": filename,
                "video_data": data, "sha256": sha256,
            }

    except Exception as error:
        db.add_batch_item(batch_id, case_id, raw_name, "error", error=str(error)[:300])
        return None


def _run_ai_for_batch(case_id, investigator, batch_id, prepared_items):
    """Slow path: batches AI inference across every image in this upload in
    one forward pass per group of 8 (instead of one call per image), then a
    per-video batched pass over each video's sampled frames. Always ends with
    exactly one batch_items row per prepared item, even if the AI step fails —
    the hash/EXIF/ELA results from the fast path already stand regardless."""
    image_items = [p for p in prepared_items if p["kind"] == "image"]
    video_items = [p for p in prepared_items if p["kind"] == "video"]

    if image_items:
        try:
            results = forensics.run_ai_detection_batch([p["image"] for p in image_items], batch_size=8)
        except Exception:
            results = [None] * len(image_items)

        for item, result in zip(image_items, results):
            evidence_id = item["evidence_id"]
            filename = item["filename"]
            sha256 = item["sha256"]
            if result is None:
                db.add_event(case_id, investigator, "Evidence received",
                              f"{filename} | SHA-256={sha256} | AI screening unavailable")
                db.add_batch_item(batch_id, case_id, filename, "ok", evidence_id=evidence_id,
                                   kind="image", recurrence=item["recurrence"])
                continue
            fake_score, real_score, raw_results = result
            image = item["image"]
            media_dir = item["media_dir"]
            faces = forensics.detect_faces(image)
            overlay_path = None
            if faces:
                overlay = forensics.create_face_overlay(image, faces)
                overlay_name = f"overlay_{sha256[:12]}.jpg"
                overlay.save(media_dir / overlay_name, format="JPEG", quality=90)
                overlay_path = str(media_dir / overlay_name)
            db.update_evidence(evidence_id, {
                "ai_fake_score": fake_score,
                "ai_real_score": real_score,
                "ai_raw_json": json.dumps([{"label": r["label"], "score": r["score"]} for r in raw_results]),
                "faces_detected": len(faces),
                "face_overlay_filepath": overlay_path,
            })
            db.add_finding(
                case_id,
                f"AI screening on '{filename}': synthetic/manipulated score "
                f"{fake_score*100:.1f}%, authentic score {real_score*100:.1f}%.",
            )
            db.add_event(case_id, investigator, "Evidence received & AI-screened",
                          f"{filename} | SHA-256={sha256} | fake={fake_score:.4f}; real={real_score:.4f}; faces={len(faces)}")
            db.add_batch_item(batch_id, case_id, filename, "ok", evidence_id=evidence_id, kind="image",
                               ai_fake_score=fake_score, faces=len(faces), recurrence=item["recurrence"])

    for item in video_items:
        evidence_id = item["evidence_id"]
        filename = item["filename"]
        sha256 = item["sha256"]
        avg_score = None
        try:
            frames, fps, total = forensics.extract_video_frames(item["video_data"], 10)
            if frames:
                results = forensics.run_ai_detection_batch([frame for _, frame in frames], batch_size=8)
                frame_results = [
                    {
                        "frame": frame_number,
                        "seconds": round(frame_number / fps, 2) if fps else 0,
                        "synthetic_score": round(fake_score, 4),
                    }
                    for (frame_number, _), (fake_score, _, _) in zip(frames, results)
                ]
                avg_score = sum(r["synthetic_score"] for r in frame_results) / len(frame_results)
                update_fields = {
                    "video_frame_results": json.dumps(frame_results),
                    "ai_fake_score": avg_score,
                }
                finding_text = (
                    f"Video screening on '{filename}' sampled {len(frame_results)} frames; "
                    f"average synthetic score {avg_score*100:.1f}%."
                )
                temporal = forensics.analyze_temporal_consistency(item["video_data"])
                if temporal:
                    update_fields["video_temporal_json"] = json.dumps(temporal)
                    if temporal["flagged_count"]:
                        finding_text += (
                            f" Frame-to-frame consistency check flagged {temporal['flagged_count']} "
                            f"discontinuity(ies) out of {temporal['total_transitions']} transitions "
                            f"(consistency score {temporal['consistency_score']}/100)."
                        )
                db.update_evidence(evidence_id, update_fields)
                db.add_finding(case_id, finding_text)
        except Exception:
            pass
        if avg_score is not None:
            db.add_event(case_id, investigator, "Evidence received & AI-screened",
                          f"{filename} | SHA-256={sha256} | avg synthetic={avg_score*100:.1f}% | sampled_frames={len(frame_results)}")
        else:
            db.add_event(case_id, investigator, "Evidence received",
                          f"{filename} | SHA-256={sha256} | AI screening unavailable")
        db.add_batch_item(batch_id, case_id, filename, "ok", evidence_id=evidence_id, kind="video",
                           ai_fake_score=avg_score)


@app.route("/evidence/upload", methods=["POST"])
def evidence_upload():
    case_id, investigator = current_case()
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        flash("No files selected.", "error")
        return redirect(url_for("evidence"))

    batch_id = uuid.uuid4().hex[:12]
    prepared = [_prepare_evidence_file(case_id, investigator, f, batch_id) for f in files]
    prepared = [p for p in prepared if p is not None]
    _run_ai_for_batch(case_id, investigator, batch_id, prepared)

    items = db.list_batch_items(batch_id)
    ok_items = [i for i in items if i["status"] == "ok"]
    if len(items) == 1 and ok_items:
        return redirect(url_for("evidence_detail", evidence_id=ok_items[0]["evidence_id"]))
    return redirect(url_for("batch_results", batch_id=batch_id))


@app.route("/evidence/batch/<batch_id>")
def batch_results(batch_id):
    items = db.list_batch_items(batch_id)
    if not items:
        flash("Batch not found.", "error")
        return redirect(url_for("evidence"))
    ok_items = [i for i in items if i["status"] == "ok"]
    skipped_items = [i for i in items if i["status"] == "skipped"]
    error_items = [i for i in items if i["status"] == "error"]
    recurrence_flagged = sum(1 for i in ok_items if i.get("recurrence"))
    return render_template(
        "batch_results.html", batch_id=batch_id, ok_items=ok_items,
        skipped_items=skipped_items, error_items=error_items,
        recurrence_flagged=recurrence_flagged,
    )


@app.route("/evidence/<int:evidence_id>")
def evidence_detail(evidence_id):
    item = db.get_evidence(evidence_id)
    if not item:
        flash("Evidence not found.", "error")
        return redirect(url_for("evidence"))
    metadata = json.loads(item["metadata_json"]) if item.get("metadata_json") else {}
    reasons = json.loads(item["forensic_reasons"]) if item.get("forensic_reasons") else []
    ai_raw = json.loads(item["ai_raw_json"]) if item.get("ai_raw_json") else None
    video_frames = json.loads(item["video_frame_results"]) if item.get("video_frame_results") else None
    video_temporal = json.loads(item["video_temporal_json"]) if item.get("video_temporal_json") else None
    landmark = json.loads(item["landmark_json"]) if item.get("landmark_json") else None
    web_detection = json.loads(item["web_detection_json"]) if item.get("web_detection_json") else None
    scene_labels = json.loads(item["scene_labels_json"]) if item.get("scene_labels_json") else None
    matches = db.find_phash_matches(item["phash"], evidence_id) if item.get("phash") else []
    priority = forensics.compute_priority(item, matches, web_detection, video_temporal, scene_labels) if item.get("ai_fake_score") is not None or item["kind"] == "image" else None
    return render_template(
        "evidence_detail.html", item=item, metadata=metadata, reasons=reasons,
        ai_raw=ai_raw, video_frames=video_frames, video_temporal=video_temporal,
        landmark=landmark, web_detection=web_detection, scene_labels=scene_labels,
        matches=matches, priority=priority,
    )


@app.route("/evidence/<int:evidence_id>/log", methods=["POST"])
def evidence_log(evidence_id):
    case_id, investigator = current_case()
    item = db.get_evidence(evidence_id)
    if item:
        db.add_event(case_id, investigator, "Evidence received",
                      f"{item['filename']} | SHA-256={item['sha256']}")
        db.update_evidence(evidence_id, {"logged": 1})
        flash("Evidence added to the activity log.", "success")
    return redirect(url_for("evidence_detail", evidence_id=evidence_id))


@app.route("/media/<int:evidence_id>/<kind>")
def media(evidence_id, kind):
    item = db.get_evidence(evidence_id)
    if not item:
        return "Not found", 404
    path_field = {"original": "filepath", "ela": "ela_filepath", "overlay": "face_overlay_filepath"}.get(kind)
    if not path_field or not item.get(path_field):
        return "Not found", 404
    path = Path(item[path_field])
    return send_from_directory(path.parent, path.name)


@app.route("/ai/<int:evidence_id>/run", methods=["POST"])
def ai_run(evidence_id):
    case_id, investigator = current_case()
    item = db.get_evidence(evidence_id)
    if not item:
        flash("Evidence not found.", "error")
        return redirect(url_for("evidence"))

    try:
        if item["kind"] == "image":
            image = Image.open(item["filepath"]).convert("RGB")
            fake_score, real_score, raw_results = forensics.run_ai_detection(image)
            faces = forensics.detect_faces(image)
            overlay_path = None
            if faces:
                overlay = forensics.create_face_overlay(image, faces)
                media_dir = case_media_dir(case_id)
                overlay_name = f"overlay_{item['sha256'][:12]}.jpg"
                overlay.save(media_dir / overlay_name, format="JPEG", quality=90)
                overlay_path = str(media_dir / overlay_name)

            db.update_evidence(evidence_id, {
                "ai_fake_score": fake_score,
                "ai_real_score": real_score,
                "ai_raw_json": json.dumps([{"label": r["label"], "score": r["score"]} for r in raw_results]),
                "faces_detected": len(faces),
                "face_overlay_filepath": overlay_path,
            })
            db.add_finding(
                case_id,
                f"AI screening on '{item['filename']}': synthetic/manipulated score "
                f"{fake_score*100:.1f}%, authentic score {real_score*100:.1f}%.",
            )
            db.add_event(case_id, investigator, "AI image analysis",
                          f"fake={fake_score:.4f}; real={real_score:.4f}; faces={len(faces)}")
        else:
            with open(item["filepath"], "rb") as f:
                video_data = f.read()
            frames, fps, total = forensics.extract_video_frames(video_data, 10)
            results = []
            for frame_number, frame in frames:
                fake_score, real_score, _ = forensics.run_ai_detection(frame)
                results.append({
                    "frame": frame_number,
                    "seconds": round(frame_number / fps, 2) if fps else 0,
                    "synthetic_score": round(fake_score, 4),
                })
            average_score = (sum(r["synthetic_score"] for r in results) / len(results)) if results else 0
            update_fields = {
                "video_frame_results": json.dumps(results),
                "ai_fake_score": average_score,
            }
            finding_text = (
                f"Video screening on '{item['filename']}' sampled {len(results)} frames; "
                f"average synthetic score {average_score*100:.1f}%."
            )
            temporal = forensics.analyze_temporal_consistency(video_data)
            if temporal:
                update_fields["video_temporal_json"] = json.dumps(temporal)
                if temporal["flagged_count"]:
                    finding_text += (
                        f" Frame-to-frame consistency check flagged {temporal['flagged_count']} "
                        f"discontinuity(ies) out of {temporal['total_transitions']} transitions "
                        f"(consistency score {temporal['consistency_score']}/100)."
                    )
            db.update_evidence(evidence_id, update_fields)
            db.add_finding(case_id, finding_text)
            db.add_event(case_id, investigator, "AI video analysis", f"sampled_frames={len(results)}")
        flash("AI analysis complete.", "success")
    except Exception as error:
        flash(f"AI model could not run: {error}", "error")

    return redirect(url_for("evidence_detail", evidence_id=evidence_id))


@app.route("/sources", methods=["GET", "POST"])
def sources():
    case_id, investigator = current_case()
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        source_id = request.form.get("source_id", "").strip()
        known_hash = request.form.get("known_hash", "").strip()
        notes = request.form.get("notes", "").strip()
        claimed_lat = request.form.get("claimed_lat", "").strip() or None
        claimed_lon = request.form.get("claimed_lon", "").strip() or None
        claimed_datetime = request.form.get("claimed_datetime", "").strip() or None
        db.add_source(case_id, url, source_id, known_hash, notes, claimed_lat, claimed_lon, claimed_datetime)
        db.add_event(case_id, investigator, "Source recorded",
                      json.dumps({"url": url, "source_id": source_id, "known_hash": known_hash}))
        flash("Source recorded.", "success")
        return redirect(url_for("sources"))

    source_list = db.list_sources(case_id)
    for source in source_list:
        source["sun_fact"] = None
        source["frame_relative"] = None
        if source.get("claimed_lat") and source.get("claimed_lon") and source.get("claimed_datetime"):
            try:
                when_utc = datetime.strptime(source["claimed_datetime"], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
                source["sun_fact"] = forensics.sun_position_at(source["claimed_lat"], source["claimed_lon"], when_utc)
            except (ValueError, TypeError):
                pass
            if source["sun_fact"] and source.get("camera_heading") is not None:
                source["frame_relative"] = forensics.relative_shadow_direction(
                    source["sun_fact"]["shadow_bearing"], source["camera_heading"])

    url_check = None
    check_url = request.args.get("check_url")
    if check_url:
        url_check = forensics.validate_source_url(check_url)
    return render_template("sources.html", source_list=source_list, url_check=url_check, check_url=check_url or "")


@app.route("/api/hamming")
def api_hamming():
    hash_a = request.args.get("a", "")
    hash_b = request.args.get("b", "")
    distance = forensics.hamming_distance(hash_a, hash_b) if hash_a and hash_b else None
    return jsonify({"distance": distance})


@app.route("/custody", methods=["GET", "POST"])
def custody():
    case_id, investigator = current_case()
    if request.method == "POST":
        action = request.form.get("action", "").strip()
        details = request.form.get("details", "").strip()
        db.add_event(case_id, investigator, action, details)
        flash("Event added.", "success")
        return redirect(url_for("custody"))

    events = db.list_events(case_id)
    chain_ok, broken_at = db.verify_chain(case_id)
    return render_template("custody.html", events=events, chain_ok=chain_ok, broken_at=broken_at)


@app.route("/custody/export.csv")
def custody_export():
    case_id, _ = current_case()
    events = db.list_events(case_id)
    lines = ["time,action,details,event_hash"]
    for e in events:
        details = str(e["details"]).replace('"', '""')
        lines.append(f'"{e["time"]}","{e["action"]}","{details}","{e["event_hash"]}"')
    csv_data = "\n".join(lines)
    return send_file(
        io.BytesIO(csv_data.encode()), mimetype="text/csv",
        as_attachment=True, download_name=f"{case_id}_activity_log.csv",
    )


@app.route("/report")
def report():
    return render_template("report.html")


@app.route("/report/generate")
def report_generate():
    case_id, investigator = current_case()
    evidence_list = db.list_evidence(case_id)
    events = db.list_events(case_id)
    findings = [f["text"] for f in db.list_findings(case_id)]
    sources_list = db.list_sources(case_id)

    pdf_bytes = forensics.generate_pdf_report(case_id, investigator, evidence_list, events, findings, sources_list)
    db.add_event(case_id, investigator, "Report generated", "PDF report generated.")
    return send_file(
        io.BytesIO(pdf_bytes), mimetype="application/pdf",
        as_attachment=True, download_name=f"{case_id}_TraceLens_Report.pdf",
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
