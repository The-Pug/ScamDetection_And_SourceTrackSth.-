import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "instance" / "tracelens.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    investigator TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT,
    kind TEXT,
    filename TEXT,
    sha256 TEXT,
    mime TEXT,
    bytes INTEGER,
    width INTEGER,
    height INTEGER,
    phash TEXT,
    ela_score REAL,
    forensic_score INTEGER,
    forensic_reasons TEXT,
    metadata_json TEXT,
    filepath TEXT,
    ela_filepath TEXT,
    received_at TEXT,
    logged INTEGER DEFAULT 0,
    ai_fake_score REAL,
    ai_real_score REAL,
    ai_raw_json TEXT,
    faces_detected INTEGER,
    face_overlay_filepath TEXT,
    video_frame_results TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT,
    time TEXT,
    investigator TEXT,
    action TEXT,
    details TEXT,
    prev_hash TEXT,
    event_hash TEXT
);

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT,
    time TEXT,
    url TEXT,
    source_id TEXT,
    known_hash TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT,
    time TEXT,
    text TEXT
);

CREATE TABLE IF NOT EXISTS batch_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT,
    case_id TEXT,
    time TEXT,
    filename TEXT,
    status TEXT,
    evidence_id INTEGER,
    kind TEXT,
    ai_fake_score REAL,
    faces INTEGER,
    recurrence INTEGER,
    error TEXT
);
"""


def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def current_time():
    return datetime.now().isoformat(timespec="seconds")


def ensure_case(case_id, investigator):
    conn = get_db()
    row = conn.execute("SELECT case_id FROM cases WHERE case_id = ?", (case_id,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO cases (case_id, investigator, created_at) VALUES (?, ?, ?)",
            (case_id, investigator, current_time()),
        )
        conn.commit()
    conn.close()


def list_cases():
    conn = get_db()
    rows = conn.execute("SELECT * FROM cases ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_event(case_id, investigator, action, details):
    conn = get_db()
    last = conn.execute(
        "SELECT event_hash FROM events WHERE case_id = ? ORDER BY id DESC LIMIT 1",
        (case_id,),
    ).fetchone()
    prev_hash = last["event_hash"] if last else "GENESIS"
    time = current_time()
    payload = f"{prev_hash}|{time}|{case_id}|{investigator}|{action}|{details}"
    event_hash = hashlib.sha256(payload.encode()).hexdigest()
    conn.execute(
        "INSERT INTO events (case_id, time, investigator, action, details, prev_hash, event_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (case_id, time, investigator, action, details, prev_hash, event_hash),
    )
    conn.commit()
    conn.close()
    return event_hash


def verify_chain(case_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM events WHERE case_id = ? ORDER BY id ASC", (case_id,)
    ).fetchall()
    conn.close()
    prev_hash = "GENESIS"
    for row in rows:
        payload = f"{prev_hash}|{row['time']}|{row['case_id']}|{row['investigator']}|{row['action']}|{row['details']}"
        expected = hashlib.sha256(payload.encode()).hexdigest()
        if expected != row["event_hash"] or row["prev_hash"] != prev_hash:
            return False, row["id"]
        prev_hash = row["event_hash"]
    return True, None


def add_finding(case_id, text):
    conn = get_db()
    conn.execute(
        "INSERT INTO findings (case_id, time, text) VALUES (?, ?, ?)",
        (case_id, current_time(), text),
    )
    conn.commit()
    conn.close()


def add_source(case_id, url, source_id, known_hash, notes):
    conn = get_db()
    conn.execute(
        "INSERT INTO sources (case_id, time, url, source_id, known_hash, notes) VALUES (?, ?, ?, ?, ?, ?)",
        (case_id, current_time(), url, source_id, known_hash, notes),
    )
    conn.commit()
    conn.close()


def insert_evidence(case_id, fields):
    conn = get_db()
    columns = ["case_id"] + list(fields.keys())
    placeholders = ",".join("?" for _ in columns)
    values = [case_id] + list(fields.values())
    cursor = conn.execute(
        f"INSERT INTO evidence ({','.join(columns)}) VALUES ({placeholders})", values
    )
    conn.commit()
    evidence_id = cursor.lastrowid
    conn.close()
    return evidence_id


def update_evidence(evidence_id, fields):
    conn = get_db()
    set_clause = ",".join(f"{k} = ?" for k in fields.keys())
    values = list(fields.values()) + [evidence_id]
    conn.execute(f"UPDATE evidence SET {set_clause} WHERE id = ?", values)
    conn.commit()
    conn.close()


def get_evidence(evidence_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM evidence WHERE id = ?", (evidence_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_evidence(case_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM evidence WHERE case_id = ? ORDER BY id DESC", (case_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_events(case_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM events WHERE case_id = ? ORDER BY id DESC", (case_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_sources(case_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM sources WHERE case_id = ? ORDER BY id DESC", (case_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_findings(case_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM findings WHERE case_id = ? ORDER BY id DESC", (case_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_batch_item(batch_id, case_id, filename, status, evidence_id=None, kind=None,
                    ai_fake_score=None, faces=None, recurrence=None, error=None):
    conn = get_db()
    conn.execute(
        "INSERT INTO batch_items (batch_id, case_id, time, filename, status, evidence_id, kind, "
        "ai_fake_score, faces, recurrence, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (batch_id, case_id, current_time(), filename, status, evidence_id, kind,
         ai_fake_score, faces, recurrence, error),
    )
    conn.commit()
    conn.close()


def list_batch_items(batch_id):
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM batch_items WHERE batch_id = ? ORDER BY id ASC", (batch_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def find_phash_matches(phash, exclude_evidence_id, max_distance=12):
    """Recurrence check: does this perceptual hash (near-)match evidence already
    seen anywhere in the system, including other cases? Returns matches sorted
    by similarity."""
    conn = get_db()
    rows = conn.execute(
        "SELECT id, case_id, filename, sha256, phash, received_at FROM evidence "
        "WHERE phash IS NOT NULL AND id != ?",
        (exclude_evidence_id,),
    ).fetchall()
    conn.close()
    matches = []
    for row in rows:
        try:
            distance = (int(phash, 16) ^ int(row["phash"], 16)).bit_count()
        except Exception:
            continue
        if distance <= max_distance:
            match = dict(row)
            match["distance"] = distance
            matches.append(match)
    matches.sort(key=lambda m: m["distance"])
    return matches
