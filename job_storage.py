"""Transactional incremental storage. state.json is a small discovery manifest."""
import json
import sqlite3


def connect(directory):
    db = sqlite3.connect(directory / "progress.sqlite3", timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, data TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS rows (id INTEGER PRIMARY KEY, data TEXT NOT NULL, words TEXT, label TEXT, parts TEXT)")
    return db


def save_job(job, metadata):
    db = connect(job.directory)
    row_start = min(getattr(job, "_stored_rows", 0), len(job.rows))
    label_start = min(getattr(job, "_stored_labels", 0), job.speaker_done)
    try:
        with db:
            db.execute("DELETE FROM rows WHERE id >= ?", (len(job.rows),))
            for i in range(row_start, len(job.rows)):
                db.execute("INSERT OR REPLACE INTO rows(id,data,words) VALUES(?,?,?)",
                           (i, json.dumps(job.rows[i], ensure_ascii=False), json.dumps(job.words[i], ensure_ascii=False)))
            # Rollback of a partial transcription invalidates any old labels after this point.
            db.execute("UPDATE rows SET label=NULL, parts=NULL WHERE id >= ? AND label IS NOT NULL", (job.speaker_done,))
            for i in range(label_start, job.speaker_done):
                db.execute("UPDATE rows SET label=?,parts=? WHERE id=?",
                           (job.speaker_labels[i], json.dumps(job.speaker_parts[i], ensure_ascii=False), i))
            db.execute("INSERT OR REPLACE INTO metadata VALUES(1,?)", (json.dumps(metadata, ensure_ascii=False),))
        job._stored_rows = len(job.rows)
        job._stored_labels = job.speaker_done
    finally:
        db.close()


def read_job(directory):
    path = directory / "progress.sqlite3"
    if not path.exists():
        return json.loads((directory / "state.json").read_text(encoding="utf-8"))
    db = sqlite3.connect(path)
    try:
        db.execute("BEGIN")
        if not db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone():
            return json.loads((directory / "state.json").read_text(encoding="utf-8"))
        saved = db.execute("SELECT data FROM metadata WHERE id=1").fetchone()
        if saved is None:  # Interrupted before the first migration transaction committed.
            return json.loads((directory / "state.json").read_text(encoding="utf-8"))
        data = json.loads(saved[0])
        rows = db.execute("SELECT data,words,label,parts FROM rows ORDER BY id").fetchall()
        data["rows"] = [json.loads(row[0]) for row in rows]
        data["words"] = [json.loads(row[1] or "[]") for row in rows]
        done = min(data.get("speaker_done", 0), len(rows))
        data["speaker_labels"] = [row[2] for row in rows[:done]]
        data["speaker_parts"] = [json.loads(row[3] or "[]") for row in rows[:done]]
        return data
    finally:
        db.close()
