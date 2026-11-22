"""Thread-safe SQLite state store with WAL mode and crash-recovery support."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import DEFAULT_DB_PATH

VALID_STATUSES: tuple[str, ...] = (
    "queued",
    "downloading",
    "converting",
    "transcribing",
    "cleaning",
    "done",
    "paused",
    "failed",
    "cancelled",
)

ACTIVE_STATUSES: tuple[str, ...] = (
    "downloading",
    "converting",
    "transcribing",
    "cleaning",
)

_MUTABLE_JOB_FIELDS: frozenset[str] = frozenset(
    {
        "course_id",
        "source_type",
        "url",
        "source_id",
        "title",
        "seq_num",
        "status",
        "stage",
        "progress",
        "speed_bps",
        "eta_seconds",
        "bytes_downloaded",
        "total_bytes",
        "audio_path",
        "output_path",
        "retries",
        "error_message",
    }
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Thread-safe SQLite database for courses and transcription jobs."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA foreign_keys=ON;")
            cur.execute("PRAGMA busy_timeout=15000;")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS courses (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    course_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    url TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    seq_num INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'queued',
                    stage TEXT NOT NULL DEFAULT 'queued',
                    progress REAL NOT NULL DEFAULT 0.0,
                    speed_bps REAL NOT NULL DEFAULT 0.0,
                    eta_seconds INTEGER,
                    bytes_downloaded INTEGER NOT NULL DEFAULT 0,
                    total_bytes INTEGER NOT NULL DEFAULT 0,
                    audio_path TEXT NOT NULL DEFAULT '',
                    output_path TEXT NOT NULL DEFAULT '',
                    retries INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_course_id ON jobs(course_id)"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(course_id, source_type, source_id)"
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection safely."""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Courses
    # ------------------------------------------------------------------

    def create_or_get_course(self, name: str, output_dir: str) -> dict[str, Any]:
        """Return an existing course with ``name`` or create a new one."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Course name must not be empty")
        clean_dir = str(output_dir).strip()
        if not clean_dir:
            raise ValueError("Course output_dir must not be empty")

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM courses WHERE name = ? LIMIT 1",
                (clean_name,),
            ).fetchone()
            if row is not None:
                return dict(row)

            course_id = uuid.uuid4().hex[:12]
            now = _utc_now_iso()
            self._conn.execute(
                "INSERT INTO courses (id, name, output_dir, created_at) VALUES (?, ?, ?, ?)",
                (course_id, clean_name, clean_dir, now),
            )
            self._conn.commit()
            return {
                "id": course_id,
                "name": clean_name,
                "output_dir": clean_dir,
                "created_at": now,
            }

    def get_course(self, course_id: str) -> dict[str, Any] | None:
        """Fetch a single course by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM courses WHERE id = ?",
                (course_id,),
            ).fetchone()
            return dict(row) if row is not None else None

    def list_courses(self) -> list[dict[str, Any]]:
        """List all courses ordered newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM courses ORDER BY created_at DESC, rowid DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def add_job(
        self,
        course_id: str,
        source_type: str,
        url: str,
        source_id: str,
        title: str,
        seq_num: int = 0,
        status: str = "queued",
        stage: str | None = None,
        progress: float = 0.0,
        speed_bps: float = 0.0,
        eta_seconds: int | None = None,
        bytes_downloaded: int = 0,
        total_bytes: int = 0,
        audio_path: str = "",
        output_path: str = "",
        retries: int = 0,
        error_message: str = "",
        job_id: str | None = None,
    ) -> dict[str, Any]:
        """Insert a new job record and return it as a dict."""
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid job status: {status!r}")

        jid = job_id or uuid.uuid4().hex[:12]
        effective_stage = stage if stage is not None else status
        now = _utc_now_iso()

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs (
                    id, course_id, source_type, url, source_id, title, seq_num,
                    status, stage, progress, speed_bps, eta_seconds,
                    bytes_downloaded, total_bytes, audio_path, output_path,
                    retries, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    jid,
                    course_id,
                    source_type,
                    url,
                    source_id,
                    title,
                    int(seq_num),
                    status,
                    effective_stage,
                    float(progress),
                    float(speed_bps),
                    eta_seconds,
                    int(bytes_downloaded),
                    int(total_bytes),
                    audio_path,
                    output_path,
                    int(retries),
                    error_message,
                    now,
                    now,
                ),
            )
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
            return dict(row)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Fetch a single job by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            return dict(row) if row is not None else None

    def find_job_by_source(
        self, course_id: str, source_type: str, source_id: str
    ) -> dict[str, Any] | None:
        """Find an existing job in a course matching (source_type, source_id)."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM jobs
                WHERE course_id = ? AND source_type = ? AND source_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (course_id, source_type, source_id),
            ).fetchone()
            return dict(row) if row is not None else None

    def list_jobs(
        self,
        course_id: str | None = None,
        status: str | Iterable[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List jobs with optional course_id and status (single or iterable) filters."""
        clauses: list[str] = []
        params: list[Any] = []

        if course_id is not None:
            clauses.append("course_id = ?")
            params.append(course_id)

        if status is not None:
            if isinstance(status, str):
                clauses.append("status = ?")
                params.append(status)
            else:
                statuses = list(status)
                if not statuses:
                    return []
                placeholders = ",".join("?" for _ in statuses)
                clauses.append(f"status IN ({placeholders})")
                params.extend(statuses)

        sql = "SELECT * FROM jobs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY seq_num ASC, created_at ASC, rowid ASC"

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def update_job(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        """Update allowed fields on a job and return the updated record (or None)."""
        if not fields:
            return self.get_job(job_id)

        unknown = set(fields.keys()) - _MUTABLE_JOB_FIELDS
        if unknown:
            raise ValueError(f"Unknown job field(s): {sorted(unknown)}")

        if "status" in fields and fields["status"] not in VALID_STATUSES:
            raise ValueError(f"Invalid job status: {fields['status']!r}")

        fields["updated_at"] = _utc_now_iso()
        assignments = ", ".join(f"{col} = ?" for col in fields.keys())
        values = list(fields.values()) + [job_id]

        with self._lock:
            cur = self._conn.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",
                values,
            )
            self._conn.commit()
            if cur.rowcount == 0:
                return None
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
            return dict(row) if row is not None else None

    def reset_interrupted_jobs(self) -> int:
        """Move any in-flight jobs back to 'queued' so interrupted work auto-resumes on restart."""
        now = _utc_now_iso()
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        with self._lock:
            cur = self._conn.execute(
                f"""
                UPDATE jobs
                SET status = 'queued',
                    stage = 'queued',
                    speed_bps = 0.0,
                    eta_seconds = NULL,
                    updated_at = ?
                WHERE status IN ({placeholders})
                """,
                [now, *ACTIVE_STATUSES],
            )
            self._conn.commit()
            return cur.rowcount

    def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID. Returns True if a row was deleted."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def pause_job(self, job_id: str) -> dict[str, Any] | None:
        """Mark a non-terminal job as paused."""
        with self._lock:
            job = self.get_job(job_id)
            if job is None:
                return None
            if job["status"] in ("done", "cancelled"):
                return job
            return self.update_job(
                job_id,
                status="paused",
                stage="paused",
                speed_bps=0.0,
                eta_seconds=None,
            )

    def resume_job(self, job_id: str) -> dict[str, Any] | None:
        """Move a paused job back into the queue."""
        with self._lock:
            job = self.get_job(job_id)
            if job is None:
                return None
            if job["status"] != "paused":
                return job
            return self.update_job(
                job_id,
                status="queued",
                stage="queued",
                error_message="",
                speed_bps=0.0,
                eta_seconds=None,
            )

    def retry_job(self, job_id: str) -> dict[str, Any] | None:
        """Re-queue a failed, cancelled, or paused job with cleared error state."""
        with self._lock:
            job = self.get_job(job_id)
            if job is None:
                return None
            if job["status"] == "done":
                return job
            return self.update_job(
                job_id,
                status="queued",
                stage="queued",
                retries=0,
                error_message="",
                speed_bps=0.0,
                eta_seconds=None,
            )

    def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        """Mark a non-completed job as cancelled."""
        with self._lock:
            job = self.get_job(job_id)
            if job is None:
                return None
            if job["status"] == "done":
                return job
            return self.update_job(
                job_id,
                status="cancelled",
                stage="cancelled",
                speed_bps=0.0,
                eta_seconds=None,
            )

    def get_stats(self, course_id: str | None = None) -> dict[str, int]:
        """Return aggregate job counts for UI/CLI dashboards."""
        jobs = self.list_jobs(course_id=course_id)
        stats = {
            "total": len(jobs),
            "queued": 0,
            "active": 0,
            "done": 0,
            "paused": 0,
            "failed": 0,
            "cancelled": 0,
        }
        for job in jobs:
            st = job["status"]
            if st in ACTIVE_STATUSES:
                stats["active"] += 1
            elif st in stats:
                stats[st] += 1
        return stats
