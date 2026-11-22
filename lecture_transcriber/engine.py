"""Decoupled concurrent orchestration engine for Lecture Transcriber.

Architecture:
- Stage 1 Pool (`download_workers`): Downloads YouTube/Google Drive audio (with
  resumable `.part` files) and converts/splits into mono 16kHz MP3 chunks via FFmpeg.
- Stage 2 Pool (`transcribe_workers`): Uploads chunks to Gemini, transcribes with
  multi-model fallback & timestamp shifting, cleans hallucination loops, writes
  Markdown output, and regenerates course summary + NotebookLM master files.
- Dispatcher Loop: Coordinates queued jobs -> Stage 1 -> Stage 2 without slow
  downloads ever blocking AI transcription of already-downloaded lectures.
- Network Watchdog: Monitors connectivity via `NetworkStatusMonitor`; on outage,
  emits `offline`, waits gracefully without failing jobs, and resumes on `online`.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from lecture_transcriber.config import (
    MEDIA_EXTENSIONS,
    AppConfig,
    ConfigManager,
    clean_filename,
    strip_media_extension,
)
from lecture_transcriber.database import ACTIVE_STATUSES, Database
from lecture_transcriber.downloader import (
    AudioProcessingError,
    DownloadCancelledError,
    DownloadError,
    DownloadPausedError,
    GDrivePermissionError,
    GDriveQuotaError,
    ParsedMediaItem,
    URLParseError,
    cleanup_audio_files,
    download_media,
    expand_urls,
    merge_audio_chunks_to_mp3,
    prepare_audio_for_transcription,
)
from lecture_transcriber.network import NetworkStatusMonitor, is_online, wait_for_internet
from lecture_transcriber.transcriber import (
    GeminiAuthError,
    GeminiQuotaError,
    GeminiTranscriber,
    TranscriptionCancelledError,
    TranscriptionError,
    TranscriptionPausedError,
    clean_transcript_body,
    format_lecture_markdown,
    rebuild_course_artifacts,
)

# Minimum size (bytes) of an existing .md file to be considered already completed
EXISTING_MD_MIN_BYTES = 1024

_LEADING_SEQ_RE = re.compile(r"^\d{1,4}[\s._\-]")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp_path, path)


def _build_output_path(output_dir: str, seq_num: int, title: str) -> str:
    """Build deterministic Windows-safe Markdown filename for a lecture."""
    clean_title = strip_media_extension(title or "")
    raw_title = (clean_title or f"Lecture_{seq_num:03d}").strip()
    safe_title = clean_filename(raw_title)
    if _LEADING_SEQ_RE.match(raw_title) or _LEADING_SEQ_RE.match(safe_title):
        filename = f"{safe_title}.md"
    else:
        filename = f"{seq_num:03d}_{safe_title}.md"
    return os.path.join(output_dir, filename)


def _find_existing_completed_md(output_dir: str, seq_num: int, title: str) -> str | None:
    """Return path if an already-completed (>1KB) .md file exists for this lecture."""
    clean_title = strip_media_extension(title or "")
    raw_title = (title or f"Lecture_{seq_num:03d}").strip()
    legacy_safe = clean_filename(raw_title)
    if _LEADING_SEQ_RE.match(raw_title) or _LEADING_SEQ_RE.match(legacy_safe):
        legacy_numbered = f"{legacy_safe}.md"
    else:
        legacy_numbered = f"{seq_num:03d}_{legacy_safe}.md"

    candidates = [
        _build_output_path(output_dir, seq_num, title),
        os.path.join(output_dir, f"{clean_filename(clean_title)}.md"),
        os.path.join(output_dir, legacy_numbered),
        os.path.join(output_dir, f"{legacy_safe}.md"),
    ]
    seen: set[str] = set()
    for path in candidates:
        norm = os.path.normcase(os.path.abspath(path))
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isfile(path) and os.path.getsize(path) > EXISTING_MD_MIN_BYTES:
            return path
    return None


def _parse_stored_chunks(audio_path_field: str) -> list[dict[str, Any]]:
    """Parse chunk metadata stored in `job['audio_path']` if all chunk files still exist."""
    raw = (audio_path_field or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list) and data:
                valid: list[dict[str, Any]] = []
                for entry in data:
                    if isinstance(entry, dict) and entry.get("path"):
                        p = str(entry["path"])
                        if os.path.isfile(p) and os.path.getsize(p) > 0:
                            valid.append(entry)
                        else:
                            return []
                return valid
        except Exception:
            return []
    # Single audio file path fallback
    if os.path.isfile(raw) and os.path.getsize(raw) > 0:
        return [{"path": raw, "offset_seconds": 0, "duration_seconds": 0.0}]
    return []


def _cleanup_job_temp_artifacts(temp_dir: str, source_id: str, chunks: list[dict[str, Any]] | None = None) -> None:
    """Remove converted chunks and any leftover temp/partial files for a cancelled/completed job."""
    if chunks:
        try:
            cleanup_audio_files(chunks)
        except Exception:
            pass
    if temp_dir and source_id and os.path.isdir(temp_dir):
        prefix = clean_filename(source_id)
        raw_prefix = str(source_id)
        try:
            for entry in os.listdir(temp_dir):
                if entry.startswith(prefix) or entry.startswith(raw_prefix):
                    full = os.path.join(temp_dir, entry)
                    if os.path.isfile(full):
                        try:
                            os.remove(full)
                        except OSError:
                            pass
        except OSError:
            pass


class TranscriptionEngine:
    """Concurrent two-stage download & Gemini transcription engine."""

    def __init__(
        self,
        config_manager: ConfigManager,
        db: Database,
        *,
        transcriber_factory: Callable[[AppConfig], Any] | None = None,
        network_monitor: NetworkStatusMonitor | None = None,
        download_fn: Callable[..., str] | None = None,
        prepare_audio_fn: Callable[..., list[dict[str, Any]]] | None = None,
        expand_urls_fn: Callable[..., list[ParsedMediaItem]] | None = None,
        network_check_interval: float = 5.0,
    ) -> None:
        self.config_manager = config_manager
        self.db = db

        self._transcriber_factory = transcriber_factory
        self._download_fn = download_fn
        self._prepare_audio_fn = prepare_audio_fn
        self._expand_urls_fn = expand_urls_fn

        self._lock = threading.RLock()
        self._course_locks: dict[str, threading.Lock] = {}

        # Lifecycle state
        self._running = False
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._dispatcher_thread: threading.Thread | None = None

        # Bounded decoupled worker pools
        self._download_executor: ThreadPoolExecutor | None = None
        self._transcribe_executor: ThreadPoolExecutor | None = None
        self._download_pool_size: int = 0
        self._transcribe_pool_size: int = 0

        # In-flight job tracking
        self._active_download_ids: set[str] = set()
        self._active_transcribe_ids: set[str] = set()
        self._ready_queue: deque[tuple[str, list[dict[str, Any]]]] = deque()
        self._ready_ids: set[str] = set()
        self._job_chunks: dict[str, list[dict[str, Any]]] = {}

        # Per-job pause & cancel events
        self._pause_events: dict[str, threading.Event] = {}
        self._cancel_events: dict[str, threading.Event] = {}

        # Pub-sub event subscribers (queues and/or callbacks)
        self._sub_lock = threading.RLock()
        self._queue_subscribers: list[queue.Queue[dict[str, Any]]] = []
        self._cb_subscribers: list[Callable[[dict[str, Any]], Any]] = []

        # Network connectivity monitor
        self.network_monitor = network_monitor or NetworkStatusMonitor(
            check_interval=network_check_interval,
            on_change=self._on_network_change,
        )
        if network_monitor is not None:
            self.network_monitor.subscribe(self._on_network_change)
        self._online: bool = bool(getattr(self.network_monitor, "is_online", True))

    # ------------------------------------------------------------------
    # Properties & Pub-Sub
    # ------------------------------------------------------------------

    @property
    def is_online(self) -> bool:
        return self._online

    @property
    def is_running(self) -> bool:
        return self._running

    def subscribe(
        self,
        callback: Callable[[dict[str, Any]], Any] | None = None,
        *,
        maxsize: int = 512,
    ) -> queue.Queue[dict[str, Any]] | Callable[[dict[str, Any]], Any]:
        """Subscribe to engine events.

        - Called with no arguments: returns a bounded ``queue.Queue`` receiving event dicts.
        - Called with a callback: registers the callback and returns it.
        """
        with self._sub_lock:
            if callback is not None:
                if callback not in self._cb_subscribers:
                    self._cb_subscribers.append(callback)
                return callback
            q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(16, int(maxsize)))
            self._queue_subscribers.append(q)
            return q

    def unsubscribe(self, subscriber: Any) -> None:
        """Unsubscribe a previously registered queue or callback."""
        with self._sub_lock:
            if subscriber in self._queue_subscribers:
                self._queue_subscribers.remove(subscriber)
            if subscriber in self._cb_subscribers:
                self._cb_subscribers.remove(subscriber)

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Broadcast a lightweight JSON-serializable event to all subscribers."""
        event = {
            "type": event_type,
            "timestamp": _utc_now_iso(),
            "data": data,
        }
        with self._sub_lock:
            queues = list(self._queue_subscribers)
            callbacks = list(self._cb_subscribers)

        for q in queues:
            try:
                q.put_nowait(event)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass

    def _emit_job(self, job: dict[str, Any] | None) -> None:
        if job is not None:
            self._emit("job_update", {"job": job, "job_id": job["id"]})

    def _emit_log(self, message: str, level: str = "info", job_id: str | None = None) -> None:
        payload: dict[str, Any] = {"level": level, "message": message}
        if job_id:
            payload["job_id"] = job_id
        self._emit("log", payload)

    # ------------------------------------------------------------------
    # Control Event Registry
    # ------------------------------------------------------------------

    def _get_control_events(self, job_id: str) -> tuple[threading.Event, threading.Event]:
        with self._lock:
            pause_ev = self._pause_events.get(job_id)
            if pause_ev is None:
                pause_ev = threading.Event()
                self._pause_events[job_id] = pause_ev
            cancel_ev = self._cancel_events.get(job_id)
            if cancel_ev is None:
                cancel_ev = threading.Event()
                self._cancel_events[job_id] = cancel_ev
            return pause_ev, cancel_ev

    def _get_course_lock(self, course_id: str) -> threading.Lock:
        with self._lock:
            lk = self._course_locks.get(course_id)
            if lk is None:
                lk = threading.Lock()
                self._course_locks[course_id] = lk
            return lk

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_executors(self, cfg: AppConfig) -> None:
        dl_workers = max(1, int(cfg.download_workers))
        tr_workers = max(1, int(cfg.transcribe_workers))

        if self._download_executor is None or dl_workers > self._download_pool_size:
            old_dl = self._download_executor
            self._download_executor = ThreadPoolExecutor(
                max_workers=dl_workers,
                thread_name_prefix="dl-worker",
            )
            self._download_pool_size = dl_workers
            if old_dl is not None:
                old_dl.shutdown(wait=False)

        if self._transcribe_executor is None or tr_workers > self._transcribe_pool_size:
            old_tr = self._transcribe_executor
            self._transcribe_executor = ThreadPoolExecutor(
                max_workers=tr_workers,
                thread_name_prefix="ai-worker",
            )
            self._transcribe_pool_size = tr_workers
            if old_tr is not None:
                old_tr.shutdown(wait=False)

    def start(self) -> None:
        """Start the engine, recover interrupted jobs, and begin background dispatching."""
        with self._lock:
            if self._running:
                return

            self.config_manager.ensure_directories()
            recovered = self.db.reset_interrupted_jobs()
            if recovered > 0:
                self._emit_log(
                    f"تمت استعادة {recovered} مهمة متوقفة للمتابعة التلقائية (Recovered {recovered} interrupted job(s)).",
                    "info",
                )

            cfg = self.config_manager.get()
            self._stop_event.clear()
            self._wake_event.clear()
            self._ensure_executors(cfg)

            self._running = True
            self.network_monitor.start()
            self._online = bool(getattr(self.network_monitor, "is_online", True))

            self._dispatcher_thread = threading.Thread(
                target=self._dispatcher_loop,
                name="TranscriptionDispatcher",
                daemon=True,
            )
            self._dispatcher_thread.start()
            self._wake_event.set()

    def stop(self, wait: bool = True, timeout: float = 5.0) -> None:
        """Stop the engine gracefully without corrupting in-progress job state."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_event.set()
            self._wake_event.set()

            # Signal active workers via pause events so blocking calls unwind promptly
            # (we do NOT overwrite their DB status to 'paused' on shutdown so next start auto-resumes).
            for jid in list(self._active_download_ids | self._active_transcribe_ids):
                ev = self._pause_events.get(jid)
                if ev is not None:
                    ev.set()

            dispatcher = self._dispatcher_thread
            dl_exec = self._download_executor
            tr_exec = self._transcribe_executor
            self._dispatcher_thread = None
            self._download_executor = None
            self._transcribe_executor = None
            self._download_pool_size = 0
            self._transcribe_pool_size = 0

        try:
            self.network_monitor.stop(timeout=min(2.0, timeout))
        except Exception:
            pass

        if dispatcher is not None and dispatcher.is_alive():
            dispatcher.join(timeout=timeout)

        if dl_exec is not None:
            dl_exec.shutdown(wait=wait, cancel_futures=True)
        if tr_exec is not None:
            tr_exec.shutdown(wait=wait, cancel_futures=True)

    def reload_config(self) -> AppConfig:
        """Reload configuration from ConfigManager and adjust worker pool limits."""
        cfg = self.config_manager.get()
        self.config_manager.ensure_directories()
        with self._lock:
            if self._running:
                self._ensure_executors(cfg)
                self._wake_event.set()
        self._emit_log("تم تحديث الإعدادات وتطبيقها بنجاح.", "info")
        return cfg

    # ------------------------------------------------------------------
    # Network Connectivity Handling
    # ------------------------------------------------------------------

    def _on_network_change(self, online: bool) -> None:
        with self._lock:
            prev = self._online
            self._online = bool(online)
            if online == prev:
                return

        if not online:
            self._emit(
                "offline",
                {
                    "online": False,
                    "message": "انقطع الاتصال بالإنترنت — بانتظار عودة الاتصال لاستئناف العمل تلقائياً...",
                },
            )
        else:
            self._emit(
                "online",
                {
                    "online": True,
                    "message": "عاد الاتصال بالإنترنت — جاري استئناف التحميل والتفريغ تلقائياً.",
                },
            )
            self._wake_event.set()

    def _wait_until_online(
        self,
        job_id: str,
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> bool:
        """Wait gracefully if offline without failing the job. Returns True if back online."""
        if self._online and is_online():
            return True

        if self._online:
            self._on_network_change(False)

        self._emit_log(
            "انقطع الاتصال بالإنترنت — المهمة في انتظار عودة الاتصال...",
            "warning",
            job_id=job_id,
        )

        stop_wait = threading.Event()

        def _watch() -> None:
            while not stop_wait.is_set():
                if self._stop_event.is_set() or cancel_event.is_set() or pause_event.is_set():
                    stop_wait.set()
                    return
                stop_wait.wait(timeout=0.15)

        t = threading.Thread(target=_watch, daemon=True)
        t.start()
        try:
            reconnected = wait_for_internet(stop_event=stop_wait, check_interval=1.5)
        finally:
            stop_wait.set()
            t.join(timeout=0.5)

        if reconnected and not self._online:
            self._on_network_change(True)
        return reconnected

    # ------------------------------------------------------------------
    # Public Queue & Job Control API
    # ------------------------------------------------------------------

    def enqueue_urls(
        self,
        course_name: str,
        urls: list[str] | str,
        *,
        fetch_metadata: bool = True,
        include_playlist: bool = False,
    ) -> dict[str, Any]:
        """Expand YouTube/Drive URLs, skip completed/duplicate items, enqueue into SQLite, and wake dispatcher."""
        clean_course_name = (course_name or "").strip()
        if not clean_course_name:
            raise ValueError("course_name must not be empty")

        if isinstance(urls, str):
            url_list = [line.strip() for line in urls.splitlines() if line.strip() and not line.strip().startswith("#")]
        else:
            url_list = [str(u).strip() for u in (urls or []) if str(u).strip() and not str(u).strip().startswith("#")]

        if not url_list:
            raise ValueError("At least one URL is required")

        cfg = self.config_manager.get()
        course_folder = clean_filename(clean_course_name)
        course_output_dir = os.path.join(cfg.output_base_dir, course_folder)
        os.makedirs(course_output_dir, exist_ok=True)

        course = self.db.create_or_get_course(clean_course_name, course_output_dir)
        course_output_dir = course["output_dir"]
        os.makedirs(course_output_dir, exist_ok=True)
        self._emit("course_updated", {"course": course})

        expander = self._expand_urls_fn or expand_urls
        items: list[ParsedMediaItem] = []
        errors: list[str] = []

        try:
            expanded = expander(
                url_list,
                config=cfg,
                fetch_metadata=fetch_metadata,
                include_playlist=include_playlist,
            )
            items.extend(expanded)
        except Exception as batch_exc:
            if len(url_list) <= 1:
                errors.append(str(batch_exc))
            else:
                # Fall back to per-URL expansion so valid URLs in the batch still succeed
                seen_keys: set[tuple[str, str]] = set()
                for single_url in url_list:
                    try:
                        sub_items = expander(
                            [single_url],
                            config=cfg,
                            fetch_metadata=fetch_metadata,
                            include_playlist=include_playlist,
                        )
                        for it in sub_items:
                            key = (it.source_type, it.source_id)
                            if key not in seen_keys:
                                seen_keys.add(key)
                                items.append(it)
                    except Exception as item_exc:
                        errors.append(f"{single_url}: {item_exc}")

        existing_jobs = self.db.list_jobs(course_id=course["id"])
        next_seq = max((int(j.get("seq_num") or 0) for j in existing_jobs), default=0)

        added_jobs: list[dict[str, Any]] = []
        skipped_existing = 0
        skipped_duplicates = 0

        for item in items:
            # 1. Check duplicate job already in DB for this course
            existing = self.db.find_job_by_source(course["id"], item.source_type, item.source_id)
            if existing is not None and existing["status"] not in ("cancelled",):
                skipped_duplicates += 1
                continue

            next_seq += 1
            seq_num = next_seq
            target_md = _build_output_path(course_output_dir, seq_num, item.title)

            # 2. Skip if an already-completed .md file (>1KB) already exists on disk
            completed_path = _find_existing_completed_md(course_output_dir, seq_num, item.title)
            if completed_path is not None:
                skipped_existing += 1
                self._emit_log(
                    f"تخطي '{item.title}' — ملف التفريغ مكتمل مسبقاً ({os.path.basename(completed_path)}).",
                    "info",
                )
                continue

            job = self.db.add_job(
                course_id=course["id"],
                source_type=item.source_type,
                url=item.url,
                source_id=item.source_id,
                title=item.title,
                seq_num=seq_num,
                status="queued",
                stage="queued",
                output_path=target_md,
            )
            added_jobs.append(job)
            self._emit_job(job)

        if added_jobs:
            self._emit_log(
                f"تمت إضافة {len(added_jobs)} محاضرة إلى مادة '{clean_course_name}'.",
                "info",
            )
            self._wake_event.set()

        total_skipped = skipped_existing + skipped_duplicates
        return {
            "course": course,
            "added": len(added_jobs),
            "skipped": total_skipped,
            "skipped_existing": skipped_existing,
            "skipped_duplicates": skipped_duplicates,
            "jobs": added_jobs,
            "job_ids": [j["id"] for j in added_jobs],
            "total_expanded": len(items),
            "errors": errors,
        }

    def pause_job(self, job_id: str) -> dict[str, Any] | None:
        """Pause a queued or active job (preserves `.part` / converted chunk files)."""
        pause_ev, _ = self._get_control_events(job_id)
        pause_ev.set()

        # Remove from pending Stage 2 ready queue if sitting there
        with self._lock:
            if job_id in self._ready_ids:
                self._ready_ids.discard(job_id)
                self._ready_queue = deque(item for item in self._ready_queue if item[0] != job_id)

        updated = self.db.pause_job(job_id)
        self._emit_job(updated)
        if updated:
            self._emit_log(f"تم إيقاف المهمة مؤقتاً: {updated['title']}", "info", job_id=job_id)
        return updated

    def resume_job(self, job_id: str) -> dict[str, Any] | None:
        """Resume a paused job from its last checkpoint."""
        pause_ev, cancel_ev = self._get_control_events(job_id)
        pause_ev.clear()
        cancel_ev.clear()

        updated = self.db.resume_job(job_id)
        self._emit_job(updated)
        if updated:
            self._emit_log(f"تم استئناف المهمة: {updated['title']}", "info", job_id=job_id)
            self._wake_event.set()
        return updated

    def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        """Cancel a job and clean up its temporary audio files."""
        pause_ev, cancel_ev = self._get_control_events(job_id)
        pause_ev.clear()
        cancel_ev.set()

        with self._lock:
            if job_id in self._ready_ids:
                self._ready_ids.discard(job_id)
                self._ready_queue = deque(item for item in self._ready_queue if item[0] != job_id)
            is_active = job_id in self._active_download_ids or job_id in self._active_transcribe_ids
            chunks = self._job_chunks.pop(job_id, None)

        job_before = self.db.get_job(job_id)
        updated = self.db.cancel_job(job_id)

        if not is_active and job_before is not None:
            cfg = self.config_manager.get()
            stored_chunks = chunks or _parse_stored_chunks(job_before.get("audio_path", ""))
            _cleanup_job_temp_artifacts(cfg.temp_dir, job_before.get("source_id", ""), stored_chunks)
            if updated is not None and updated.get("audio_path"):
                updated = self.db.update_job(job_id, audio_path="")

        self._emit_job(updated)
        if updated:
            self._emit_log(f"تم إلغاء المهمة: {updated['title']}", "warning", job_id=job_id)
        return updated

    def retry_job(self, job_id: str) -> dict[str, Any] | None:
        """Reset and re-queue a failed, cancelled, or paused job."""
        pause_ev, cancel_ev = self._get_control_events(job_id)
        pause_ev.clear()
        cancel_ev.clear()

        updated = self.db.retry_job(job_id)
        self._emit_job(updated)
        if updated:
            self._emit_log(f"إعادة محاولة المهمة: {updated['title']}", "info", job_id=job_id)
            self._wake_event.set()
        return updated

    def delete_job(self, job_id: str) -> bool:
        """Cancel if needed, clean temp files, and delete the job record."""
        job = self.db.get_job(job_id)
        if job is None:
            return False

        if job["status"] not in ("done", "cancelled"):
            self.cancel_job(job_id)
        else:
            cfg = self.config_manager.get()
            with self._lock:
                chunks = self._job_chunks.pop(job_id, None)
            stored_chunks = chunks or _parse_stored_chunks(job.get("audio_path", ""))
            if stored_chunks:
                _cleanup_job_temp_artifacts(cfg.temp_dir, job.get("source_id", ""), stored_chunks)

        deleted = self.db.delete_job(job_id)
        with self._lock:
            self._pause_events.pop(job_id, None)
            self._cancel_events.pop(job_id, None)
            self._job_chunks.pop(job_id, None)

        if deleted:
            self._emit("job_deleted", {"job_id": job_id})
        return deleted

    def pause_all(self) -> dict[str, Any]:
        """Pause all queued and active jobs."""
        targets = self.db.list_jobs(status=("queued", *ACTIVE_STATUSES))
        paused_ids: list[str] = []
        for job in targets:
            res = self.pause_job(job["id"])
            if res is not None and res["status"] == "paused":
                paused_ids.append(job["id"])
        return {"paused_count": len(paused_ids), "job_ids": paused_ids}

    def resume_all(self) -> dict[str, Any]:
        """Resume all paused jobs."""
        targets = self.db.list_jobs(status="paused")
        resumed_ids: list[str] = []
        for job in targets:
            res = self.resume_job(job["id"])
            if res is not None and res["status"] == "queued":
                resumed_ids.append(job["id"])
        return {"resumed_count": len(resumed_ids), "job_ids": resumed_ids}

    def get_snapshot(self) -> dict[str, Any]:
        """Return a complete JSON-serializable state snapshot for UI/CLI."""
        cfg = self.config_manager.get()
        return {
            "online": self._online,
            "running": self._running,
            "config": cfg.to_dict(mask_api_key=True),
            "courses": self.db.list_courses(),
            "jobs": self.db.list_jobs(),
            "stats": self.db.get_stats(),
        }

    # ------------------------------------------------------------------
    # Dispatcher Loop & Worker Pipelines
    # ------------------------------------------------------------------

    def _dispatcher_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._dispatch_cycle()
            except Exception as exc:
                self._emit_log(f"Dispatcher error: {exc}", "error")

            self._wake_event.wait(timeout=0.5)
            self._wake_event.clear()

    def _dispatch_cycle(self) -> None:
        if self._stop_event.is_set() or not self._online:
            return

        cfg = self.config_manager.get()
        max_dl = max(1, int(cfg.download_workers))
        max_tr = max(1, int(cfg.transcribe_workers))

        # 1. Dispatch ready converted jobs into Stage 2 (Transcribe Pool) first
        while True:
            job_item: tuple[str, list[dict[str, Any]]] | None = None
            with self._lock:
                if (
                    self._stop_event.is_set()
                    or self._transcribe_executor is None
                    or len(self._active_transcribe_ids) >= max_tr
                    or not self._ready_queue
                ):
                    break
                job_id, chunks = self._ready_queue.popleft()
                self._ready_ids.discard(job_id)

                job = self.db.get_job(job_id)
                if job is None or job["status"] in ("paused", "cancelled", "done"):
                    continue

                self._active_transcribe_ids.add(job_id)
                job_item = (job_id, chunks)
                tr_exec = self._transcribe_executor

            if job_item is not None and tr_exec is not None:
                tr_exec.submit(self._run_transcribe_stage, job_item[0], job_item[1])

        # 2. Dispatch queued jobs into Stage 1 (Download + Convert Pool)
        with self._lock:
            dl_slots = max_dl - len(self._active_download_ids)
            if self._stop_event.is_set() or self._download_executor is None or dl_slots <= 0:
                return

        queued_jobs = self.db.list_jobs(status="queued")
        for job in queued_jobs:
            if dl_slots <= 0 or self._stop_event.is_set() or not self._online:
                break

            job_id = job["id"]
            with self._lock:
                if (
                    job_id in self._active_download_ids
                    or job_id in self._active_transcribe_ids
                    or job_id in self._ready_ids
                ):
                    continue

                pause_ev, cancel_ev = self._get_control_events(job_id)
                if pause_ev.is_set() or cancel_ev.is_set():
                    continue

                # Check if chunks are already prepared on disk (e.g., resumed after pause during transcription)
                existing_chunks = self._job_chunks.get(job_id) or _parse_stored_chunks(
                    job.get("audio_path", "")
                )
                if existing_chunks:
                    self._job_chunks[job_id] = existing_chunks
                    self._ready_ids.add(job_id)
                    self._ready_queue.append((job_id, existing_chunks))
                    self._wake_event.set()
                    continue

                if self._download_executor is None:
                    break
                self._active_download_ids.add(job_id)
                dl_exec = self._download_executor
                dl_slots -= 1

            dl_exec.submit(self._run_download_stage, job_id)

    def _run_download_stage(self, job_id: str) -> None:
        """Stage 1: Download audio (with `.part` resume) and convert/split via FFmpeg."""
        raw_path: str | None = None
        chunks: list[dict[str, Any]] | None = None
        cfg = self.config_manager.get()
        pause_ev, cancel_ev = self._get_control_events(job_id)

        try:
            job = self.db.get_job(job_id)
            if job is None or job["status"] in ("paused", "cancelled", "done"):
                return
            if cancel_ev.is_set():
                return
            if pause_ev.is_set():
                return

            os.makedirs(cfg.temp_dir, exist_ok=True)
            updated = self.db.update_job(
                job_id,
                status="downloading",
                stage="downloading",
                progress=0.0,
                speed_bps=0.0,
                eta_seconds=None,
                error_message="",
            )
            self._emit_job(updated)
            self._emit_log(f"بدء التحميل: {job['title']}", "info", job_id=job_id)

            last_emit_ts = [0.0]
            last_emit_pct = [-1.0]

            def _dl_progress(
                pct: float,
                speed_bps: float,
                eta_sec: float | None,
                downloaded_bytes: int,
                total_bytes: int,
            ) -> None:
                now = time.monotonic()
                overall_pct = min(50.0, max(0.0, float(pct) * 0.50))
                if (now - last_emit_ts[0]) < 0.25 and abs(overall_pct - last_emit_pct[0]) < 1.0 and pct < 100.0:
                    return
                last_emit_ts[0] = now
                last_emit_pct[0] = overall_pct
                eta_int = int(eta_sec) if eta_sec is not None else None
                u = self.db.update_job(
                    job_id,
                    status="downloading",
                    stage="downloading",
                    progress=round(overall_pct, 1),
                    speed_bps=float(speed_bps or 0.0),
                    eta_seconds=eta_int,
                    bytes_downloaded=int(downloaded_bytes or 0),
                    total_bytes=int(total_bytes or 0),
                )
                self._emit_job(u)

            def _dl_status(status_or_msg: bool | str, elapsed: float | None = None) -> None:
                if isinstance(status_or_msg, bool):
                    online_now = status_or_msg
                    prev_online = self._online
                    if online_now != prev_online:
                        self._on_network_change(online_now)
                    if not online_now and (prev_online or elapsed == 0.0):
                        self._emit_log(
                            "انقطع الاتصال بالإنترنت أثناء التحميل — بانتظار عودة الاتصال...",
                            "warning",
                            job_id=job_id,
                        )
                    elif online_now and not prev_online:
                        self._emit_log(
                            "عاد الاتصال بالإنترنت — استئناف التحميل تلقائياً.",
                            "info",
                            job_id=job_id,
                        )
                    return

                msg = str(status_or_msg or "")
                low = msg.lower()
                if "offline" in low or "انقطع" in msg:
                    if self._online:
                        self._on_network_change(False)
                    self._emit_log(msg, "warning", job_id=job_id)
                elif "online" in low or "عاد" in msg:
                    if not self._online:
                        self._on_network_change(True)
                    self._emit_log(msg, "info", job_id=job_id)
                elif msg:
                    self._emit_log(msg, "info", job_id=job_id)

            downloader = self._download_fn or download_media
            raw_path = downloader(
                source_type=job["source_type"],
                url=job["url"],
                source_id=job["source_id"],
                temp_dir=cfg.temp_dir,
                config=cfg,
                progress_cb=_dl_progress,
                cancel_event=cancel_ev,
                pause_event=pause_ev,
                on_status=_dl_status,
            )

            if cancel_ev.is_set():
                raise DownloadCancelledError("Job cancelled after download")
            if pause_ev.is_set():
                raise DownloadPausedError("Job paused after download")

            # Stage 1b: Convert & chunk audio with FFmpeg
            updated = self.db.update_job(
                job_id,
                status="converting",
                stage="converting",
                progress=52.0,
                speed_bps=0.0,
                eta_seconds=None,
            )
            self._emit_job(updated)
            self._emit_log(f"جاري تحويل وتقسيم الصوت: {job['title']}", "info", job_id=job_id)

            preparer = self._prepare_audio_fn or prepare_audio_for_transcription
            chunks = preparer(
                raw_path,
                cfg.temp_dir,
                job["source_id"],
                config=cfg,
                cancel_event=cancel_ev,
            )

            if cancel_ev.is_set():
                raise DownloadCancelledError("Job cancelled after conversion")
            if pause_ev.is_set():
                # Save converted chunks so resume skips download+convert!
                self.db.update_job(job_id, audio_path=json.dumps(chunks, ensure_ascii=False), progress=60.0)
                with self._lock:
                    self._job_chunks[job_id] = chunks
                raise DownloadPausedError("Job paused after conversion")

            # Persist chunk metadata & hand off to Stage 2 (Transcribe Pool)
            updated = self.db.update_job(
                job_id,
                audio_path=json.dumps(chunks, ensure_ascii=False),
                progress=60.0,
                speed_bps=0.0,
                eta_seconds=None,
            )
            self._emit_job(updated)

            with self._lock:
                self._job_chunks[job_id] = chunks
                self._ready_ids.add(job_id)
                self._ready_queue.append((job_id, chunks))
            self._wake_event.set()

        except DownloadPausedError:
            if not self._stop_event.is_set():
                updated = self.db.pause_job(job_id)
                self._emit_job(updated)

        except DownloadCancelledError:
            if raw_path and os.path.isfile(raw_path):
                try:
                    os.remove(raw_path)
                except OSError:
                    pass
            job_cur = self.db.get_job(job_id)
            _cleanup_job_temp_artifacts(cfg.temp_dir, (job_cur or {}).get("source_id", ""), chunks)
            updated = self.db.cancel_job(job_id)
            if updated is not None:
                updated = self.db.update_job(job_id, audio_path="")
            self._emit_job(updated)

        except (GDrivePermissionError, GDriveQuotaError) as non_retry_exc:
            updated = self.db.update_job(
                job_id,
                status="failed",
                stage="failed",
                speed_bps=0.0,
                eta_seconds=None,
                error_message=str(non_retry_exc),
            )
            self._emit_job(updated)
            self._emit_log(f"فشل التحميل: {non_retry_exc}", "error", job_id=job_id)
            self._rebuild_artifacts_for_job(updated)

        except Exception as exc:
            if self._stop_event.is_set():
                return
            if cancel_ev.is_set():
                updated = self.db.cancel_job(job_id)
                self._emit_job(updated)
                return
            if pause_ev.is_set():
                updated = self.db.pause_job(job_id)
                self._emit_job(updated)
                return

            # If failure happened due to offline state, wait for internet and re-queue gracefully!
            if not self._online or not is_online():
                reconnected = self._wait_until_online(job_id, cancel_ev, pause_ev)
                if self._stop_event.is_set():
                    return
                if cancel_ev.is_set():
                    if raw_path and os.path.isfile(raw_path):
                        try:
                            os.remove(raw_path)
                        except OSError:
                            pass
                    job_cur = self.db.get_job(job_id)
                    _cleanup_job_temp_artifacts(cfg.temp_dir, (job_cur or {}).get("source_id", ""), chunks)
                    updated = self.db.cancel_job(job_id)
                    if updated is not None:
                        updated = self.db.update_job(job_id, audio_path="")
                    self._emit_job(updated)
                    return
                if pause_ev.is_set():
                    updated = self.db.pause_job(job_id)
                    self._emit_job(updated)
                    return
                if reconnected:
                    updated = self.db.update_job(
                        job_id,
                        status="queued",
                        stage="queued",
                        speed_bps=0.0,
                        eta_seconds=None,
                        error_message="",
                    )
                    self._emit_job(updated)
                    self._wake_event.set()
                    return

            # Retryable transient failure check
            job_cur = self.db.get_job(job_id)
            retries = int((job_cur or {}).get("retries", 0)) + 1
            if retries <= max(1, int(cfg.max_retries)) and not isinstance(exc, AudioProcessingError):
                updated = self.db.update_job(
                    job_id,
                    status="queued",
                    stage="queued",
                    retries=retries,
                    speed_bps=0.0,
                    eta_seconds=None,
                    error_message=f"إعادة محاولة ({retries}/{cfg.max_retries}): {exc}",
                )
                self._emit_job(updated)
                self._emit_log(
                    f"خطأ مؤقت أثناء التحميل (محاولة {retries}/{cfg.max_retries}): {exc}",
                    "warning",
                    job_id=job_id,
                )
                self._wake_event.set()
            else:
                if raw_path and os.path.isfile(raw_path):
                    try:
                        os.remove(raw_path)
                    except OSError:
                        pass
                updated = self.db.update_job(
                    job_id,
                    status="failed",
                    stage="failed",
                    retries=retries,
                    speed_bps=0.0,
                    eta_seconds=None,
                    error_message=str(exc),
                )
                self._emit_job(updated)
                self._emit_log(f"فشل التحميل نهائياً: {exc}", "error", job_id=job_id)
                self._rebuild_artifacts_for_job(updated)

        finally:
            with self._lock:
                self._active_download_ids.discard(job_id)
            self._wake_event.set()

    def _run_transcribe_stage(self, job_id: str, chunks: list[dict[str, Any]]) -> None:
        """Stage 2: Transcribe chunks with Gemini, clean output, save Markdown & rebuild course files."""
        cfg = self.config_manager.get()
        pause_ev, cancel_ev = self._get_control_events(job_id)
        should_cleanup_chunks = False

        try:
            job = self.db.get_job(job_id)
            if job is None or job["status"] in ("paused", "cancelled", "done"):
                if job is None or job["status"] == "cancelled":
                    should_cleanup_chunks = True
                return

            if cancel_ev.is_set():
                should_cleanup_chunks = True
                return
            if pause_ev.is_set():
                return

            # Wait if offline before hitting Gemini
            if not self._online or not is_online():
                reconnected = self._wait_until_online(job_id, cancel_ev, pause_ev)
                if not reconnected:
                    if cancel_ev.is_set():
                        raise TranscriptionCancelledError("Cancelled while waiting for internet")
                    if pause_ev.is_set():
                        raise TranscriptionPausedError("Paused while waiting for internet")
                    return

            course = self.db.get_course(job["course_id"])
            course_name = course["name"] if course else "General"
            course_output_dir = (
                course["output_dir"]
                if course
                else os.path.join(cfg.output_base_dir, clean_filename(course_name))
            )
            os.makedirs(course_output_dir, exist_ok=True)

            updated = self.db.update_job(
                job_id,
                status="transcribing",
                stage="transcribing",
                progress=60.0,
                speed_bps=0.0,
                eta_seconds=None,
                error_message="",
            )
            self._emit_job(updated)
            self._emit_log(
                f"بدء التفريغ بالذكاء الاصطناعي ({len(chunks)} جزء): {job['title']}",
                "info",
                job_id=job_id,
            )

            def _ai_progress(pct: float, chunk_idx: int = 1, total_chunks: int = 1, message: str = "") -> None:
                overall = 60.0 + min(100.0, max(0.0, float(pct))) * 0.35
                u = self.db.update_job(
                    job_id,
                    status="transcribing",
                    stage="transcribing",
                    progress=round(overall, 1),
                    speed_bps=0.0,
                    eta_seconds=None,
                )
                self._emit_job(u)
                if message:
                    self._emit_log(f"[{job['title']}] {message}", "info", job_id=job_id)

            def _ai_status(msg: str, level: str = "info") -> None:
                if "انقطع" in msg or "offline" in msg.lower():
                    if self._online:
                        self._on_network_change(False)
                elif "عاد" in msg or "online" in msg.lower():
                    if not self._online:
                        self._on_network_change(True)
                self._emit_log(f"[{job['title']}] {msg}", level, job_id=job_id)

            transcriber = (
                self._transcriber_factory(cfg)
                if self._transcriber_factory is not None
                else GeminiTranscriber(cfg)
            )
            raw_transcript = transcriber.transcribe_chunks(
                chunks,
                language_hint="ar",
                cancel_event=cancel_ev,
                pause_event=pause_ev,
                progress_cb=_ai_progress,
                on_status=_ai_status,
            )

            if cancel_ev.is_set():
                raise TranscriptionCancelledError("Cancelled after transcription")
            if pause_ev.is_set():
                raise TranscriptionPausedError("Paused after transcription")

            # Stage 2b: Clean & Format
            updated = self.db.update_job(
                job_id,
                status="cleaning",
                stage="cleaning",
                progress=96.0,
                speed_bps=0.0,
                eta_seconds=None,
            )
            self._emit_job(updated)

            cleaned_body = clean_transcript_body(raw_transcript)
            if not cleaned_body:
                raise TranscriptionError("Cleaned transcript is empty after post-processing.")

            display_title = strip_media_extension(job["title"]) or job["title"]
            markdown_content = format_lecture_markdown(
                title=display_title,
                course_name=course_name,
                seq_num=job["seq_num"],
                source_type=job["source_type"],
                source_id=job["source_id"],
                url=job["url"],
                cleaned_body=cleaned_body,
            )

            expected_clean_path = _build_output_path(
                course_output_dir, int(job.get("seq_num") or 1), job["title"]
            )
            stored_output_path = job.get("output_path") or ""
            if not stored_output_path or any(
                stored_output_path.lower().endswith(f"{ext}.md") for ext in MEDIA_EXTENSIONS
            ):
                output_path = expected_clean_path
            else:
                output_path = stored_output_path
            _atomic_write_text(output_path, markdown_content)

            # Optionally save a clean lightweight MP3 copy to <course_output_dir>/audio/<base_stem>.mp3
            latest_cfg = self.config_manager.get()
            if bool(getattr(latest_cfg, "keep_audio", cfg.keep_audio)):
                base_stem = Path(output_path).stem
                audio_dest = Path(course_output_dir) / "audio" / f"{base_stem}.mp3"
                eff_br_fn = getattr(latest_cfg, "effective_audio_bitrate", None)
                eff_br = (
                    str(eff_br_fn())
                    if callable(eff_br_fn)
                    else str(getattr(latest_cfg, "audio_bitrate", "32k") or "32k")
                )
                try:
                    saved_audio = merge_audio_chunks_to_mp3(
                        chunks,
                        audio_dest,
                        bitrate=eff_br,
                        cancel_event=cancel_ev,
                    )
                    self._emit_log(
                        f"تم حفظ الملف الصوتي: audio/{Path(saved_audio).name}",
                        "info",
                        job_id=job_id,
                    )
                except Exception as audio_save_exc:
                    self._emit_log(
                        f"تعذر حفظ نسخة الصوت (MP3): {audio_save_exc}",
                        "warning",
                        job_id=job_id,
                    )

            # Clean up temp audio chunks before marking done so observers immediately see a clean temp dir
            try:
                cleanup_audio_files(chunks)
            except Exception:
                pass
            with self._lock:
                self._job_chunks.pop(job_id, None)
            should_cleanup_chunks = False

            # Under the per-course lock: rebuild artifacts with this job marked done,
            # then commit status="done" to SQLite BEFORE releasing the lock.
            # This guarantees (1) artifacts are ready the instant status="done" is visible,
            # and (2) concurrent completions in the same course always include each other.
            done_preview = dict(job)
            done_preview.update(
                status="done",
                stage="done",
                progress=100.0,
                speed_bps=0.0,
                eta_seconds=None,
                output_path=output_path,
                audio_path="",
                error_message="",
            )
            with self._get_course_lock(job["course_id"]):
                self._rebuild_artifacts_unlocked(job["course_id"], override_job=done_preview)
                updated = self.db.update_job(
                    job_id,
                    status="done",
                    stage="done",
                    progress=100.0,
                    speed_bps=0.0,
                    eta_seconds=None,
                    output_path=output_path,
                    audio_path="",
                    error_message="",
                )

            self._emit_job(updated)
            self._emit_log(
                f"اكتمل التفريغ بنجاح: {os.path.basename(output_path)}",
                "info",
                job_id=job_id,
            )

        except TranscriptionPausedError:
            # Keep converted chunks on disk so resume skips download+convert!
            should_cleanup_chunks = False
            if not self._stop_event.is_set():
                updated = self.db.pause_job(job_id)
                self._emit_job(updated)

        except TranscriptionCancelledError:
            should_cleanup_chunks = True
            updated = self.db.cancel_job(job_id)
            if updated is not None:
                updated = self.db.update_job(job_id, audio_path="")
            self._emit_job(updated)

        except (GeminiAuthError, GeminiQuotaError) as fatal_ai_exc:
            should_cleanup_chunks = False  # Keep chunks so user can retry after fixing key/quota
            updated = self.db.update_job(
                job_id,
                status="failed",
                stage="failed",
                speed_bps=0.0,
                eta_seconds=None,
                error_message=str(fatal_ai_exc),
            )
            self._emit_job(updated)
            self._emit_log(f"فشل التفريغ: {fatal_ai_exc}", "error", job_id=job_id)
            self._rebuild_artifacts_for_job(updated)

        except Exception as exc:
            if self._stop_event.is_set():
                return
            if cancel_ev.is_set():
                should_cleanup_chunks = True
                updated = self.db.cancel_job(job_id)
                self._emit_job(updated)
                return
            if pause_ev.is_set():
                updated = self.db.pause_job(job_id)
                self._emit_job(updated)
                return

            if not self._online or not is_online():
                reconnected = self._wait_until_online(job_id, cancel_ev, pause_ev)
                if self._stop_event.is_set():
                    return
                if cancel_ev.is_set():
                    should_cleanup_chunks = True
                    updated = self.db.cancel_job(job_id)
                    if updated is not None:
                        updated = self.db.update_job(job_id, audio_path="")
                    self._emit_job(updated)
                    return
                if pause_ev.is_set():
                    updated = self.db.pause_job(job_id)
                    self._emit_job(updated)
                    return
                if reconnected:
                    updated = self.db.update_job(
                        job_id,
                        status="queued",
                        stage="queued",
                        speed_bps=0.0,
                        eta_seconds=None,
                        error_message="",
                    )
                    self._emit_job(updated)
                    self._wake_event.set()
                    return

            job_cur = self.db.get_job(job_id)
            retries = int((job_cur or {}).get("retries", 0)) + 1
            if retries <= max(1, int(cfg.max_retries)):
                updated = self.db.update_job(
                    job_id,
                    status="queued",
                    stage="queued",
                    retries=retries,
                    speed_bps=0.0,
                    eta_seconds=None,
                    error_message=f"إعادة محاولة التفريغ ({retries}/{cfg.max_retries}): {exc}",
                )
                self._emit_job(updated)
                self._emit_log(
                    f"خطأ مؤقت أثناء التفريغ (محاولة {retries}/{cfg.max_retries}): {exc}",
                    "warning",
                    job_id=job_id,
                )
                self._wake_event.set()
            else:
                should_cleanup_chunks = True
                updated = self.db.update_job(
                    job_id,
                    status="failed",
                    stage="failed",
                    retries=retries,
                    speed_bps=0.0,
                    eta_seconds=None,
                    error_message=str(exc),
                )
                self._emit_job(updated)
                self._emit_log(f"فشل التفريغ نهائياً: {exc}", "error", job_id=job_id)
                self._rebuild_artifacts_for_job(updated)

        finally:
            if should_cleanup_chunks:
                try:
                    cleanup_audio_files(chunks)
                except Exception:
                    pass
                with self._lock:
                    self._job_chunks.pop(job_id, None)
            with self._lock:
                self._active_transcribe_ids.discard(job_id)
            self._wake_event.set()

    def _rebuild_artifacts_unlocked(
        self,
        course_id: str,
        override_job: dict[str, Any] | None = None,
    ) -> None:
        course = self.db.get_course(course_id)
        if not course:
            return
        try:
            course_jobs = self.db.list_jobs(course_id=course_id)
            if override_job is not None:
                oid = override_job.get("id")
                course_jobs = [
                    override_job if j.get("id") == oid else j
                    for j in course_jobs
                ]
            rebuild_course_artifacts(
                course_name=course["name"],
                output_dir=course["output_dir"],
                jobs_list=course_jobs,
            )
        except Exception as exc:
            self._emit_log(f"فشل تحديث ملفات المادة المجمعة: {exc}", "warning")

    def _rebuild_artifacts_for_job(
        self,
        job: dict[str, Any] | None,
        override_job: dict[str, Any] | None = None,
    ) -> None:
        if job is None:
            return
        course_id = job.get("course_id")
        if not course_id:
            return
        with self._get_course_lock(course_id):
            self._rebuild_artifacts_unlocked(course_id, override_job=override_job)
