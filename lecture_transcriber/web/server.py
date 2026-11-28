"""Ultra-lightweight zero-dependency local Web UI & REST/SSE server.

Uses Python's standard library ``http.server.ThreadingHTTPServer`` so startup is
instant (<0.2s) with minimal RAM footprint and zero external web frameworks.
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML_PATH = STATIC_DIR / "index.html"

_JOB_ACTION_RE = re.compile(r"^/api/jobs/([^/]+)/(pause|resume|retry|cancel|transcript|download_video)$")
_JOB_ITEM_RE = re.compile(r"^/api/jobs/([^/]+)$")
_MEDIA_EXT_STRIP_RE = re.compile(r"\.(mp4|mkv|webm|mov|avi|m4v|m4a|mp3|wav|ogg)$", re.IGNORECASE)
_VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")


def mask_api_key(key: str) -> str:
    """Return a masked representation of an API key, never exposing the secret."""
    if not key:
        return ""
    key = key.strip()
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def serialize_config_safe(config: Any) -> dict[str, Any]:
    """Convert AppConfig (dataclass or dict) to a JSON-safe dict with masked API key."""
    effective_key = ""
    if hasattr(config, "effective_api_key") and callable(config.effective_api_key):
        try:
            effective_key = str(config.effective_api_key() or "")
        except Exception:
            effective_key = ""

    if hasattr(config, "to_dict") and callable(config.to_dict):
        try:
            raw = dict(config.to_dict(mask_api_key=True))
            if not effective_key and raw.get("has_api_key"):
                masked = str(raw.get("gemini_api_key", ""))
            else:
                masked = mask_api_key(effective_key)
                raw["gemini_api_key"] = masked
                raw["has_api_key"] = bool(effective_key.strip())
            raw["api_key_masked"] = masked
            return raw
        except Exception:
            pass

    if is_dataclass(config):
        raw = asdict(config)
    elif isinstance(config, dict):
        raw = dict(config)
    else:
        raw = {k: getattr(config, k) for k in dir(config) if not k.startswith("_") and not callable(getattr(config, k))}

    api_key = effective_key or str(raw.get("gemini_api_key", "") or "")
    masked = mask_api_key(api_key)
    raw["gemini_api_key"] = masked
    raw["has_api_key"] = bool(api_key.strip())
    raw["api_key_masked"] = masked
    return raw


def _default_ffmpeg_status() -> dict[str, Any]:
    """Detect ffmpeg via downloader.audio_processor.check_ffmpeg or shutil.which."""
    try:
        from lecture_transcriber.downloader.audio_processor import check_ffmpeg  # type: ignore

        res = check_ffmpeg()
        if isinstance(res, dict):
            return res
        if isinstance(res, tuple):
            avail = bool(res[0])
            info = str(res[1]) if len(res) > 1 else ""
            return {"available": avail, "version": info if avail else "", "error": "" if avail else info}
        if isinstance(res, bool):
            return {"available": res, "path": shutil.which("ffmpeg") or ""}
    except Exception:
        pass

    ff_path = shutil.which("ffmpeg")
    if not ff_path:
        return {"available": False, "path": "", "error": "ffmpeg not found in PATH"}
    return {"available": True, "path": ff_path, "version": ff_path}


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _parse_urls_input(payload: dict[str, Any]) -> list[str]:
    """Extract clean URL list from either urls_text (newline string) or urls (list)."""
    urls: list[str] = []
    if isinstance(payload.get("urls"), list):
        for item in payload["urls"]:
            s = str(item).strip()
            if s and not s.startswith("#"):
                urls.append(s)
    urls_text = payload.get("urls_text", "")
    if isinstance(urls_text, str) and urls_text.strip():
        for line in urls_text.splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                urls.append(s)
    return urls


def _open_folder_in_os(folder_path: str) -> None:
    """Safely open a directory in the OS file explorer."""
    os.makedirs(folder_path, exist_ok=True)
    if sys.platform == "win32":
        os.startfile(folder_path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", folder_path])
    else:
        subprocess.Popen(["xdg-open", folder_path])


class TranscriberHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with clean SSE shutdown signaling and app context."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        *,
        engine: Any,
        config_manager: Any,
        db: Any,
        ffmpeg_status_fn: Optional[Callable[[], dict[str, Any]]] = None,
    ) -> None:
        self.engine = engine
        self.config_manager = config_manager
        self.db = db
        self.ffmpeg_status_fn = ffmpeg_status_fn or _default_ffmpeg_status
        self.shutdown_event = threading.Event()
        self._ffmpeg_cache: Optional[dict[str, Any]] = None
        self._video_dl_lock = threading.Lock()
        self._video_downloads: dict[str, dict[str, Any]] = {}
        self._video_dl_sem = threading.Semaphore(2)
        super().__init__(server_address, handler_cls)

    def _video_target_for_job(self, job: dict[str, Any], courses_by_id: dict[str, dict[str, Any]], cfg: Any) -> tuple[Path, str]:
        from lecture_transcriber.config import clean_filename

        course = courses_by_id.get(str(job.get("course_id") or ""))
        course_dir = (course or {}).get("output_dir") or (
            cfg.get("output_base_dir") if isinstance(cfg, dict) else getattr(cfg, "output_base_dir", "")
        )
        videos_dir = Path(course_dir or ".") / "Videos"
        seq = int(job.get("seq_num") or 0)
        raw_title = _MEDIA_EXT_STRIP_RE.sub("", str(job.get("title") or f"Lecture_{seq}").strip())
        safe_title = clean_filename(raw_title or f"Lecture_{seq}", max_length=100)
        stem = f"{seq:03d}_{safe_title}" if seq > 0 else safe_title
        return videos_dir, stem

    def get_video_downloads_snapshot(self, courses: list[dict[str, Any]], jobs: list[dict[str, Any]], cfg: Any) -> dict[str, dict[str, Any]]:
        courses_by_id = {str(c.get("id") or ""): c for c in courses}
        existing_by_dir: dict[Path, dict[str, Path]] = {}
        for c in courses:
            out_dir = c.get("output_dir")
            if not out_dir:
                continue
            vdir = Path(out_dir) / "Videos"
            if vdir.is_dir():
                try:
                    existing_by_dir[vdir] = {
                        p.stem.lower(): p
                        for p in vdir.iterdir()
                        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS and p.stat().st_size > 50_000
                    }
                except OSError:
                    existing_by_dir[vdir] = {}

        with self._video_dl_lock:
            snap: dict[str, dict[str, Any]] = {k: dict(v) for k, v in self._video_downloads.items()}

        for j in jobs:
            jid = str(j.get("id") or "")
            if not jid or jid in snap:
                continue
            vdir, stem = self._video_target_for_job(j, courses_by_id, cfg)
            found = existing_by_dir.get(vdir, {}).get(stem.lower())
            if found is not None:
                try:
                    size_bytes = found.stat().st_size
                except OSError:
                    size_bytes = 0
                snap[jid] = {
                    "job_id": jid,
                    "status": "done",
                    "progress": 100.0,
                    "path": str(found),
                    "filename": found.name,
                    "size_bytes": size_bytes,
                }
        return snap

    def emit_video_download_event(self, payload: dict[str, Any]) -> None:
        emit_fn = getattr(self.engine, "_emit", None)
        if callable(emit_fn):
            try:
                emit_fn("video_download", dict(payload))
            except Exception:
                pass

    def get_ffmpeg_status(self) -> dict[str, Any]:
        if self._ffmpeg_cache is None:
            try:
                self._ffmpeg_cache = self.ffmpeg_status_fn()
            except Exception as exc:
                self._ffmpeg_cache = {"available": False, "error": str(exc)}
        return self._ffmpeg_cache

    def shutdown(self) -> None:
        self.shutdown_event.set()
        super().shutdown()

    def server_close(self) -> None:
        self.shutdown_event.set()
        super().server_close()


class TranscriberRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the single-file Web UI, REST API, and SSE stream."""

    protocol_version = "HTTP/1.1"

    @property
    def app_server(self) -> TranscriberHTTPServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Keep console output clean unless debugging
        if os.environ.get("TRANSCRIBER_HTTP_DEBUG"):
            super().log_message(format, *args)

    # ---------- Response helpers ----------

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(_to_plain(payload), ensure_ascii=False).encode("utf-8")
        self._send_bytes(int(status), body, "application/json; charset=utf-8")

    def _send_error_json(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json({"ok": False, "error": message}, status=status)

    def _read_json_body(self) -> dict[str, Any]:
        raw_len = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_len)
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        if length > 5 * 1024 * 1024:
            raise ValueError("Request body too large")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _get_config(self) -> Any:
        cm = self.app_server.config_manager
        if hasattr(cm, "get"):
            return cm.get()
        if hasattr(cm, "config"):
            return cm.config
        if hasattr(cm, "load"):
            return cm.load()
        return cm

    def _get_online_status(self) -> bool:
        engine = self.app_server.engine
        for attr in ("is_online", "online"):
            val = getattr(engine, attr, None)
            if callable(val):
                try:
                    return bool(val())
                except Exception:
                    pass
            elif isinstance(val, bool):
                return val
        net_mon = getattr(engine, "network_monitor", None)
        if net_mon is not None:
            val = getattr(net_mon, "is_online", None)
            if callable(val):
                return bool(val())
            if isinstance(val, bool):
                return val
        return True

    def _build_state_snapshot(self) -> dict[str, Any]:
        cfg = self._get_config()
        db = self.app_server.db
        courses = _to_plain(db.list_courses()) if hasattr(db, "list_courses") else []
        jobs = _to_plain(db.list_jobs()) if hasattr(db, "list_jobs") else []
        stats = _to_plain(db.get_stats()) if hasattr(db, "get_stats") else {}
        return {
            "config": serialize_config_safe(cfg),
            "courses": courses,
            "jobs": jobs,
            "stats": stats,
            "online": self._get_online_status(),
            "ffmpeg": self.app_server.get_ffmpeg_status(),
            "video_downloads": self.app_server.get_video_downloads_snapshot(courses, jobs, cfg),
        }

    # ---------- HTTP Verb Dispatch ----------

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            if not INDEX_HTML_PATH.is_file():
                self._send_error_json("index.html not found", HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(
                int(HTTPStatus.OK),
                INDEX_HTML_PATH.read_bytes(),
                "text/html; charset=utf-8",
            )
            return

        if path == "/api/state":
            try:
                self._send_json(self._build_state_snapshot())
            except Exception as exc:
                self._send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if path == "/api/events":
            self._handle_sse()
            return

        match = _JOB_ACTION_RE.match(path)
        if match and match.group(2) == "transcript":
            job_id = urllib.parse.unquote(match.group(1))
            self._handle_get_transcript(job_id)
            return

        self._send_error_json("Endpoint not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path

        try:
            body = self._read_json_body()
        except ValueError as exc:
            self._send_error_json(str(exc), HTTPStatus.BAD_REQUEST)
            return
        except json.JSONDecodeError as exc:
            self._send_error_json(f"Invalid JSON: {exc}", HTTPStatus.BAD_REQUEST)
            return

        if path == "/api/enqueue":
            self._handle_enqueue(body)
            return

        if path == "/api/pause_all":
            try:
                res = self.app_server.engine.pause_all()
                self._send_json({"ok": True, "result": res})
            except Exception as exc:
                self._send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if path == "/api/resume_all":
            try:
                res = self.app_server.engine.resume_all()
                self._send_json({"ok": True, "result": res})
            except Exception as exc:
                self._send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if path == "/api/settings":
            self._handle_update_settings(body)
            return

        if path == "/api/open_folder":
            self._handle_open_folder(body)
            return

        match = _JOB_ACTION_RE.match(path)
        if match:
            job_id = urllib.parse.unquote(match.group(1))
            action = match.group(2)
            if action in ("pause", "resume", "retry", "cancel"):
                self._handle_job_action(job_id, action)
                return
            if action == "download_video":
                self._handle_download_video(job_id, body)
                return

        self._send_error_json("Endpoint not found", HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        match = _JOB_ITEM_RE.match(parsed.path)
        if not match:
            self._send_error_json("Endpoint not found", HTTPStatus.NOT_FOUND)
            return

        job_id = urllib.parse.unquote(match.group(1))
        try:
            db = self.app_server.db
            engine = self.app_server.engine
            job = db.get_job(job_id) if hasattr(db, "get_job") else None
            if job is None:
                self._send_error_json("Job not found", HTTPStatus.NOT_FOUND)
                return

            if hasattr(engine, "delete_job"):
                engine.delete_job(job_id)
            else:
                if hasattr(engine, "cancel_job"):
                    try:
                        engine.cancel_job(job_id)
                    except Exception:
                        pass
                db.delete_job(job_id)

            self._send_json({"ok": True, "deleted": job_id})
        except Exception as exc:
            self._send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    # ---------- Endpoint implementations ----------

    def _handle_enqueue(self, body: dict[str, Any]) -> None:
        course_name = str(body.get("course_name", "")).strip()
        urls = _parse_urls_input(body)
        if not course_name:
            self._send_error_json("course_name is required", HTTPStatus.BAD_REQUEST)
            return
        if not urls:
            self._send_error_json("At least one URL is required", HTTPStatus.BAD_REQUEST)
            return

        try:
            include_playlist = bool(body.get("include_playlist", False))
            try:
                result = self.app_server.engine.enqueue_urls(
                    course_name,
                    urls,
                    include_playlist=include_playlist,
                )
            except TypeError:
                result = self.app_server.engine.enqueue_urls(course_name, urls)
            if isinstance(result, dict):
                payload = {"ok": True, **result}
            else:
                payload = {"ok": True, "result": result}
            self._send_json(payload)
        except Exception as exc:
            self._send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_job_action(self, job_id: str, action: str) -> None:
        db = self.app_server.db
        engine = self.app_server.engine
        if hasattr(db, "get_job") and db.get_job(job_id) is None:
            self._send_error_json("Job not found", HTTPStatus.NOT_FOUND)
            return

        method_name = f"{action}_job"
        method = getattr(engine, method_name, None)
        if method is None:
            self._send_error_json(f"Engine does not support {method_name}", HTTPStatus.NOT_IMPLEMENTED)
            return

        try:
            result = method(job_id)
            updated = db.get_job(job_id) if hasattr(db, "get_job") else None
            self._send_json({"ok": True, "action": action, "job_id": job_id, "result": result, "job": updated})
        except Exception as exc:
            self._send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_get_transcript(self, job_id: str) -> None:
        db = self.app_server.db
        job = db.get_job(job_id) if hasattr(db, "get_job") else None
        if not job:
            self._send_error_json("Job not found", HTTPStatus.NOT_FOUND)
            return
        job_dict = _to_plain(job)
        out_path = job_dict.get("output_path") or ""
        if not out_path or not os.path.isfile(out_path):
            self._send_error_json("Transcript file not available yet", HTTPStatus.NOT_FOUND)
            return
        try:
            content = Path(out_path).read_text(encoding="utf-8", errors="replace")
            self._send_json(
                {
                    "ok": True,
                    "job_id": job_id,
                    "title": job_dict.get("title") or Path(out_path).stem,
                    "filename": os.path.basename(out_path),
                    "path": out_path,
                    "content": content,
                }
            )
        except Exception as exc:
            self._send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_update_settings(self, body: dict[str, Any]) -> None:
        cm = self.app_server.config_manager
        updates: dict[str, Any] = {}

        if "gemini_api_key" in body and isinstance(body["gemini_api_key"], str):
            if body["gemini_api_key"].strip():
                updates["gemini_api_key"] = body["gemini_api_key"].strip()

        if "output_base_dir" in body and isinstance(body["output_base_dir"], str):
            val = body["output_base_dir"].strip()
            if val:
                updates["output_base_dir"] = val

        for int_key, lo, hi in (
            ("download_workers", 1, 12),
            ("transcribe_workers", 1, 10),
            ("max_chunk_minutes", 5, 120),
            ("port", 1024, 65535),
        ):
            if int_key in body and body[int_key] is not None:
                try:
                    v = int(body[int_key])
                except (TypeError, ValueError):
                    self._send_error_json(f"Invalid integer for {int_key}", HTTPStatus.BAD_REQUEST)
                    return
                if not (lo <= v <= hi):
                    self._send_error_json(f"{int_key} must be between {lo} and {hi}", HTTPStatus.BAD_REQUEST)
                    return
                updates[int_key] = v

        if "audio_bitrate" in body and isinstance(body["audio_bitrate"], str):
            br = body["audio_bitrate"].strip()
            if re.match(r"^\d{2,3}k$", br):
                updates["audio_bitrate"] = br
            else:
                self._send_error_json("Invalid audio_bitrate (expected e.g. '64k')", HTTPStatus.BAD_REQUEST)
                return

        if "download_mode" in body and body["download_mode"] is not None:
            if not isinstance(body["download_mode"], str):
                self._send_error_json("Invalid download_mode", HTTPStatus.BAD_REQUEST)
                return
            mode_val = body["download_mode"].strip().lower()
            from lecture_transcriber.config import VALID_DOWNLOAD_MODES

            if mode_val not in VALID_DOWNLOAD_MODES:
                self._send_error_json(
                    f"Invalid download_mode '{body['download_mode']}'. Expected one of: {', '.join(VALID_DOWNLOAD_MODES)}",
                    HTTPStatus.BAD_REQUEST,
                )
                return
            updates["download_mode"] = mode_val

        if "keep_audio" in body and body["keep_audio"] is not None:
            raw_keep = body["keep_audio"]
            if isinstance(raw_keep, bool):
                updates["keep_audio"] = raw_keep
            elif isinstance(raw_keep, str) and raw_keep.strip().lower() in ("true", "false", "1", "0"):
                updates["keep_audio"] = raw_keep.strip().lower() in ("true", "1")
            elif isinstance(raw_keep, int) and raw_keep in (0, 1):
                updates["keep_audio"] = bool(raw_keep)
            else:
                self._send_error_json("Invalid boolean for keep_audio", HTTPStatus.BAD_REQUEST)
                return

        if "proxy" in body and isinstance(body["proxy"], str):
            updates["proxy"] = body["proxy"].strip()

        if "models" in body and body["models"] is not None:
            raw_models = body["models"]
            parsed_models: list[str] = []
            if isinstance(raw_models, str):
                parsed_models = [
                    m.strip()
                    for m in re.split(r"[,\r\n]+", raw_models)
                    if m.strip()
                ]
            elif isinstance(raw_models, (list, tuple)):
                for item in raw_models:
                    if not isinstance(item, str):
                        self._send_error_json("Invalid models entry: expected string", HTTPStatus.BAD_REQUEST)
                        return
                    cleaned_m = item.strip()
                    if cleaned_m:
                        parsed_models.append(cleaned_m)
            else:
                self._send_error_json("Invalid models format", HTTPStatus.BAD_REQUEST)
                return

            deduped_models = list(dict.fromkeys(parsed_models))
            if not deduped_models:
                self._send_error_json("models must contain at least one model name", HTTPStatus.BAD_REQUEST)
                return
            updates["models"] = deduped_models

        try:
            if hasattr(cm, "update"):
                cfg = cm.update(**updates)
            else:
                cfg = self._get_config()
                for k, v in updates.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
                    elif isinstance(cfg, dict):
                        cfg[k] = v
                if hasattr(cm, "save"):
                    cm.save(cfg)

            engine = self.app_server.engine
            if hasattr(engine, "reload_config"):
                engine.reload_config()
            elif hasattr(engine, "update_config"):
                engine.update_config(cfg)

            self._send_json({"ok": True, "config": serialize_config_safe(self._get_config())})
        except Exception as exc:
            self._send_error_json(str(exc), HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_download_video(self, job_id: str, body: dict[str, Any]) -> None:
        db = self.app_server.db
        job_raw = db.get_job(job_id) if hasattr(db, "get_job") else None
        if not job_raw:
            self._send_error_json("Job not found", HTTPStatus.NOT_FOUND)
            return

        job = _to_plain(job_raw)
        url = str(job.get("url") or "").strip()
        source_type = str(job.get("source_type") or "").strip().lower()
        source_id = str(job.get("source_id") or job_id).strip()
        if not url:
            self._send_error_json("Job has no source URL", HTTPStatus.BAD_REQUEST)
            return

        cfg = self._get_config()
        courses = _to_plain(db.list_courses()) if hasattr(db, "list_courses") else []
        courses_by_id = {str(c.get("id") or ""): c for c in courses}
        videos_dir, stem = self.app_server._video_target_for_job(job, courses_by_id, cfg)

        # If already downloaded, optionally open Videos folder and return immediately
        if videos_dir.is_dir():
            for ext in _VIDEO_EXTS:
                candidate = videos_dir / f"{stem}{ext}"
                if candidate.is_file() and candidate.stat().st_size > 50_000:
                    if body.get("open_if_exists", True):
                        try:
                            _open_folder_in_os(str(videos_dir))
                        except Exception:
                            pass
                    entry = {
                        "job_id": job_id,
                        "status": "done",
                        "progress": 100.0,
                        "path": str(candidate),
                        "filename": candidate.name,
                        "size_bytes": candidate.stat().st_size,
                    }
                    with self.app_server._video_dl_lock:
                        self.app_server._video_downloads[job_id] = entry
                    self._send_json({"ok": True, "already_exists": True, "video": entry})
                    return

        with self.app_server._video_dl_lock:
            existing_state = self.app_server._video_downloads.get(job_id)
            if existing_state and existing_state.get("status") in ("queued", "downloading"):
                self._send_json({"ok": True, "already_running": True, "video": dict(existing_state)})
                return
            init_state: dict[str, Any] = {
                "job_id": job_id,
                "status": "queued",
                "progress": 0.0,
                "speed_bps": 0.0,
                "path": "",
                "filename": f"{stem}.mp4",
                "error": "",
            }
            self.app_server._video_downloads[job_id] = init_state

        self.app_server.emit_video_download_event(init_state)

        def _worker() -> None:
            server_ref = self.app_server
            last_emit = [0.0]

            def _update(status: str, progress: float = 0.0, speed_bps: float = 0.0, path: str = "", filename: str = "", error: str = "", size_bytes: int = 0, force_emit: bool = False) -> None:
                payload = {
                    "job_id": job_id,
                    "status": status,
                    "progress": round(float(progress), 1),
                    "speed_bps": float(speed_bps or 0.0),
                    "path": path,
                    "filename": filename or f"{stem}.mp4",
                    "error": error,
                    "size_bytes": int(size_bytes or 0),
                }
                with server_ref._video_dl_lock:
                    server_ref._video_downloads[job_id] = payload
                now = time.monotonic()
                if force_emit or (now - last_emit[0] >= 0.45):
                    last_emit[0] = now
                    server_ref.emit_video_download_event(payload)

            with server_ref._video_dl_sem:
                if server_ref.shutdown_event.is_set():
                    return
                _update("downloading", 0.0, force_emit=True)
                try:
                    videos_dir.mkdir(parents=True, exist_ok=True)
                    tmp_dir = videos_dir / "_tmp"
                    tmp_dir.mkdir(parents=True, exist_ok=True)

                    if source_type == "gdrive":
                        from lecture_transcriber.config import AppConfig
                        from lecture_transcriber.downloader.gdrive_dl import download_gdrive_file

                        app_cfg = cfg if isinstance(cfg, AppConfig) else AppConfig()

                        def _prog_cb(pct: float, spd: float, eta: Any, dl: int, tot: int) -> None:
                            _update("downloading", pct, spd)

                        dl_path = Path(
                            download_gdrive_file(
                                url=url,
                                source_id=source_id,
                                temp_dir=tmp_dir,
                                config=app_cfg,
                                progress_cb=_prog_cb,
                                skip_transcoded=True,
                            )
                        )
                        ext = dl_path.suffix.lower() if dl_path.suffix.lower() in _VIDEO_EXTS else ".mp4"
                        final_path = videos_dir / f"{stem}{ext}"
                        shutil.move(str(dl_path), str(final_path))
                    else:
                        import yt_dlp

                        out_tmpl = str(tmp_dir / f"{stem}.%(ext)s")
                        recorded = {"path": ""}

                        def _yt_hook(d: dict[str, Any]) -> None:
                            st = d.get("status")
                            if st == "downloading":
                                tot = float(d.get("total_bytes") or d.get("total_bytes_estimate") or 0)
                                dl = float(d.get("downloaded_bytes") or 0)
                                spd = float(d.get("speed") or 0)
                                pct = min(99.0, (dl / tot) * 100.0) if tot > 0 else 0.0
                                _update("downloading", pct, spd)
                            elif st == "finished" and d.get("filename"):
                                recorded["path"] = str(d["filename"])

                        ydl_opts: dict[str, Any] = {
                            "outtmpl": out_tmpl,
                            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                            "merge_output_format": "mp4",
                            "noplaylist": True,
                            "quiet": True,
                            "no_warnings": True,
                            "continuedl": True,
                            "progress_hooks": [_yt_hook],
                        }
                        proxy_val = cfg.get("proxy") if isinstance(cfg, dict) else getattr(cfg, "proxy", "")
                        if proxy_val:
                            ydl_opts["proxy"] = proxy_val

                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(url, download=True)
                            if not recorded["path"] and isinstance(info, dict):
                                req = info.get("requested_downloads") or []
                                if req and isinstance(req[0], dict) and req[0].get("filepath"):
                                    recorded["path"] = str(req[0]["filepath"])
                                else:
                                    recorded["path"] = ydl.prepare_filename(info)

                        dl_path = Path(recorded["path"])
                        if not dl_path.is_file():
                            matches = [p for p in tmp_dir.glob(f"{stem}.*") if p.is_file() and not p.name.endswith(".part")]
                            if not matches:
                                raise FileNotFoundError("Downloaded video file not found")
                            dl_path = max(matches, key=lambda p: p.stat().st_size)
                        ext = dl_path.suffix.lower() if dl_path.suffix.lower() in _VIDEO_EXTS else ".mp4"
                        final_path = videos_dir / f"{stem}{ext}"
                        shutil.move(str(dl_path), str(final_path))

                    try:
                        if tmp_dir.is_dir() and not any(tmp_dir.iterdir()):
                            tmp_dir.rmdir()
                    except OSError:
                        pass

                    size_b = final_path.stat().st_size if final_path.is_file() else 0
                    _update(
                        "done",
                        100.0,
                        0.0,
                        path=str(final_path),
                        filename=final_path.name,
                        size_bytes=size_b,
                        force_emit=True,
                    )
                except Exception as exc:
                    _update("error", 0.0, 0.0, error=str(exc), force_emit=True)

        t = threading.Thread(target=_worker, name=f"video-dl-{job_id[:8]}", daemon=True)
        t.start()
        self._send_json({"ok": True, "started": True, "video": init_state})

    def _handle_open_folder(self, body: dict[str, Any]) -> None:
        cfg = self._get_config()
        base_dir = cfg.get("output_base_dir") if isinstance(cfg, dict) else getattr(cfg, "output_base_dir", "")
        target_dir = base_dir

        course_id = body.get("course_id")
        if course_id:
            db = self.app_server.db
            courses = _to_plain(db.list_courses()) if hasattr(db, "list_courses") else []
            for c in courses:
                if c.get("id") == course_id and c.get("output_dir"):
                    target_dir = c["output_dir"]
                    break

        subfolder = str(body.get("subfolder") or "").strip()
        if subfolder and target_dir and subfolder in ("Videos",):
            sub_path = os.path.join(target_dir, subfolder)
            if os.path.isdir(sub_path):
                target_dir = sub_path

        if not target_dir:
            self._send_error_json("Output folder is not configured", HTTPStatus.BAD_REQUEST)
            return

        resolved = os.path.abspath(target_dir)
        # Ensure we only open the configured output_base_dir or a subdirectory / registered course dir
        try:
            _open_folder_in_os(resolved)
            self._send_json({"ok": True, "path": resolved})
        except Exception as exc:
            self._send_error_json(f"Failed to open folder: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_sse(self) -> None:
        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        ev_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=256)
        engine = self.app_server.engine

        def _push_cb(event: dict[str, Any]) -> None:
            try:
                ev_queue.put_nowait(event)
            except queue.Full:
                try:
                    ev_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    ev_queue.put_nowait(event)
                except queue.Full:
                    pass

        sub_token: Any = None
        use_queue: queue.Queue[Any] = ev_queue
        subscribed_callback = False

        if hasattr(engine, "subscribe"):
            try:
                # Support both subscribe() -> Queue and subscribe(callback) -> token/id
                sub_ret = engine.subscribe()
                if isinstance(sub_ret, queue.Queue):
                    use_queue = sub_ret
                    sub_token = sub_ret
                else:
                    sub_token = sub_ret
            except TypeError:
                sub_token = engine.subscribe(_push_cb)
                subscribed_callback = True

        try:
            # Send initial connected comment & snapshot immediately
            init_payload = {
                "type": "snapshot",
                "data": {
                    "online": self._get_online_status(),
                    "courses": _to_plain(self.app_server.db.list_courses()) if hasattr(self.app_server.db, "list_courses") else [],
                    "jobs": _to_plain(self.app_server.db.list_jobs()) if hasattr(self.app_server.db, "list_jobs") else [],
                },
            }
            self.wfile.write(f"data: {json.dumps(init_payload, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

            last_ping = time.monotonic()
            while not self.app_server.shutdown_event.is_set():
                try:
                    item = use_queue.get(timeout=0.5)
                    msg = json.dumps(_to_plain(item), ensure_ascii=False)
                    self.wfile.write(f"data: {msg}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    last_ping = time.monotonic()
                except queue.Empty:
                    if time.monotonic() - last_ping >= 15.0:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                        last_ping = time.monotonic()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            if hasattr(engine, "unsubscribe"):
                try:
                    if subscribed_callback:
                        engine.unsubscribe(sub_token if sub_token is not None else _push_cb)
                    elif sub_token is not None:
                        engine.unsubscribe(sub_token)
                except Exception:
                    pass


def create_server(
    engine: Any,
    config_manager: Any,
    db: Any,
    host: str = "127.0.0.1",
    port: int = 8765,
    ffmpeg_status_fn: Optional[Callable[[], dict[str, Any]]] = None,
) -> TranscriberHTTPServer:
    """Create a configured ThreadingHTTPServer (pass port=0 for ephemeral port)."""
    return TranscriberHTTPServer(
        (host, port),
        TranscriberRequestHandler,
        engine=engine,
        config_manager=config_manager,
        db=db,
        ffmpeg_status_fn=ffmpeg_status_fn,
    )


def run_server(
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    open_browser: bool = True,
) -> None:
    """Initialize ConfigManager, Database, TranscriptionEngine and serve the Web UI."""
    from lecture_transcriber.config import ConfigManager, ensure_utf8_stdio
    from lecture_transcriber.database import Database
    from lecture_transcriber.engine import TranscriptionEngine

    ensure_utf8_stdio()
    config_manager = ConfigManager()
    cfg = config_manager.get() if hasattr(config_manager, "get") else config_manager.load()
    actual_port = port if port is not None else getattr(cfg, "port", 8765)

    db = Database()
    engine = TranscriptionEngine(config_manager, db)
    engine.start()

    server = create_server(engine, config_manager, db, host=host, port=actual_port)
    bound_host, bound_port = server.server_address[:2]
    url = f"http://{bound_host}:{bound_port}"
    print(f"Lecture Transcriber Web UI running at {url}")
    print("Press Ctrl+C to stop safely (active downloads can resume next launch).")

    if open_browser:
        threading.Timer(0.35, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever(poll_interval=0.4)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        server.shutdown_event.set()
        server.server_close()
        if hasattr(engine, "stop"):
            engine.stop()
        if hasattr(db, "close"):
            db.close()
