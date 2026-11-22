"""Ultra-resilient YouTube audio downloader powered by yt-dlp."""

from __future__ import annotations

import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Callable

import yt_dlp

from ..config import AppConfig, clean_filename
from ..network import is_online, wait_for_internet
from .url_parser import (
    DownloadCancelledError,
    DownloadError,
    DownloadPausedError,
)

ECO_AUDIO_FORMAT = "bestaudio[abr<=64][ext=m4a]/bestaudio[abr<=64]/bestaudio[abr<=96]/bestaudio/ba"
BALANCED_AUDIO_FORMAT = "bestaudio[ext=m4a]/bestaudio[abr<=128]/bestaudio/ba"
STRICT_AUDIO_FORMAT = "bestaudio[abr<=96]/bestaudio/ba"

PRIMARY_AUDIO_FORMAT = ECO_AUDIO_FORMAT
FALLBACK_AUDIO_FORMAT = "18/ba/b"

_FALLBACK_TRIGGER_MARKERS = (
    "403",
    "forbidden",
    "sabr",
    "requested format",
    "format is not available",
    "unable to extract",
    "nsig",
    "signature extraction",
    "fragment",
    "did not get any data blocks",
    "empty",
    "http error 429",
    "sign in to confirm",
)

_PERMANENT_ERROR_MARKERS = (
    "private video",
    "video unavailable",
    "this video has been removed",
    "account associated with this video has been terminated",
    "copyright claim",
    "members-only content",
    "join this channel to get access",
)

_PARTIAL_EXTENSIONS = (".part", ".ytdl", ".temp")


class _QuietYtdlpLogger:
    """Captures yt-dlp error/warning messages without spamming stdout/stderr."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        self.warnings.append(str(msg))

    def error(self, msg: str) -> None:
        self.errors.append(str(msg))


def _check_control_events(
    cancel_event: threading.Event | None,
    pause_event: threading.Event | None,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise DownloadCancelledError("تم إلغاء التحميل / Download cancelled")
    if pause_event is not None and pause_event.is_set():
        raise DownloadPausedError("تم إيقاف التحميل مؤقتاً / Download paused")


def _wait_interruptible(
    seconds: float,
    cancel_event: threading.Event | None,
    pause_event: threading.Event | None,
) -> None:
    """Sleep up to ``seconds``, waking immediately if paused or cancelled."""
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        _check_control_events(cancel_event, pause_event)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        step = min(0.25, remaining)
        if cancel_event is not None:
            if cancel_event.wait(timeout=step):
                raise DownloadCancelledError("تم إلغاء التحميل / Download cancelled")
        elif pause_event is not None:
            if pause_event.wait(timeout=step):
                raise DownloadPausedError("تم إيقاف التحميل مؤقتاً / Download paused")
        else:
            time.sleep(step)


def _wait_for_network_or_control(
    cancel_event: threading.Event | None,
    pause_event: threading.Event | None,
    on_status: Callable[..., Any] | None = None,
) -> None:
    """Wait until internet connectivity returns, aborting immediately on pause/cancel."""
    _check_control_events(cancel_event, pause_event)
    if is_online():
        return

    bridge_stop = threading.Event()

    def _watcher() -> None:
        while not bridge_stop.is_set():
            if (cancel_event and cancel_event.is_set()) or (
                pause_event and pause_event.is_set()
            ):
                bridge_stop.set()
                return
            bridge_stop.wait(timeout=0.2)

    watcher_thread = threading.Thread(target=_watcher, daemon=True)
    watcher_thread.start()
    try:
        reconnected = wait_for_internet(
            stop_event=bridge_stop,
            check_interval=3.0,
            on_status=on_status,
        )
    finally:
        bridge_stop.set()
        watcher_thread.join(timeout=1.0)

    _check_control_events(cancel_event, pause_event)
    if not reconnected:
        raise DownloadError("انقطع الاتصال بالإنترنت / Internet connection unavailable")


def _cleanup_partial_files(temp_dir: Path, safe_id: str) -> None:
    """Delete .part/.ytdl artifacts for a cancelled YouTube download."""
    for entry in temp_dir.glob(f"{safe_id}.*"):
        name_lower = entry.name.lower()
        if any(name_lower.endswith(ext) for ext in _PARTIAL_EXTENSIONS) or ".part-frag" in name_lower:
            try:
                entry.unlink()
            except OSError:
                pass


def _find_downloaded_file(
    temp_dir: Path,
    safe_id: str,
    recorded_path: str | None = None,
) -> Path | None:
    """Locate the completed downloaded media file in ``temp_dir``."""
    if recorded_path:
        p = Path(recorded_path)
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            if not any(p.name.lower().endswith(ext) for ext in _PARTIAL_EXTENSIONS):
                return p

    candidates: list[Path] = []
    prefix = f"{safe_id}."
    for entry in temp_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        lower = name.lower()
        if not name.startswith(prefix):
            continue
        if any(lower.endswith(ext) for ext in _PARTIAL_EXTENSIONS) or ".part-frag" in lower:
            continue
        if "_chunk_" in lower or lower.endswith("_converted.mp3"):
            continue
        if entry.stat().st_size > 0:
            candidates.append(entry)

    if not candidates:
        return None
    # Prefer most recently modified non-empty candidate
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _should_use_fallback(err_text: str) -> bool:
    lower = err_text.lower()
    return any(marker in lower for marker in _FALLBACK_TRIGGER_MARKERS)


def _is_permanent_youtube_error(err_text: str) -> bool:
    lower = err_text.lower()
    return any(marker in lower for marker in _PERMANENT_ERROR_MARKERS)


def _resolve_download_mode(cfg: AppConfig) -> str:
    mode = str(getattr(cfg, "download_mode", "eco") or "eco").strip().lower()
    if mode in ("eco", "balanced", "strict_audio"):
        return mode
    return "eco"


def _build_ydl_profiles(
    out_tmpl: str,
    cfg: AppConfig,
    progress_hook: Callable[[dict[str, Any]], None],
    logger: _QuietYtdlpLogger,
) -> list[dict[str, Any]]:
    """Build ordered yt-dlp option profiles based on ``cfg.download_mode``.

    - ``eco`` / ``balanced`` try audio-only streams across multiple YouTube clients
      before ever considering the last-resort muxed ``18/ba/b`` fallback.
    - ``strict_audio`` tries audio-only across all clients and NEVER falls back to video.
    """
    common: dict[str, Any] = {
        "outtmpl": out_tmpl,
        "continuedl": True,
        "concurrent_fragment_downloads": 4,
        "retries": 15,
        "fragment_retries": 15,
        "file_access_retries": 5,
        "socket_timeout": cfg.socket_timeout,
        "http_chunk_size": 1048576,  # 1MB chunks to bypass YouTube throttling
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [progress_hook],
        "logger": logger,
    }
    if cfg.proxy:
        common["proxy"] = cfg.proxy

    mode = _resolve_download_mode(cfg)
    if mode == "balanced":
        audio_fmt = BALANCED_AUDIO_FORMAT
    elif mode == "strict_audio":
        audio_fmt = STRICT_AUDIO_FORMAT
    else:
        audio_fmt = ECO_AUDIO_FORMAT

    profile_android_ios = dict(common)
    profile_android_ios["format"] = audio_fmt
    profile_android_ios["extractor_args"] = {"youtube": {"player_client": ["android", "ios"]}}

    profile_tv_web = dict(common)
    profile_tv_web["format"] = audio_fmt
    profile_tv_web["extractor_args"] = {
        "youtube": {"player_client": ["tv_embedded", "web_safari", "web"]}
    }

    profile_default = dict(common)
    profile_default["format"] = audio_fmt

    profiles: list[dict[str, Any]] = [
        profile_default,
        profile_android_ios,
        profile_tv_web,
    ]

    if mode != "strict_audio":
        last_resort_video = dict(common)
        last_resort_video["format"] = FALLBACK_AUDIO_FORMAT
        last_resort_video["extractor_args"] = {"youtube": {"player_client": ["android", "web"]}}
        profiles.append(last_resort_video)

    return profiles


def download_youtube_audio(
    url: str,
    source_id: str,
    temp_dir: str | Path,
    config: AppConfig | None = None,
    progress_cb: Callable[[float, float, float | None, int, int], Any] | None = None,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
    on_status: Callable[..., Any] | None = None,
) -> str:
    """Download YouTube audio resiliently with resume, SABR/403 fallback, and outage recovery.

    Args:
        url: YouTube video URL.
        source_id: Video identifier used for naming temp/partial files.
        temp_dir: Directory to store partial and completed raw audio files.
        config: ``AppConfig`` instance (uses defaults if omitted).
        progress_cb: Optional callback ``(progress_pct, speed_bps, eta_sec, downloaded_bytes, total_bytes)``.
        cancel_event: Setting aborts download and cleans ``.part`` files with ``DownloadCancelledError``.
        pause_event: Setting pauses download and preserves ``.part`` files with ``DownloadPausedError``.
        on_status: Optional status callback passed to ``wait_for_internet``.

    Returns:
        Absolute path string to the downloaded raw media file.
    """
    cfg = config or AppConfig()
    out_dir = Path(temp_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = clean_filename(source_id or "youtube_audio", max_length=80)
    out_tmpl = str(out_dir / f"{safe_id}.%(ext)s")

    recorded_file: dict[str, str | None] = {"path": None}

    def _progress_hook(d: dict[str, Any]) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise DownloadCancelledError("تم إلغاء التحميل / Download cancelled")
        if pause_event is not None and pause_event.is_set():
            raise DownloadPausedError("تم إيقاف التحميل مؤقتاً / Download paused")

        status = d.get("status")
        if status == "finished":
            if d.get("filename"):
                recorded_file["path"] = str(d["filename"])
            total = int(d.get("total_bytes") or d.get("downloaded_bytes") or 0)
            if progress_cb:
                try:
                    progress_cb(100.0, float(d.get("speed") or 0.0), 0.0, total, total)
                except Exception:
                    pass
            return

        if status == "downloading":
            downloaded = int(d.get("downloaded_bytes") or 0)
            total = int(d.get("total_bytes") or d.get("total_bytes_estimate") or 0)
            speed = float(d.get("speed") or 0.0)
            eta = d.get("eta")
            eta_val = float(eta) if isinstance(eta, (int, float)) else None
            pct = min(100.0, (downloaded / total) * 100.0) if total > 0 else 0.0
            if progress_cb:
                try:
                    progress_cb(pct, speed, eta_val, downloaded, total)
                except Exception:
                    pass

    mode = _resolve_download_mode(cfg)
    dummy_logger = _QuietYtdlpLogger()
    profile_count = len(_build_ydl_profiles(out_tmpl, cfg, _progress_hook, dummy_logger))
    raw_retries = int(cfg.max_retries)
    max_attempts = max(profile_count, raw_retries + 1) if raw_retries >= 1 else 1
    profile_index = 0
    attempt = 0
    last_error: Exception | None = None

    try:
        while attempt < max_attempts:
            _check_control_events(cancel_event, pause_event)
            _wait_for_network_or_control(cancel_event, pause_event, on_status=on_status)

            logger = _QuietYtdlpLogger()
            profiles = _build_ydl_profiles(out_tmpl, cfg, _progress_hook, logger)
            last_profile_idx = len(profiles) - 1
            opts = profiles[min(profile_index, last_profile_idx)]

            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if not recorded_file["path"] and isinstance(info, dict):
                        req_dls = info.get("requested_downloads") or []
                        if req_dls and isinstance(req_dls[0], dict) and req_dls[0].get("filepath"):
                            recorded_file["path"] = str(req_dls[0]["filepath"])
                        else:
                            try:
                                recorded_file["path"] = ydl.prepare_filename(info)
                            except Exception:
                                pass

                found = _find_downloaded_file(out_dir, safe_id, recorded_file["path"])
                if found is not None:
                    return str(found)

                raise DownloadError(
                    f"لم يتم العثور على الملف المحمّل بعد انتهاء التحميل / "
                    f"Downloaded file not found for {source_id}"
                )

            except DownloadCancelledError:
                raise
            except DownloadPausedError:
                # Explicitly preserve .part files so resume continues from exact byte offset!
                raise
            except Exception as exc:
                # Re-check control events in case yt-dlp caught and re-wrapped our hook error
                if cancel_event is not None and cancel_event.is_set():
                    raise DownloadCancelledError("تم إلغاء التحميل / Download cancelled") from exc
                if pause_event is not None and pause_event.is_set():
                    raise DownloadPausedError("تم إيقاف التحميل مؤقتاً / Download paused") from exc

                last_error = exc
                err_msg = " | ".join([str(exc), *logger.errors])

                if _is_permanent_youtube_error(err_msg):
                    raise DownloadError(
                        f"فيديو يوتيوب غير متاح أو خاص / YouTube video unavailable or private ({source_id}): {exc}"
                    ) from exc

                # If offline, wait for reconnect without burning retry budget unnecessarily
                if not is_online():
                    _wait_for_network_or_control(cancel_event, pause_event, on_status=on_status)
                    continue

                lower_err = err_msg.lower()
                # Rotate smoothly through ALL profiles on SABR/403/format-unavailable errors
                if _should_use_fallback(err_msg) and profile_index < last_profile_idx:
                    profile_index += 1
                    _wait_interruptible(1.0, cancel_event, pause_event)
                    attempt += 1
                    continue

                # In Strict Audio mode, once all audio-only client profiles report format unavailable, fail fast
                if (
                    mode == "strict_audio"
                    and profile_index >= last_profile_idx
                    and ("requested format" in lower_err or "format is not available" in lower_err)
                ):
                    break

                attempt += 1
                if attempt >= max_attempts:
                    break

                # Rotate to next profile on general retries as well
                if profile_index < last_profile_idx:
                    profile_index += 1

                delay = min(30.0, 2.0 * (2 ** min(attempt - 1, 4))) * random.uniform(0.5, 1.0)
                _wait_interruptible(delay, cancel_event, pause_event)
    except DownloadCancelledError:
        _cleanup_partial_files(out_dir, safe_id)
        raise

    if mode == "strict_audio":
        raise DownloadError(
            f"لم يتم العثور على بث صوتي فقط في وضع الصوت الصارم (Strict Audio) لـ {source_id} / "
            f"No audio-only stream available in Strict Audio mode for {source_id}: {last_error}"
        ) from last_error

    raise DownloadError(
        f"فشل تحميل الصوت من يوتيوب بعد {max_attempts} محاولات / "
        f"YouTube audio download failed for {source_id}: {last_error}"
    ) from last_error
