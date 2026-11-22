"""Resilient Google Drive downloader with virus-scan confirmation parsing and HTTP Range resume."""

from __future__ import annotations

import html as html_lib
import os
import random
import re
import threading
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlparse, urlunparse

import requests
import urllib3.exceptions

from ..config import AppConfig, clean_filename
from ..network import is_online, wait_for_internet
from .url_parser import (
    DownloadCancelledError,
    DownloadError,
    DownloadPausedError,
    GDriveError,
    GDrivePermissionError,
    GDriveQuotaError,
    extract_gdrive_file_id,
)

CHUNK_SIZE = 256 * 1024  # 256 KB streaming chunks
MAX_CONFIRM_HOPS = 4
MAX_TOTAL_RESUMES = 60

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

_KNOWN_MEDIA_EXTS = frozenset(
    {
        ".mp4",
        ".mkv",
        ".m4a",
        ".mp3",
        ".wav",
        ".ogg",
        ".oga",
        ".webm",
        ".aac",
        ".flac",
        ".opus",
        ".mov",
        ".avi",
        ".wma",
        ".3gp",
        ".ts",
        ".flv",
        ".m4v",
        ".mpeg",
        ".mpg",
    }
)

_CONTENT_TYPE_EXT_MAP: dict[str, str] = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "video/mp4": ".mp4",
    "video/x-matroska": ".mkv",
    "audio/x-matroska": ".mka",
    "video/webm": ".webm",
    "audio/webm": ".webm",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/ogg": ".ogg",
    "video/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/flac": ".flac",
    "audio/aac": ".aac",
    "video/quicktime": ".mov",
    "video/x-msvideo": ".avi",
    "video/x-flv": ".flv",
}

_CONTENT_RANGE_RE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)
_CONTENT_RANGE_416_RE = re.compile(r"bytes\s+\*/(\d+)", re.IGNORECASE)


class _ConfirmPageParser(HTMLParser):
    """Lightweight stdlib HTML parser for Google Drive virus-scan confirmation forms & links."""

    def __init__(self) -> None:
        super().__init__()
        self.forms: list[dict[str, Any]] = []
        self.links: list[dict[str, str]] = []
        self._current_form: dict[str, Any] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if t == "form":
            self._current_form = {
                "id": attr_map.get("id", ""),
                "action": html_lib.unescape(attr_map.get("action", "")),
                "method": attr_map.get("method", "get").lower(),
                "inputs": {},
            }
            self.forms.append(self._current_form)
        elif t == "input" and self._current_form is not None:
            name = attr_map.get("name", "")
            if name:
                self._current_form["inputs"][name] = html_lib.unescape(
                    attr_map.get("value", "")
                )
        elif t == "a":
            href = html_lib.unescape(attr_map.get("href", ""))
            if href:
                self.links.append({"id": attr_map.get("id", ""), "href": href})

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "form":
            self._current_form = None


def parse_gdrive_confirm_url(
    html_text: str,
    base_url: str,
    cookies: requests.cookies.RequestsCookieJar | dict[str, str] | None = None,
) -> str | None:
    """Extract the confirmed direct download URL from a Google Drive virus-scan HTML page."""
    parser = _ConfirmPageParser()
    try:
        parser.feed(html_text)
    except Exception:
        pass

    # 1. Modern Google Drive: <form id="download-form" action="https://drive.usercontent.google.com/download">
    for form in parser.forms:
        action = form.get("action") or ""
        form_id = (form.get("id") or "").lower()
        inputs: dict[str, str] = form.get("inputs") or {}
        if (
            form_id == "download-form"
            or "usercontent.google.com/download" in action.lower()
            or "confirm" in inputs
            or ("download" in action.lower() and ("id" in inputs or "uuid" in inputs))
        ):
            target = urljoin(base_url, action) if action else base_url
            parsed = urlparse(target)
            merged_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
            merged_params.update(inputs)
            if "confirm" not in merged_params:
                merged_params["confirm"] = "t"
            new_query = urlencode(merged_params)
            return urlunparse(parsed._replace(query=new_query))

    # 2. Classic anchor: <a id="uc-download-link" href="/uc?export=download&confirm=...">
    for link in parser.links:
        href = link["href"]
        if link["id"].lower() == "uc-download-link" or "confirm=" in href.lower():
            return urljoin(base_url, href)

    # 3. Cookie download_warning_*
    if cookies:
        for k, v in cookies.items():
            if str(k).startswith("download_warning") and v:
                parsed = urlparse(base_url)
                params = dict(parse_qsl(parsed.query, keep_blank_values=True))
                params["confirm"] = str(v)
                return urlunparse(parsed._replace(query=urlencode(params)))

    # 4. Regex fallback for confirm / uuid tokens embedded in HTML/JSON
    confirm_m = re.search(
        r'(?:name=["\']confirm["\']\s+value=["\']([^"\']+)["\']|confirm=([0-9A-Za-z_-]+)|"confirm"\s*:\s*"([^"]+)")',
        html_text,
        re.IGNORECASE,
    )
    if confirm_m:
        token = next((g for g in confirm_m.groups() if g), None)
        if token:
            parsed = urlparse(base_url)
            params = dict(parse_qsl(parsed.query, keep_blank_values=True))
            params["confirm"] = token
            uuid_m = re.search(
                r'(?:name=["\']uuid["\']\s+value=["\']([^"\']+)["\']|uuid=([0-9A-Za-z_-]+))',
                html_text,
                re.IGNORECASE,
            )
            if uuid_m:
                uval = next((g for g in uuid_m.groups() if g), None)
                if uval:
                    params["uuid"] = uval
            return urlunparse(parsed._replace(query=urlencode(params)))

    return None


def _classify_html_error(html_text: str, status_code: int = 200, final_url: str = "") -> GDriveError:
    """Classify an unexpected HTML response into a clear Arabic+English Google Drive error."""
    lower = (html_text or "").lower()
    url_lower = (final_url or "").lower()

    quota_markers = (
        "quota exceeded",
        "too many users have viewed or downloaded",
        "download quota",
        "can't download this file at this time",
        "cannot download this file at this time",
        "تجاوزت حصة التنزيل",
        "عدد كبير",
    )
    if any(m in lower for m in quota_markers):
        return GDriveQuotaError(
            "تم تجاوز حد التحميل المسموح لهذا الملف على Google Drive. يرجى المحاولة لاحقاً أو حفظ نسخة في حسابك. "
            "| Google Drive download quota exceeded for this file. Please try again later."
        )

    permission_markers = (
        "accounts.google.com",
        "servicelogin",
        "you need access",
        "request access",
        "ask for access",
        "access denied",
        "don't have access",
        "sign in to continue",
        "sign in",
        "طلب إذن الوصول",
        "تسجيل الدخول",
        "تحتاج إلى إذن",
    )
    if (
        status_code in (401, 403)
        or "accounts.google.com" in url_lower
        or any(m in lower for m in permission_markers)
    ):
        return GDrivePermissionError(
            "الملف خاص على Google Drive ويتطلب صلاحية وصول. يرجى مشاركته كـ (أي شخص لديه الرابط). "
            "| Google Drive file is private or requires permission. Please share it as 'Anyone with the link'."
        )

    if status_code == 404 or "does not exist" in lower or "page not found" in lower:
        return GDriveError(
            "الملف غير موجود على Google Drive (404) / Google Drive file not found (404)."
        )

    return GDrivePermissionError(
        "استجاب الخادم بصفحة HTML بدلاً من الملف الصوتي (قد يكون الملف خاصاً أو الرابط غير صالح). "
        "| Server returned an HTML error page instead of media (file may be private or unavailable)."
    )


def _detect_extension_from_headers(headers: dict[str, Any] | requests.structures.CaseInsensitiveDict) -> str | None:
    cd = headers.get("Content-Disposition") or ""
    if cd:
        # RFC 5987 filename*=UTF-8''...
        m_star = re.search(r"filename\*\s*=\s*(?:UTF-8''|utf-8'')([^;\r\n]+)", cd)
        if m_star:
            fname = unquote(m_star.group(1).strip().strip("\"'"))
            ext = Path(fname).suffix.lower()
            if ext in _KNOWN_MEDIA_EXTS:
                return ext
        # Standard filename="..." or filename=...
        m_plain = re.search(r'filename\s*=\s*"([^"]+)"|filename\s*=\s*([^;\r\n]+)', cd)
        if m_plain:
            fname = (m_plain.group(1) or m_plain.group(2) or "").strip().strip("\"'")
            ext = Path(fname).suffix.lower()
            if ext in _KNOWN_MEDIA_EXTS:
                return ext

    ct = (headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
    if ct in _CONTENT_TYPE_EXT_MAP:
        return _CONTENT_TYPE_EXT_MAP[ct]
    return None


def detect_extension_from_magic(file_path: Path) -> str:
    """Detect media file extension from container magic bytes."""
    try:
        with open(file_path, "rb") as f:
            head = f.read(64)
    except OSError:
        return ".bin"

    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"M4A ", b"M4B ", b"F4A "):
            return ".m4a"
        return ".mp4"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        if b"webm" in head.lower():
            return ".webm"
        return ".mkv"
    if head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        # Differentiate ADTS AAC vs MP3 frame sync
        if len(head) >= 2 and head[0] == 0xFF and head[1] in (0xF1, 0xF9):
            return ".aac"
        return ".mp3"
    if len(head) >= 12 and head[:4] == b"RIFF":
        if head[8:12] == b"WAVE":
            return ".wav"
        if head[8:12] == b"AVI ":
            return ".avi"
    if head.startswith(b"OggS"):
        return ".ogg"
    if head.startswith(b"fLaC"):
        return ".flac"
    if head.startswith(b"FLV\x01"):
        return ".flv"
    return ".bin"


def _resolve_initial_url(url: str, source_id: str) -> str:
    """Normalize Google Drive share URLs to /uc?export=download, preserving direct/local URLs."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path_lower = parsed.path.lower().rstrip("/")

    # Preserve direct download URLs and non-Google URLs (e.g., local test HTTP servers)
    if host not in ("drive.google.com", "docs.google.com"):
        return url
    if path_lower in ("/uc", "/download") or path_lower.endswith("/uc"):
        return url

    fid = extract_gdrive_file_id(url) or source_id
    return f"https://drive.google.com/uc?export=download&id={fid}"


def _is_loopback_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("127.0.0.1", "localhost", "::1")


def _is_real_gdrive_url(url: str) -> bool:
    if _is_loopback_url(url):
        return False
    host = (urlparse(url).hostname or "").lower()
    return host in ("drive.google.com", "docs.google.com")


class _QuietDriveYtdlpLogger:
    """Suppress noisy yt-dlp output during quick Google Drive smart-saver probes."""

    def debug(self, msg: str) -> None:
        pass

    def info(self, msg: str) -> None:
        pass

    def warning(self, msg: str) -> None:
        pass

    def error(self, msg: str) -> None:
        pass


def _is_source_drive_format(fmt: dict[str, Any]) -> bool:
    fid = str(fmt.get("format_id") or "").strip().lower()
    note = str(fmt.get("format_note") or "").strip().lower()
    return fid == "source" or "source" in note or "original" in note


def _format_filesize(fmt: dict[str, Any]) -> int | None:
    for key in ("filesize", "filesize_approx"):
        val = fmt.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    return None


def _select_gdrive_transcoded_format(info: dict[str, Any] | None) -> str | None:
    """Return a lightweight yt-dlp format selector if Google Drive offers a genuine transcoded stream."""
    if not isinstance(info, dict):
        return None
    formats = info.get("formats") or []
    if not isinstance(formats, list) or not formats:
        return None

    dict_formats = [f for f in formats if isinstance(f, dict)]
    if not dict_formats:
        return None

    source_fmts = [f for f in dict_formats if _is_source_drive_format(f)]
    non_source_fmts = [f for f in dict_formats if not _is_source_drive_format(f)]
    if not non_source_fmts:
        return None

    source_sizes = [s for s in (_format_filesize(f) for f in source_fmts) if s is not None]
    max_source_size = max(source_sizes) if source_sizes else None

    # Check if the file is already pure audio on Drive (prefer direct Range download unless a much smaller stream exists)
    info_ext = str(info.get("ext") or "").strip().lower().lstrip(".")
    any_video = any(
        f.get("vcodec") not in (None, "none")
        for f in dict_formats
    )
    if not any_video and info_ext in {"mp3", "m4a", "aac", "ogg", "opus", "wav", "flac"}:
        if max_source_size is None:
            return None
        smaller_audio = [
            f
            for f in non_source_fmts
            if (_format_filesize(f) or max_source_size) < max_source_size * 0.85
        ]
        if not smaller_audio:
            return None

    audio_only = [
        f
        for f in non_source_fmts
        if f.get("vcodec") == "none" and f.get("acodec") not in (None, "none")
    ]
    low_video = [
        f
        for f in non_source_fmts
        if (isinstance(f.get("height"), (int, float)) and 0 < float(f["height"]) <= 360)
        or str(f.get("format_id") or "") == "18"
    ]

    if max_source_size is not None:
        if audio_only:
            known = [_format_filesize(f) for f in audio_only if _format_filesize(f) is not None]
            if known and min(known) >= max_source_size * 0.85:
                audio_only = []
        if low_video:
            known = [_format_filesize(f) for f in low_video if _format_filesize(f) is not None]
            if known and min(known) >= max_source_size * 0.85:
                low_video = []

    if audio_only or low_video:
        return "bestaudio/worst[height>=240][height<=360]/worst[height>=240]/18"

    if max_source_size is not None:
        smaller_other = [
            f
            for f in non_source_fmts
            if (_format_filesize(f) or max_source_size) < max_source_size * 0.85
        ]
        if smaller_other:
            best = min(smaller_other, key=lambda f: _format_filesize(f) or max_source_size)
            fmt_id = str(best.get("format_id") or "").strip()
            if fmt_id:
                return fmt_id

    return None


def _cleanup_ytdlp_gdrive_partials(out_dir: Path, safe_id: str) -> None:
    for entry in out_dir.glob(f"{safe_id}.*"):
        if not entry.is_file():
            continue
        lower = entry.name.lower()
        if lower == f"{safe_id.lower()}.gdrive.part":
            continue
        if lower.endswith((".part", ".ytdl", ".temp")) or ".part-frag" in lower or entry.stat().st_size == 0:
            try:
                entry.unlink()
            except OSError:
                pass


def _find_ytdlp_gdrive_file(out_dir: Path, safe_id: str, recorded_path: str | None) -> Path | None:
    if recorded_path:
        p = Path(recorded_path)
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            low = p.name.lower()
            if not low.endswith((".part", ".ytdl", ".temp", ".gdrive.part")):
                return p
    prefix = f"{safe_id}."
    candidates: list[Path] = []
    for entry in out_dir.iterdir():
        if not entry.is_file():
            continue
        low = entry.name.lower()
        if not entry.name.startswith(prefix):
            continue
        if low.endswith((".part", ".ytdl", ".temp", ".gdrive.part")) or ".part-frag" in low:
            continue
        if "_chunk_" in low or low.endswith("_converted.mp3"):
            continue
        if entry.stat().st_size > 0:
            candidates.append(entry)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _try_gdrive_transcoded_download(
    url: str,
    source_id: str,
    out_dir: Path,
    safe_id: str,
    part_path: Path,
    cfg: AppConfig,
    progress_cb: Callable[[float, float, float | None, int, int], Any] | None = None,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
    on_status: Callable[..., Any] | None = None,
) -> str | None:
    """Attempt lightweight transcoded stream download via yt-dlp on real Drive URLs.

    Returns the downloaded file path if a smaller transcoded stream was fetched cleanly,
    or ``None`` to fall back silently to the direct resumable HTTP Range downloader.
    """
    if not _is_real_gdrive_url(url):
        return None

    mode = str(getattr(cfg, "download_mode", "eco") or "eco").strip().lower()
    if mode not in ("eco", "balanced", "strict_audio"):
        return None

    _check_control(cancel_event, pause_event, part_path)
    _wait_for_network(url, cancel_event, pause_event, part_path, on_status=on_status)

    fid = extract_gdrive_file_id(url) or source_id
    probe_url = f"https://drive.google.com/file/d/{fid}/view" if fid else url
    probe_timeout = min(10, max(5, int(cfg.socket_timeout or 30)))
    logger = _QuietDriveYtdlpLogger()

    try:
        import yt_dlp
    except Exception:
        return None

    probe_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": probe_timeout,
        "retries": 0,
        "extractor_retries": 0,
        "logger": logger,
    }
    if cfg.proxy:
        probe_opts["proxy"] = cfg.proxy

    try:
        with yt_dlp.YoutubeDL(probe_opts) as ydl:
            info = ydl.extract_info(probe_url, download=False)
    except (DownloadCancelledError, DownloadPausedError):
        raise
    except Exception:
        _check_control(cancel_event, pause_event, part_path)
        return None

    _check_control(cancel_event, pause_event, part_path)
    fmt_spec = _select_gdrive_transcoded_format(info)
    if not fmt_spec:
        return None

    out_tmpl = str(out_dir / f"{safe_id}.%(ext)s")
    recorded: dict[str, str | None] = {"path": None}

    def _hook(d: dict[str, Any]) -> None:
        _check_control(cancel_event, pause_event, part_path)
        status = d.get("status")
        if status == "finished":
            if d.get("filename"):
                recorded["path"] = str(d["filename"])
            total = int(d.get("total_bytes") or d.get("downloaded_bytes") or 0)
            if progress_cb:
                try:
                    progress_cb(100.0, float(d.get("speed") or 0.0), 0.0, total, total)
                except Exception:
                    pass
        elif status == "downloading" and progress_cb:
            downloaded = int(d.get("downloaded_bytes") or 0)
            total = int(d.get("total_bytes") or d.get("total_bytes_estimate") or 0)
            speed = float(d.get("speed") or 0.0)
            eta = d.get("eta")
            eta_val = float(eta) if isinstance(eta, (int, float)) else None
            pct = min(100.0, (downloaded / total) * 100.0) if total > 0 else 0.0
            try:
                progress_cb(pct, speed, eta_val, downloaded, total)
            except Exception:
                pass

    dl_opts: dict[str, Any] = {
        "outtmpl": out_tmpl,
        "format": fmt_spec,
        "continuedl": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 1,
        "fragment_retries": 1,
        "socket_timeout": min(15, max(5, int(cfg.socket_timeout or 30))),
        "progress_hooks": [_hook],
        "logger": logger,
    }
    if cfg.proxy:
        dl_opts["proxy"] = cfg.proxy

    try:
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            dl_info = ydl.extract_info(probe_url, download=True)
            if not recorded["path"] and isinstance(dl_info, dict):
                req_dls = dl_info.get("requested_downloads") or []
                if req_dls and isinstance(req_dls[0], dict) and req_dls[0].get("filepath"):
                    recorded["path"] = str(req_dls[0]["filepath"])
                else:
                    try:
                        recorded["path"] = ydl.prepare_filename(dl_info)
                    except Exception:
                        pass

        found = _find_ytdlp_gdrive_file(out_dir, safe_id, recorded["path"])
        if found is None:
            _cleanup_ytdlp_gdrive_partials(out_dir, safe_id)
            return None

        # Safety check: verify yt-dlp didn't save an HTML page
        with open(found, "rb") as fh:
            head = fh.read(64)
        if not head or _looks_like_html_bytes(head):
            try:
                found.unlink()
            except OSError:
                pass
            _cleanup_ytdlp_gdrive_partials(out_dir, safe_id)
            return None

        return str(found)
    except DownloadCancelledError:
        _cleanup_ytdlp_gdrive_partials(out_dir, safe_id)
        raise
    except DownloadPausedError:
        raise
    except Exception:
        if cancel_event is not None and cancel_event.is_set():
            _cleanup_ytdlp_gdrive_partials(out_dir, safe_id)
            raise DownloadCancelledError("تم إلغاء التحميل / Download cancelled")
        if pause_event is not None and pause_event.is_set():
            raise DownloadPausedError("تم إيقاف التحميل مؤقتاً / Download paused")
        _cleanup_ytdlp_gdrive_partials(out_dir, safe_id)
        return None


def _looks_like_html_bytes(chunk: bytes) -> bool:
    sample = chunk.lstrip()[:64].lower()
    return sample.startswith((b"<!doctype html", b"<html", b"<head", b"<body", b"<!--"))


def _iter_response_chunks(
    resp: requests.Response,
    chunk_size: int = CHUNK_SIZE,
) -> Iterator[bytes]:
    """Stream chunks via urllib3 ``read1`` so zero bytes are lost on mid-chunk disconnects.

    Falls back to ``resp.iter_content`` for non-urllib3 response objects (e.g., mocks).
    """
    raw = getattr(resp, "raw", None)
    read1 = getattr(raw, "read1", None)
    if callable(read1):
        try:
            while True:
                data = read1(chunk_size, decode_content=True)
                if not data:
                    break
                yield data
        except urllib3.exceptions.HTTPError as exc:
            raise requests.exceptions.ChunkedEncodingError(exc) from exc
        return

    yield from resp.iter_content(chunk_size=chunk_size)


def _check_control(
    cancel_event: threading.Event | None,
    pause_event: threading.Event | None,
    part_path: Path,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        try:
            if part_path.exists():
                part_path.unlink()
        except OSError:
            pass
        raise DownloadCancelledError("تم إلغاء التحميل / Download cancelled")
    if pause_event is not None and pause_event.is_set():
        # Strictly preserve .gdrive.part on pause!
        raise DownloadPausedError("تم إيقاف التحميل مؤقتاً / Download paused")


def _interruptible_sleep(
    seconds: float,
    cancel_event: threading.Event | None,
    pause_event: threading.Event | None,
    part_path: Path,
) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        _check_control(cancel_event, pause_event, part_path)
        rem = deadline - time.monotonic()
        if rem <= 0:
            return
        step = min(0.2, rem)
        if cancel_event is not None:
            if cancel_event.wait(timeout=step):
                _check_control(cancel_event, pause_event, part_path)
        elif pause_event is not None:
            if pause_event.wait(timeout=step):
                _check_control(cancel_event, pause_event, part_path)
        else:
            time.sleep(step)


def _wait_for_network(
    url: str,
    cancel_event: threading.Event | None,
    pause_event: threading.Event | None,
    part_path: Path,
    on_status: Callable[..., Any] | None = None,
) -> None:
    _check_control(cancel_event, pause_event, part_path)
    if _is_loopback_url(url) or is_online():
        return

    bridge = threading.Event()

    def _watch() -> None:
        while not bridge.is_set():
            if (cancel_event and cancel_event.is_set()) or (
                pause_event and pause_event.is_set()
            ):
                bridge.set()
                return
            bridge.wait(timeout=0.2)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    try:
        reconnected = wait_for_internet(
            stop_event=bridge, check_interval=3.0, on_status=on_status
        )
    finally:
        bridge.set()
        t.join(timeout=1.0)

    _check_control(cancel_event, pause_event, part_path)
    if not reconnected:
        raise GDriveError("انقطع الاتصال بالإنترنت / Internet connection unavailable")


def download_gdrive_file(
    url: str,
    source_id: str,
    temp_dir: str | Path,
    config: AppConfig | None = None,
    progress_cb: Callable[[float, float, float | None, int, int], Any] | None = None,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
    on_status: Callable[..., Any] | None = None,
) -> str:
    """Download a public/shared Google Drive file of any size with HTTP Range resume.

    Features:
    - Automatically handles Google Drive virus-scan warning pages (>100MB files) by
      parsing ``download-form``, ``confirm`` / ``uuid`` tokens, and ``drive.usercontent.google.com``.
    - Saves incrementally to ``<source_id>.gdrive.part`` and resumes from the exact byte
      offset via ``Range: bytes=<offset>-`` across disconnects or pause/resume.
    - Detects HTML error pages (private file, quota exceeded) and raises bilingual
      ``GDrivePermissionError`` / ``GDriveQuotaError``.
    - Detects output file extension from ``Content-Disposition``, ``Content-Type``, or magic bytes.
    """
    cfg = config or AppConfig()
    out_dir = Path(temp_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = clean_filename(source_id or "gdrive_file", max_length=80)
    part_path = out_dir / f"{safe_id}.gdrive.part"

    # Smart bandwidth saver: try Google's lightweight transcoded stream via yt-dlp on real Drive URLs
    transcoded_path = _try_gdrive_transcoded_download(
        url=url,
        source_id=source_id,
        out_dir=out_dir,
        safe_id=safe_id,
        part_path=part_path,
        cfg=cfg,
        progress_cb=progress_cb,
        cancel_event=cancel_event,
        pause_event=pause_event,
        on_status=on_status,
    )
    if transcoded_path is not None:
        return transcoded_path

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": _USER_AGENT,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
        }
    )
    if cfg.proxy and not _is_loopback_url(url):
        session.proxies.update({"http": cfg.proxy, "https": cfg.proxy})

    initial_url = _resolve_initial_url(url, source_id)
    active_url = initial_url
    timeout = (cfg.socket_timeout, cfg.socket_timeout)
    max_consecutive_failures = max(1, int(cfg.max_retries) + 1)

    consecutive_failures = 0
    total_resumes = 0
    detected_ext: str | None = None
    last_error: Exception | None = None
    last_speed = 0.0

    try:
        while total_resumes < MAX_TOTAL_RESUMES:
            _check_control(cancel_event, pause_event, part_path)
            _wait_for_network(active_url, cancel_event, pause_event, part_path, on_status=on_status)

            offset = part_path.stat().st_size if part_path.exists() else 0
            start_offset = offset
            req_headers: dict[str, str] = {}
            if offset > 0:
                req_headers["Range"] = f"bytes={offset}-"

            resp: requests.Response | None = None
            try:
                # Resolve virus-scan confirmation hops before streaming
                confirm_hops = 0
                while True:
                    _check_control(cancel_event, pause_event, part_path)
                    resp = session.get(
                        active_url,
                        headers=req_headers,
                        stream=True,
                        timeout=timeout,
                        allow_redirects=True,
                    )

                    if resp.status_code in (401, 403):
                        body = resp.text[:65536]
                        final_u = resp.url
                        resp.close()
                        raise _classify_html_error(body, status_code=resp.status_code, final_url=final_u)

                    if resp.status_code == 404:
                        body = resp.text[:65536]
                        final_u = resp.url
                        resp.close()
                        raise _classify_html_error(body, status_code=404, final_url=final_u)

                    if resp.status_code == 416:
                        cr = resp.headers.get("Content-Range") or ""
                        resp.close()
                        m416 = _CONTENT_RANGE_416_RE.search(cr)
                        if m416 and int(m416.group(1)) == offset and offset > 0:
                            # File already completely downloaded in .part
                            final_ext = detected_ext or detect_extension_from_magic(part_path)
                            final_path = out_dir / f"{safe_id}{final_ext}"
                            os.replace(part_path, final_path)
                            if progress_cb:
                                try:
                                    progress_cb(100.0, last_speed, 0.0, offset, offset)
                                except Exception:
                                    pass
                            return str(final_path)
                        # Offset invalid; truncate and restart from 0
                        if part_path.exists():
                            part_path.unlink()
                        offset = 0
                        req_headers.pop("Range", None)
                        continue

                    if resp.status_code >= 500:
                        code = resp.status_code
                        resp.close()
                        raise requests.HTTPError(f"Server error HTTP {code}")

                    content_type = (resp.headers.get("Content-Type") or "").lower()
                    if "text/html" in content_type:
                        html_body = resp.text
                        final_u = resp.url
                        resp_cookies = resp.cookies
                        resp.close()

                        next_url = parse_gdrive_confirm_url(
                            html_body, base_url=final_u, cookies=resp_cookies
                        )
                        if next_url and confirm_hops < MAX_CONFIRM_HOPS:
                            confirm_hops += 1
                            active_url = next_url
                            continue

                        # If confirmed URL expired on a retry, fall back once to initial URL
                        if active_url != initial_url and confirm_hops == 0:
                            active_url = initial_url
                            confirm_hops += 1
                            continue

                        raise _classify_html_error(
                            html_body, status_code=resp.status_code, final_url=final_u
                        )

                    resp.raise_for_status()
                    break

                # Now we have a media response (200 or 206)
                header_ext = _detect_extension_from_headers(resp.headers)
                if header_ext:
                    detected_ext = header_ext

                total_bytes = 0
                write_mode = "wb"
                write_offset = 0

                if resp.status_code == 206:
                    cr_header = resp.headers.get("Content-Range") or ""
                    m_cr = _CONTENT_RANGE_RE.search(cr_header)
                    if m_cr:
                        range_start = int(m_cr.group(1))
                        range_total = m_cr.group(3)
                        if range_total != "*":
                            total_bytes = int(range_total)
                        else:
                            cl = int(resp.headers.get("Content-Length") or 0)
                            total_bytes = range_start + cl if cl > 0 else 0
                        write_offset = range_start
                    else:
                        cl = int(resp.headers.get("Content-Length") or 0)
                        write_offset = offset
                        total_bytes = offset + cl if cl > 0 else 0

                    if write_offset == offset and part_path.exists():
                        write_mode = "ab"
                    elif write_offset > 0 and part_path.exists():
                        write_mode = "r+b"
                    else:
                        write_mode = "wb"
                        write_offset = 0
                else:
                    # HTTP 200: full stream from byte 0
                    cl = int(resp.headers.get("Content-Length") or 0)
                    total_bytes = cl if cl > 0 else 0
                    write_mode = "wb"
                    write_offset = 0

                downloaded = write_offset
                win_start = time.monotonic()
                win_bytes = 0
                last_cb_time = 0.0
                first_chunk_checked = write_offset > 0
                html_redirect_triggered = False

                with open(part_path, write_mode) as fh:
                    if write_mode == "r+b":
                        fh.seek(write_offset)
                        fh.truncate(write_offset)

                    chunk_iter = _iter_response_chunks(resp, chunk_size=CHUNK_SIZE)
                    for chunk in chunk_iter:
                        _check_control(cancel_event, pause_event, part_path)
                        if not chunk:
                            continue

                        if not first_chunk_checked:
                            first_chunk_checked = True
                            if _looks_like_html_bytes(chunk):
                                # Server sent HTML without text/html Content-Type
                                extra = b"".join(
                                    [chunk]
                                    + [c for _, c in zip(range(4), chunk_iter) if c]
                                )
                                html_body = extra.decode("utf-8", errors="replace")
                                final_u = resp.url
                                resp_cookies = resp.cookies
                                fh.close()
                                resp.close()
                                if part_path.exists():
                                    part_path.unlink()
                                next_url = parse_gdrive_confirm_url(
                                    html_body, base_url=final_u, cookies=resp_cookies
                                )
                                if next_url:
                                    active_url = next_url
                                    html_redirect_triggered = True
                                    break
                                raise _classify_html_error(html_body, final_url=final_u)

                        fh.write(chunk)
                        chunk_len = len(chunk)
                        downloaded += chunk_len
                        win_bytes += chunk_len

                        now = time.monotonic()
                        elapsed = now - win_start
                        if elapsed >= 0.2:
                            inst_speed = win_bytes / elapsed
                            last_speed = (
                                inst_speed
                                if last_speed <= 0
                                else (0.7 * last_speed + 0.3 * inst_speed)
                            )
                            win_start = now
                            win_bytes = 0

                        if progress_cb and (now - last_cb_time >= 0.1 or (total_bytes > 0 and downloaded >= total_bytes)):
                            last_cb_time = now
                            pct = (
                                min(100.0, (downloaded / total_bytes) * 100.0)
                                if total_bytes > 0
                                else 0.0
                            )
                            eta = (
                                max(0.0, (total_bytes - downloaded) / last_speed)
                                if total_bytes > 0 and last_speed > 0
                                else None
                            )
                            try:
                                progress_cb(pct, last_speed, eta, downloaded, total_bytes)
                            except Exception:
                                pass

                    fh.flush()

                resp.close()

                if html_redirect_triggered:
                    continue

                # Check for silent premature EOF (connection closed before Content-Length reached)
                actual_size = part_path.stat().st_size if part_path.exists() else 0
                if total_bytes > 0 and actual_size < total_bytes:
                    if actual_size > start_offset:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                    total_resumes += 1
                    if consecutive_failures >= max_consecutive_failures:
                        raise GDriveError(
                            f"توقف تحميل Google Drive عند {actual_size}/{total_bytes} بايت / "
                            f"Incomplete Google Drive download ({actual_size}/{total_bytes} bytes)"
                        )
                    delay = 0.05 if _is_loopback_url(active_url) else min(15.0, 1.5 * (2 ** min(consecutive_failures, 3)))
                    _interruptible_sleep(delay, cancel_event, pause_event, part_path)
                    continue

                if actual_size == 0:
                    raise GDriveError(
                        "الملف المحمّل من Google Drive فارغ (0 بايت) / Downloaded Google Drive file is empty (0 bytes)."
                    )

                final_ext = detected_ext or detect_extension_from_magic(part_path)
                final_path = out_dir / f"{safe_id}{final_ext}"
                os.replace(part_path, final_path)

                if progress_cb:
                    try:
                        progress_cb(100.0, last_speed, 0.0, actual_size, actual_size)
                    except Exception:
                        pass

                return str(final_path)

            except (DownloadCancelledError, DownloadPausedError, GDrivePermissionError, GDriveQuotaError):
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
                raise
            except GDriveError as exc:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
                # 404 is permanent; other GDriveErrors may have exhausted retries
                if "404" in str(exc) or "empty (0 bytes)" in str(exc):
                    raise
                last_error = exc
                raise
            except (requests.RequestException, OSError) as exc:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass

                _check_control(cancel_event, pause_event, part_path)
                last_error = exc
                new_size = part_path.stat().st_size if part_path.exists() else 0

                if new_size > start_offset:
                    # Real bytes were persisted; reset consecutive failure streak and resume!
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

                total_resumes += 1

                if not _is_loopback_url(active_url) and not is_online():
                    _wait_for_network(
                        active_url, cancel_event, pause_event, part_path, on_status=on_status
                    )
                    continue

                if consecutive_failures >= max_consecutive_failures:
                    break

                delay = (
                    0.05
                    if _is_loopback_url(active_url)
                    else min(30.0, 2.0 * (2 ** min(consecutive_failures - 1, 4))) * random.uniform(0.5, 1.0)
                )
                _interruptible_sleep(delay, cancel_event, pause_event, part_path)

        raise GDriveError(
            f"فشل تحميل ملف Google Drive بعد عدة محاولات / "
            f"Google Drive download failed for {source_id}: {last_error}"
        ) from last_error
    finally:
        session.close()
