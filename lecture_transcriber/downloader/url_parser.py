"""URL detection, normalization, and expansion for YouTube and Google Drive."""

from __future__ import annotations

import html as html_lib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlparse

import requests

from ..config import AppConfig, clean_filename


# ---------------------------------------------------------------------------
# Shared Downloader Exception Hierarchy
# ---------------------------------------------------------------------------


class DownloadError(Exception):
    """Base exception for all downloader and audio-processing errors."""


class DownloadPausedError(DownloadError):
    """Raised when an active download is paused by the user (preserves .part files)."""


class DownloadCancelledError(DownloadError):
    """Raised when an active download is cancelled by the user."""


class URLParseError(DownloadError):
    """Raised when a URL is malformed, unsupported, or cannot be parsed."""


class GDriveError(DownloadError):
    """Base exception for Google Drive download/folder errors."""


class GDrivePermissionError(GDriveError):
    """Raised when a Google Drive file or folder is private or requires sign-in."""


class GDriveQuotaError(GDriveError):
    """Raised when a Google Drive file has exceeded its public download quota."""


class AudioProcessingError(DownloadError):
    """Raised when FFmpeg/FFprobe inspection, conversion, or splitting fails."""


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------


@dataclass
class ParsedMediaItem:
    """Normalized media entry ready to be queued for download and transcription."""

    source_type: str  # "youtube" | "gdrive"
    url: str
    source_id: str
    title: str
    seq_num: int = 1
    duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Regex & Detection Constants
# ---------------------------------------------------------------------------

_YT_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtube-nocookie.com",
        "www.youtube-nocookie.com",
        "youtu.be",
        "www.youtu.be",
    }
)

_GDRIVE_HOSTS = frozenset(
    {
        "drive.google.com",
        "docs.google.com",
        "drive.usercontent.google.com",
    }
)

_YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,16}$")
_GDRIVE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{5,}$")
_URL_EXTRACT_RE = re.compile(r"https?://[^\s<>\"'\]\)]+")

_GDRIVE_FILE_PATH_RE = re.compile(
    r"^/(?:u/\d+/)?file/(?:u/\d+/)?d/([A-Za-z0-9_-]{5,})(?:/|$)"
)
_GDRIVE_FOLDER_PATH_RE = re.compile(
    r"^/(?:drive/(?:u/\d+/)?(?:mobile/)?folders|folders)/([A-Za-z0-9_-]{5,})(?:/|$)"
)

_NON_MEDIA_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".txt",
        ".md",
        ".rtf",
        ".csv",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".svg",
        ".ico",
        ".zip",
        ".rar",
        ".7z",
        ".tar",
        ".gz",
        ".html",
        ".htm",
        ".css",
        ".js",
        ".json",
        ".xml",
        ".exe",
        ".msi",
        ".apk",
    }
)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# URL Normalization & ID Extraction Helpers
# ---------------------------------------------------------------------------


def _normalize_raw_url(raw: str) -> str:
    url = raw.strip().strip("\"'<>")
    if not url:
        return ""
    if "://" not in url:
        lower = url.lower()
        if lower.startswith(
            (
                "youtube.com/",
                "www.youtube.com/",
                "m.youtube.com/",
                "youtu.be/",
                "drive.google.com/",
                "docs.google.com/",
            )
        ):
            url = f"https://{url}"
    return url


def extract_youtube_id(url: str) -> str | None:
    """Extract a YouTube video ID from watch, youtu.be, shorts, live, or embed URLs."""
    norm = _normalize_raw_url(url)
    if not norm:
        return None
    parsed = urlparse(norm)
    host = (parsed.hostname or "").lower()
    if host not in _YT_HOSTS:
        return None

    path_parts = [p for p in parsed.path.split("/") if p]
    qs = parse_qs(parsed.query)

    if host in ("youtu.be", "www.youtu.be"):
        if path_parts and _YT_ID_RE.match(path_parts[0]):
            return path_parts[0]
        return None

    if "v" in qs and qs["v"]:
        candidate = qs["v"][0].strip()
        if _YT_ID_RE.match(candidate):
            return candidate

    if len(path_parts) >= 2 and path_parts[0] in ("shorts", "live", "embed", "v"):
        candidate = path_parts[1].strip()
        if _YT_ID_RE.match(candidate):
            return candidate

    return None


def extract_youtube_playlist_id(url: str) -> str | None:
    """Extract a YouTube playlist ID from a URL containing a `list=` query parameter."""
    norm = _normalize_raw_url(url)
    if not norm:
        return None
    parsed = urlparse(norm)
    host = (parsed.hostname or "").lower()
    if host not in _YT_HOSTS:
        return None
    qs = parse_qs(parsed.query)
    if "list" in qs and qs["list"]:
        pid = qs["list"][0].strip()
        if pid and re.match(r"^[A-Za-z0-9_-]{6,}$", pid):
            return pid
    return None


def extract_gdrive_file_id(url: str) -> str | None:
    """Extract a Google Drive file ID from /file/d/<ID>, open?id=<ID>, or uc?id=<ID>."""
    norm = _normalize_raw_url(url)
    if not norm:
        return None
    parsed = urlparse(norm)
    host = (parsed.hostname or "").lower()
    if host not in _GDRIVE_HOSTS:
        return None

    m = _GDRIVE_FILE_PATH_RE.match(parsed.path)
    if m:
        return m.group(1)

    path_lower = parsed.path.lower().rstrip("/")
    qs = parse_qs(parsed.query)
    if path_lower in ("/open", "/uc", "/download") or path_lower.endswith("/uc"):
        if "id" in qs and qs["id"]:
            cid = qs["id"][0].strip()
            if _GDRIVE_ID_RE.match(cid):
                return cid

    return None


def extract_gdrive_folder_id(url: str) -> str | None:
    """Extract a Google Drive shared folder ID from /drive/folders/<ID> or folderview?id=<ID>."""
    norm = _normalize_raw_url(url)
    if not norm:
        return None
    parsed = urlparse(norm)
    host = (parsed.hostname or "").lower()
    if host not in _GDRIVE_HOSTS:
        return None

    m = _GDRIVE_FOLDER_PATH_RE.match(parsed.path)
    if m:
        return m.group(1)

    path_lower = parsed.path.lower().rstrip("/")
    qs = parse_qs(parsed.query)
    if path_lower in ("/folderview", "/embeddedfolderview") and "id" in qs and qs["id"]:
        cid = qs["id"][0].strip()
        if _GDRIVE_ID_RE.match(cid):
            return cid

    return None


def detect_url_type(url: str, include_playlist: bool = False) -> tuple[str, str]:
    """Classify a URL into (kind, id).

    Returns one of:
        ("youtube_playlist", playlist_id)
        ("youtube_video", video_id)
        ("gdrive_folder", folder_id)
        ("gdrive_file", file_id)
    Raises:
        URLParseError if URL is not a recognized YouTube or Google Drive link.
    """
    norm = _normalize_raw_url(url)
    if not norm:
        raise URLParseError("الرابط فارغ / Empty URL")

    parsed = urlparse(norm)
    host = (parsed.hostname or "").lower()
    path_lower = parsed.path.lower().rstrip("/")

    if host in _YT_HOSTS:
        playlist_id = extract_youtube_playlist_id(norm)
        video_id = extract_youtube_id(norm)
        if path_lower == "/playlist" and playlist_id:
            return ("youtube_playlist", playlist_id)
        if playlist_id and (include_playlist or not video_id):
            return ("youtube_playlist", playlist_id)
        if video_id:
            return ("youtube_video", video_id)
        raise URLParseError(
            f"رابط يوتيوب غير صالح أو لا يحتوي على معرف فيديو / Invalid YouTube URL: {url}"
        )

    if host in _GDRIVE_HOSTS:
        folder_id = extract_gdrive_folder_id(norm)
        if folder_id:
            return ("gdrive_folder", folder_id)
        file_id = extract_gdrive_file_id(norm)
        if file_id:
            return ("gdrive_file", file_id)
        raise URLParseError(
            f"رابط Google Drive غير مدعوم / Unsupported Google Drive URL: {url}"
        )

    raise URLParseError(
        f"رابط غير مدعوم (يدعم فقط YouTube و Google Drive) / Unsupported URL: {url}"
    )


# ---------------------------------------------------------------------------
# Metadata & Expansion Implementations
# ---------------------------------------------------------------------------


def _build_session(config: AppConfig | None = None) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": _USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
        }
    )
    if config and config.proxy:
        session.proxies.update({"http": config.proxy, "https": config.proxy})
    return session


def _is_non_media_filename(name: str) -> bool:
    lower = name.strip().lower()
    for ext in _NON_MEDIA_EXTENSIONS:
        if lower.endswith(ext):
            return True
    return False


def _looks_like_private_gdrive_html(html_text: str, final_url: str = "") -> bool:
    final_lower = (final_url or "").lower()
    if "accounts.google.com" in final_lower or "servicelogin" in final_lower:
        return True
    lower = (html_text or "").lower()
    private_markers = (
        "accounts.google.com/v3/signin",
        "accounts.google.com/servicelogin",
        "you need access",
        "request access",
        "ask for access",
        "access denied",
        "sign in to continue to google drive",
        "طلب إذن الوصول",
        "تحتاج إلى إذن",
    )
    return any(marker in lower for marker in private_markers)


def parse_gdrive_folder_html(
    html_text: str,
    folder_id: str = "",
    final_url: str = "",
) -> list[tuple[str, str]]:
    """Parse public Google Drive folder HTML (embeddedfolderview or standard view).

    Returns a list of ``(file_id, title)`` tuples in display order.
    Raises ``GDrivePermissionError`` if the HTML indicates a private/sign-in page.
    """
    if _looks_like_private_gdrive_html(html_text, final_url=final_url):
        raise GDrivePermissionError(
            "مجلد Google Drive خاص ويتطلب صلاحية وصول. يرجى جعل المجلد متاحاً لأي شخص لديه الرابط "
            "(Anyone with the link). | Google Drive folder is private or requires sign-in. "
            "Please set sharing to 'Anyone with the link'."
        )

    entries: list[tuple[str, str]] = []
    seen_ids: set[str] = set()

    def _add(fid: str, title: str) -> None:
        fid = fid.strip()
        if not fid or fid == folder_id or fid in seen_ids or not _GDRIVE_ID_RE.match(fid):
            return
        clean_title = html_lib.unescape(title or "").strip()
        if not clean_title:
            clean_title = f"Drive_{fid}"
        if _is_non_media_filename(clean_title):
            return
        seen_ids.add(fid)
        entries.append((fid, clean_title))

    # Strategy 1: embeddedfolderview markup
    # <div class="flip-entry" id="entry-<ID>" ...>...<div class="flip-entry-title">NAME</div>
    flip_pattern = re.compile(
        r'id=["\']entry-([A-Za-z0-9_-]{5,})["\'][^>]*>.*?'
        r'class=["\']flip-entry-title["\'][^>]*>(.*?)</div>',
        re.DOTALL | re.IGNORECASE,
    )
    for m in flip_pattern.finditer(html_text):
        fid, raw_title = m.group(1), m.group(2)
        # Skip subfolders if link points to /drive/folders/
        snippet = m.group(0)
        if "/folders/" in snippet or "folderview" in snippet:
            continue
        title = re.sub(r"<[^>]+>", "", raw_title).strip()
        _add(fid, title)

    if entries:
        return entries

    # Strategy 2: window['_DRIVE_ivd'] embedded JSON array in standard Drive folder HTML
    ivd_match = re.search(
        r"window\[['\"]_DRIVE_ivd['\"]\]\s*=\s*'([^']+)'", html_text
    )
    if ivd_match:
        raw_js = ivd_match.group(1)
        try:
            # Decode JS \xHH hex escapes and \' while preserving literal UTF-8 and JSON \uXXXX escapes
            decoded = re.sub(
                r"\\x([0-9a-fA-F]{2})",
                lambda hx: chr(int(hx.group(1), 16)),
                raw_js,
            )
            decoded = decoded.replace("\\'", "'")
            payload = json.loads(decoded)
            items = payload[0] if isinstance(payload, list) and payload and isinstance(payload[0], list) else []
            for item in items:
                if not isinstance(item, list) or len(item) < 4:
                    continue
                fid = item[0]
                title = item[2]
                mime = item[3] if len(item) > 3 else ""
                if isinstance(mime, str) and mime == "application/vnd.google-apps.folder":
                    continue
                if isinstance(fid, str) and isinstance(title, str):
                    _add(fid, title)
        except Exception:
            pass

    if entries:
        return entries

    # Strategy 3: data-id attributes paired with file links or labels in standard HTML
    data_id_pattern = re.compile(
        r'data-id=["\']([A-Za-z0-9_-]{5,})["\'][^>]*>.*?'
        r'(?:data-tooltip|aria-label)=["\']([^"\']+)["\']',
        re.DOTALL | re.IGNORECASE,
    )
    for m in data_id_pattern.finditer(html_text):
        fid, label = m.group(1), m.group(2)
        if "folder" in label.lower() and not re.search(r"\.[a-z0-9]{2,4}$", label.lower()):
            continue
        _add(fid, label)

    if entries:
        return entries

    # Strategy 4: direct /file/d/<ID> anchor tags
    link_pattern = re.compile(
        r'href=["\'][^"\']*drive\.google\.com/file/d/([A-Za-z0-9_-]{5,})[^"\']*["\'][^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    for m in link_pattern.finditer(html_text):
        fid = m.group(1)
        inner = re.sub(r"<[^>]+>", " ", m.group(2)).strip()
        inner = re.sub(r"\s+", " ", inner)
        _add(fid, inner)

    return entries


def _fetch_gdrive_folder(
    folder_id: str,
    config: AppConfig | None = None,
) -> list[ParsedMediaItem]:
    timeout = config.socket_timeout if config else 30
    session = _build_session(config)
    candidate_urls = [
        f"https://drive.google.com/embeddedfolderview?id={folder_id}#list",
        f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing",
    ]

    last_error: Exception | None = None
    for idx, folder_url in enumerate(candidate_urls):
        try:
            resp = session.get(folder_url, timeout=timeout, allow_redirects=True)
            if resp.status_code in (401, 403):
                raise GDrivePermissionError(
                    "مجلد Google Drive خاص ويتطلب صلاحية وصول. يرجى جعل المجلد متاحاً لأي شخص لديه الرابط "
                    "(Anyone with the link). | Google Drive folder is private or requires permission."
                )
            if resp.status_code == 404:
                raise GDriveError(
                    f"مجلد Google Drive غير موجود (404) / Google Drive folder not found: {folder_id}"
                )
            resp.raise_for_status()
            parsed_entries = parse_gdrive_folder_html(
                resp.text, folder_id=folder_id, final_url=resp.url
            )
            if parsed_entries:
                return [
                    ParsedMediaItem(
                        source_type="gdrive",
                        url=f"https://drive.google.com/file/d/{fid}/view",
                        source_id=fid,
                        title=title,
                        seq_num=i,
                        duration_seconds=None,
                    )
                    for i, (fid, title) in enumerate(parsed_entries, start=1)
                ]
        except GDrivePermissionError:
            raise
        except GDriveError:
            raise
        except Exception as exc:
            last_error = exc
            if idx == len(candidate_urls) - 1:
                raise GDriveError(
                    f"فشل قراءة مجلد Google Drive / Failed to read Google Drive folder {folder_id}: {exc}"
                ) from exc

    if last_error:
        raise GDriveError(
            f"فشل قراءة مجلد Google Drive / Failed to read Google Drive folder {folder_id}: {last_error}"
        ) from last_error

    raise GDriveError(
        f"لم يتم العثور على ملفات صوتية/فيديو في مجلد Google Drive أو أن المجلد فارغ/خاص "
        f"/ No downloadable media files found in Google Drive folder {folder_id} (folder may be empty or private)."
    )


def _fetch_gdrive_file_title(file_id: str, config: AppConfig | None = None) -> str:
    """Best-effort title lookup for a single Google Drive file; falls back safely."""
    timeout = min(15, config.socket_timeout if config else 15)
    session = _build_session(config)
    try:
        resp = session.get(
            f"https://drive.google.com/file/d/{file_id}/view",
            timeout=timeout,
            allow_redirects=True,
        )
        if resp.status_code == 200 and "text/html" in (resp.headers.get("Content-Type") or "").lower():
            og = re.search(
                r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)["\']',
                resp.text,
                re.IGNORECASE,
            )
            if og:
                title = html_lib.unescape(og.group(1)).strip()
                if title and "google drive" not in title.lower():
                    return title
            t_match = re.search(r"<title>(.*?)</title>", resp.text, re.IGNORECASE | re.DOTALL)
            if t_match:
                title = html_lib.unescape(t_match.group(1)).strip()
                title = re.sub(r"\s*-\s*Google Drive\s*$", "", title, flags=re.IGNORECASE).strip()
                if title and "sign in" not in title.lower() and "google drive" not in title.lower():
                    return title
    except Exception:
        pass
    return f"Drive_{file_id}"


def _fetch_youtube_metadata(
    url: str,
    video_id: str | None = None,
    is_playlist: bool = False,
    config: AppConfig | None = None,
) -> list[ParsedMediaItem]:
    """Fetch YouTube single-video or playlist metadata via yt-dlp without downloading."""
    import yt_dlp

    timeout = config.socket_timeout if config else 30
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "ignoreerrors": True,
        "socket_timeout": timeout,
        "extract_flat": "in_playlist" if is_playlist else False,
        "noplaylist": not is_playlist,
    }
    if config and config.proxy:
        ydl_opts["proxy"] = config.proxy

    if is_playlist:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
        except Exception as exc:
            raise URLParseError(
                f"فشل قراءة قائمة تشغيل يوتيوب / Failed to expand YouTube playlist: {exc}"
            ) from exc

        if not info or not info.get("entries"):
            raise URLParseError(
                f"قائمة التشغيل فارغة أو خاصة / YouTube playlist is empty or private: {url}"
            )

        items: list[ParsedMediaItem] = []
        for idx, entry in enumerate(info.get("entries") or [], start=1):
            if not isinstance(entry, dict):
                continue
            vid = entry.get("id")
            if not vid:
                entry_url = entry.get("url") or entry.get("webpage_url") or ""
                vid = extract_youtube_id(entry_url)
            if not vid:
                continue
            title = (entry.get("title") or "").strip()
            if title in ("[Private video]", "[Deleted video]"):
                continue
            if not title:
                title = f"YouTube_{vid}"
            dur = entry.get("duration")
            duration_val = float(dur) if isinstance(dur, (int, float)) and dur >= 0 else None
            items.append(
                ParsedMediaItem(
                    source_type="youtube",
                    url=f"https://www.youtube.com/watch?v={vid}",
                    source_id=str(vid),
                    title=title,
                    seq_num=idx,
                    duration_seconds=duration_val,
                )
            )
        return items

    # Single video metadata (fast best-effort with fallback to oEmbed / ID)
    vid = video_id or extract_youtube_id(url) or ""
    canonical_url = f"https://www.youtube.com/watch?v={vid}" if vid else url
    title = ""
    duration_val: float | None = None

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(canonical_url, download=False)
        if isinstance(info, dict):
            title = (info.get("title") or "").strip()
            dur = info.get("duration")
            if isinstance(dur, (int, float)) and dur >= 0:
                duration_val = float(dur)
            if not vid and info.get("id"):
                vid = str(info["id"])
                canonical_url = f"https://www.youtube.com/watch?v={vid}"
    except Exception:
        pass

    # Lightweight oEmbed fallback if yt-dlp metadata lookup failed
    if not title and vid:
        try:
            session = _build_session(config)
            resp = session.get(
                "https://www.youtube.com/oembed",
                params={"url": canonical_url, "format": "json"},
                timeout=min(10, timeout),
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and data.get("title"):
                    title = str(data["title"]).strip()
        except Exception:
            pass

    if not title:
        title = f"YouTube_{vid}" if vid else clean_filename(url)

    return [
        ParsedMediaItem(
            source_type="youtube",
            url=canonical_url,
            source_id=vid,
            title=title,
            seq_num=1,
            duration_seconds=duration_val,
        )
    ]


def _flatten_input_urls(urls: str | Iterable[str]) -> list[str]:
    """Normalize raw URL list or multiline text into individual URL strings."""
    if isinstance(urls, str):
        raw_lines = urls.splitlines()
    else:
        raw_lines = []
        for item in urls:
            if item is None:
                continue
            raw_lines.extend(str(item).splitlines())

    result: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # If line contains multiple URLs or descriptive prefix, extract embedded http(s) URLs
        embedded = _URL_EXTRACT_RE.findall(stripped)
        if embedded:
            for u in embedded:
                result.append(u.rstrip(",;.)"))
        else:
            result.append(stripped)
    return result


def expand_urls(
    urls: list[str] | str,
    config: AppConfig | None = None,
    fetch_metadata: bool = True,
    include_playlist: bool = False,
) -> list[ParsedMediaItem]:
    """Detect, expand, normalize, and deduplicate YouTube and Google Drive URLs.

    Supports:
    - YouTube single videos: ``watch?v=``, ``youtu.be/``, ``shorts/``, ``live/``, ``embed/``
    - YouTube playlists: ``playlist?list=`` (and ``watch?v=...&list=`` when ``include_playlist=True``)
    - Google Drive single files: ``/file/d/<ID>``, ``open?id=<ID>``, ``uc?id=<ID>``
    - Google Drive shared folders: ``/drive/folders/<ID>`` (public HTML parsing, no OAuth)

    Deduplicates items by ``(source_type, source_id)`` preserving first-seen order
    and assigns 1-based sequential ``seq_num``.
    """
    raw_urls = _flatten_input_urls(urls)
    if not raw_urls:
        return []

    expanded: list[ParsedMediaItem] = []

    for raw_url in raw_urls:
        kind, item_id = detect_url_type(raw_url, include_playlist=include_playlist)

        if kind == "youtube_video":
            canonical = f"https://www.youtube.com/watch?v={item_id}"
            if fetch_metadata:
                expanded.extend(
                    _fetch_youtube_metadata(
                        canonical, video_id=item_id, is_playlist=False, config=config
                    )
                )
            else:
                expanded.append(
                    ParsedMediaItem(
                        source_type="youtube",
                        url=canonical,
                        source_id=item_id,
                        title=f"YouTube_{item_id}",
                        seq_num=1,
                        duration_seconds=None,
                    )
                )

        elif kind == "youtube_playlist":
            playlist_url = f"https://www.youtube.com/playlist?list={item_id}"
            expanded.extend(
                _fetch_youtube_metadata(
                    playlist_url, is_playlist=True, config=config
                )
            )

        elif kind == "gdrive_file":
            canonical = f"https://drive.google.com/file/d/{item_id}/view"
            title = (
                _fetch_gdrive_file_title(item_id, config=config)
                if fetch_metadata
                else f"Drive_{item_id}"
            )
            expanded.append(
                ParsedMediaItem(
                    source_type="gdrive",
                    url=canonical,
                    source_id=item_id,
                    title=title,
                    seq_num=1,
                    duration_seconds=None,
                )
            )

        elif kind == "gdrive_folder":
            expanded.extend(_fetch_gdrive_folder(item_id, config=config))

    # Deduplicate by (source_type, source_id) preserving order
    deduped: list[ParsedMediaItem] = []
    seen: set[tuple[str, str]] = set()
    for item in expanded:
        key = (item.source_type, item.source_id)
        if key in seen:
            continue
        seen.add(key)
        item.seq_num = len(deduped) + 1
        deduped.append(item)

    return deduped
