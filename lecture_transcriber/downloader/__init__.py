"""Unified download and audio processing interface for YouTube and Google Drive."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from ..config import AppConfig
from .audio_processor import (
    check_ffmpeg,
    cleanup_audio_files,
    merge_audio_chunks_to_mp3,
    prepare_audio_for_transcription,
    probe_audio_info,
)
from .gdrive_dl import (
    detect_extension_from_magic,
    download_gdrive_file,
    parse_gdrive_confirm_url,
)
from .url_parser import (
    AudioProcessingError,
    DownloadCancelledError,
    DownloadError,
    DownloadPausedError,
    GDriveError,
    GDrivePermissionError,
    GDriveQuotaError,
    ParsedMediaItem,
    URLParseError,
    detect_url_type,
    expand_urls,
    extract_gdrive_file_id,
    extract_gdrive_folder_id,
    extract_youtube_id,
    extract_youtube_playlist_id,
    parse_gdrive_folder_html,
)
from .youtube_dl import (
    BALANCED_AUDIO_FORMAT,
    ECO_AUDIO_FORMAT,
    FALLBACK_AUDIO_FORMAT,
    PRIMARY_AUDIO_FORMAT,
    STRICT_AUDIO_FORMAT,
    download_youtube_audio,
)


def download_media(
    source_type: str,
    url: str,
    source_id: str,
    temp_dir: str | Path,
    config: AppConfig | None = None,
    progress_cb: Callable[[float, float, float | None, int, int], Any] | None = None,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
    on_status: Callable[..., Any] | None = None,
) -> str:
    """Dispatch media download to YouTube or Google Drive handler based on ``source_type``."""
    src = (source_type or "").strip().lower()
    if src == "youtube":
        return download_youtube_audio(
            url=url,
            source_id=source_id,
            temp_dir=temp_dir,
            config=config,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
            pause_event=pause_event,
            on_status=on_status,
        )
    if src == "gdrive":
        return download_gdrive_file(
            url=url,
            source_id=source_id,
            temp_dir=temp_dir,
            config=config,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
            pause_event=pause_event,
            on_status=on_status,
        )
    raise URLParseError(
        f"نوع مصدر غير مدعوم / Unsupported source_type: {source_type!r}"
    )


__all__ = [
    # Data models & URL parsing
    "ParsedMediaItem",
    "expand_urls",
    "detect_url_type",
    "extract_youtube_id",
    "extract_youtube_playlist_id",
    "extract_gdrive_file_id",
    "extract_gdrive_folder_id",
    "parse_gdrive_folder_html",
    # Downloaders
    "download_media",
    "download_youtube_audio",
    "download_gdrive_file",
    "parse_gdrive_confirm_url",
    "detect_extension_from_magic",
    "PRIMARY_AUDIO_FORMAT",
    "FALLBACK_AUDIO_FORMAT",
    "ECO_AUDIO_FORMAT",
    "BALANCED_AUDIO_FORMAT",
    "STRICT_AUDIO_FORMAT",
    # Audio processing
    "check_ffmpeg",
    "probe_audio_info",
    "prepare_audio_for_transcription",
    "merge_audio_chunks_to_mp3",
    "cleanup_audio_files",
    # Exceptions
    "DownloadError",
    "DownloadPausedError",
    "DownloadCancelledError",
    "URLParseError",
    "GDriveError",
    "GDrivePermissionError",
    "GDriveQuotaError",
    "AudioProcessingError",
]
