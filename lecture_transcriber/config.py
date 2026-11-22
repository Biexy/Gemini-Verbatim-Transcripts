"""Configuration management and cross-platform filename/stdio utilities."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PACKAGE_DIR / "data"
DEFAULT_CONFIG_PATH = DEFAULT_DATA_DIR / "config.json"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "state.db"
DEFAULT_TEMP_DIR = str(DEFAULT_DATA_DIR / "temp")
DEFAULT_OUTPUT_BASE_DIR = str(Path.home() / "Desktop" / "YouTube Transcripts" / "Transcripts")

DEFAULT_MODELS: list[str] = [
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

VALID_DOWNLOAD_MODES: tuple[str, ...] = ("eco", "balanced", "strict_audio")

MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mkv",
        ".webm",
        ".m4a",
        ".mp3",
        ".wav",
        ".ogg",
        ".oga",
        ".opus",
        ".aac",
        ".flac",
        ".avi",
        ".mov",
        ".flv",
        ".ts",
        ".m4v",
        ".3gp",
        ".wma",
        ".mpg",
        ".mpeg",
    }
)


def strip_media_extension(name: str) -> str:
    """Strip a known trailing media extension (e.g., '.mp4', '.m4a') from a lecture title."""
    text = (name or "").strip()
    if not text:
        return ""
    stem, ext = os.path.splitext(text)
    if ext.lower() in MEDIA_EXTENSIONS and stem.strip():
        return stem.strip()
    return text

# Windows illegal path characters and ASCII control characters (0x00-0x1F, 0x7F)
_ILLEGAL_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_WHITESPACE_RE = re.compile(r"\s+")
_MULTI_UNDERSCORE_RE = re.compile(r"_+")

# Windows reserved device names (case-insensitive, bare or with extension)
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)


def ensure_utf8_stdio() -> None:
    """Safely configure stdout/stderr for UTF-8 output (especially on Windows consoles)."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                # Stream may be redirected, closed, or non-reconfigurable in tests/services.
                pass


def clean_filename(name: str, max_length: int = 120) -> str:
    """Sanitize a string into an Arabic-safe, Windows-safe filename stem.

    - Preserves Arabic letters, diacritics, Latin letters, digits, and safe punctuation.
    - Removes Windows-illegal characters (< > : " / \\ | ? *) and control characters.
    - Guards against Windows reserved device names (CON, PRN, AUX, NUL, COM1-9, LPT1-9).
    - Strips leading/trailing spaces and dots, and enforces ``max_length``.
    """
    if not name or not isinstance(name, str):
        return "untitled"

    text = unicodedata.normalize("NFC", name)
    # Remove bidirectional formatting / zero-width chars that can confuse Windows Explorer
    text = "".join(
        ch for ch in text if unicodedata.category(ch) not in ("Cf", "Cc", "Cs")
    )
    text = _ILLEGAL_FILENAME_RE.sub("_", text)
    text = _WHITESPACE_RE.sub("_", text.strip())
    text = _MULTI_UNDERSCORE_RE.sub("_", text)
    # Windows does not allow trailing dots or spaces; also tidy leading/trailing underscores
    text = text.strip(" ._-")

    if not text:
        return "untitled"

    # Guard against Windows reserved names (check base stem before first dot)
    base_stem = text.split(".", 1)[0].upper()
    if base_stem in _WINDOWS_RESERVED_NAMES:
        text = f"_{text}"

    if max_length > 0 and len(text) > max_length:
        text = text[:max_length].rstrip(" ._-")
        if not text:
            return "untitled"
        # Re-check reserved names after truncation
        if text.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
            text = f"_{text}"[:max_length].rstrip(" ._-") or "untitled"

    return text


def _mask_key(key: str) -> str:
    """Return a safe masked representation of an API key."""
    key = (key or "").strip()
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"


def _coerce_bool(value: Any, default: bool = False) -> bool:
    """Safely coerce booleans, ints, and string flags ('true'/'false'/'1'/'0')."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


@dataclass
class AppConfig:
    """Application configuration with sane defaults for unstable/slow networks."""

    gemini_api_key: str = ""
    output_base_dir: str = DEFAULT_OUTPUT_BASE_DIR
    temp_dir: str = DEFAULT_TEMP_DIR
    download_workers: int = 3
    transcribe_workers: int = 1
    audio_bitrate: str = "64k"
    download_mode: str = "eco"
    keep_audio: bool = False
    max_chunk_minutes: int = 12
    socket_timeout: int = 30
    max_retries: int = 8
    models: list[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    port: int = 8765
    proxy: str = ""

    def __post_init__(self) -> None:
        self.gemini_api_key = (self.gemini_api_key or "").strip()
        if not self.gemini_api_key:
            self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()

        self.output_base_dir = (self.output_base_dir or DEFAULT_OUTPUT_BASE_DIR).strip()
        self.temp_dir = (self.temp_dir or DEFAULT_TEMP_DIR).strip()
        self.download_workers = max(1, int(self.download_workers))
        self.transcribe_workers = max(1, int(self.transcribe_workers))
        self.audio_bitrate = (self.audio_bitrate or "64k").strip()
        mode = str(self.download_mode or "eco").strip().lower()
        self.download_mode = mode if mode in VALID_DOWNLOAD_MODES else "eco"
        self.keep_audio = _coerce_bool(self.keep_audio, default=False)
        self.max_chunk_minutes = max(1, int(self.max_chunk_minutes))
        self.socket_timeout = max(5, int(self.socket_timeout))
        self.max_retries = max(0, int(self.max_retries))
        self.port = max(1, min(65535, int(self.port)))
        self.proxy = (self.proxy or "").strip()

        if isinstance(self.models, str):
            self.models = [m.strip() for m in re.split(r"[,\r\n]+", self.models) if m.strip()]
        elif isinstance(self.models, (list, tuple)):
            self.models = [str(m).strip() for m in self.models if str(m).strip()]
        else:
            self.models = list(DEFAULT_MODELS)
        if not self.models:
            self.models = list(DEFAULT_MODELS)

    def effective_api_key(self) -> str:
        """Return configured API key, falling back to the GEMINI_API_KEY environment variable."""
        return self.gemini_api_key.strip() or os.environ.get("GEMINI_API_KEY", "").strip()

    def effective_audio_bitrate(self) -> str:
        """Return effective FFmpeg audio bitrate based on download_mode and audio_bitrate."""
        br = (self.audio_bitrate or "64k").strip()
        if self.download_mode in ("eco", "strict_audio") and br == "64k":
            return "32k"
        return br

    def to_dict(self, mask_api_key: bool = False) -> dict[str, Any]:
        """Serialize configuration to a plain JSON-safe dictionary."""
        data = asdict(self)
        if mask_api_key:
            effective = self.effective_api_key()
            data["gemini_api_key"] = _mask_key(effective)
            data["has_api_key"] = bool(effective)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AppConfig:
        """Build an AppConfig from a dict, safely ignoring unknown fields."""
        if not isinstance(data, dict):
            return cls()
        allowed = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in allowed and v is not None}
        return cls(**filtered)


class ConfigManager:
    """Thread-safe JSON config loader and saver."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._lock = threading.RLock()
        self._config: AppConfig = self.load()

    @property
    def config(self) -> AppConfig:
        """Return a snapshot copy of the current configuration."""
        return self.get()

    def get(self) -> AppConfig:
        """Return a thread-safe snapshot copy of the current AppConfig."""
        with self._lock:
            return AppConfig.from_dict(asdict(self._config))

    def load(self) -> AppConfig:
        """Load configuration from disk, falling back cleanly to defaults."""
        with self._lock:
            raw: dict[str, Any] = {}
            if self.config_path.exists():
                try:
                    content = self.config_path.read_text(encoding="utf-8")
                    loaded = json.loads(content)
                    if isinstance(loaded, dict):
                        raw = loaded
                except Exception:
                    raw = {}
            self._config = AppConfig.from_dict(raw)
            return self.get()

    def save(self, config: AppConfig | None = None) -> AppConfig:
        """Atomically persist configuration to disk as UTF-8 JSON."""
        with self._lock:
            if config is not None:
                self._config = AppConfig.from_dict(asdict(config))
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.config_path.with_suffix(".json.tmp")
            payload = json.dumps(asdict(self._config), ensure_ascii=False, indent=2)
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, self.config_path)
            return self.get()

    def update(self, **kwargs: Any) -> AppConfig:
        """Update one or more fields on the config, validate, and persist to disk."""
        with self._lock:
            current = asdict(self._config)
            allowed = set(AppConfig.__dataclass_fields__.keys())
            for key, value in kwargs.items():
                if key not in allowed:
                    raise ValueError(f"Unknown config field: {key}")
                if value is None:
                    continue
                # Preserve existing API key if UI sends a masked placeholder
                if key == "gemini_api_key" and isinstance(value, str) and "*" in value:
                    continue
                if key == "download_mode":
                    mode_val = str(value).strip().lower()
                    if mode_val not in VALID_DOWNLOAD_MODES:
                        raise ValueError(
                            f"Invalid download_mode: {value!r}. Expected one of {VALID_DOWNLOAD_MODES}"
                        )
                    value = mode_val
                elif key == "keep_audio":
                    value = _coerce_bool(value, default=False)
                current[key] = value
            self._config = AppConfig.from_dict(current)
            return self.save(self._config)

    def ensure_directories(self) -> None:
        """Create configured data, temp, and output directories if needed."""
        cfg = self.get()
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.temp_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.output_base_dir).mkdir(parents=True, exist_ok=True)
