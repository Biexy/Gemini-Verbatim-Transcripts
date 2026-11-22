"""Lightweight, resilient lecture audio downloader & Arabic AI transcriber."""

from .config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DATA_DIR,
    DEFAULT_DB_PATH,
    DEFAULT_MODELS,
    DEFAULT_OUTPUT_BASE_DIR,
    DEFAULT_TEMP_DIR,
    VALID_DOWNLOAD_MODES,
    AppConfig,
    ConfigManager,
    clean_filename,
    ensure_utf8_stdio,
)
from .database import ACTIVE_STATUSES, VALID_STATUSES, Database
from .network import (
    DEFAULT_CHECK_HOSTS,
    NetworkStatusMonitor,
    call_with_retry,
    is_online,
    retry_with_backoff,
    wait_for_internet,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DATA_DIR",
    "DEFAULT_DB_PATH",
    "DEFAULT_MODELS",
    "DEFAULT_OUTPUT_BASE_DIR",
    "DEFAULT_TEMP_DIR",
    "VALID_DOWNLOAD_MODES",
    "AppConfig",
    "ConfigManager",
    "clean_filename",
    "ensure_utf8_stdio",
    "ACTIVE_STATUSES",
    "VALID_STATUSES",
    "Database",
    "DEFAULT_CHECK_HOSTS",
    "NetworkStatusMonitor",
    "call_with_retry",
    "is_online",
    "retry_with_backoff",
    "wait_for_internet",
]
