"""AI transcription and post-processing subpackage."""

from .gemini_client import (
    ARABIC_VERBATIM_PROMPT,
    AUTO_VERBATIM_PROMPT,
    ENGLISH_VERBATIM_PROMPT,
    GeminiAuthError,
    GeminiQuotaError,
    GeminiTranscriber,
    TranscriptionCancelledError,
    TranscriptionError,
    TranscriptionPausedError,
    build_transcription_prompt,
)
from .postprocessor import (
    clean_transcript_body,
    format_lecture_markdown,
    rebuild_course_artifacts,
    shift_timestamps,
)

__all__ = [
    "ARABIC_VERBATIM_PROMPT",
    "AUTO_VERBATIM_PROMPT",
    "ENGLISH_VERBATIM_PROMPT",
    "GeminiAuthError",
    "GeminiQuotaError",
    "GeminiTranscriber",
    "TranscriptionCancelledError",
    "TranscriptionError",
    "TranscriptionPausedError",
    "build_transcription_prompt",
    "clean_transcript_body",
    "format_lecture_markdown",
    "rebuild_course_artifacts",
    "shift_timestamps",
]
