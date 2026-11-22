"""Resilient Google Gemini audio transcription client with chunked timestamp stitching."""

from __future__ import annotations

import inspect
import mimetypes
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from lecture_transcriber.config import AppConfig, DEFAULT_MODELS
from lecture_transcriber.network import is_online, wait_for_internet
from lecture_transcriber.transcriber.postprocessor import shift_timestamps


class TranscriptionError(Exception):
    """Base exception for AI transcription failures."""


class TranscriptionCancelledError(TranscriptionError):
    """Raised when a transcription job is cancelled by the user."""


class TranscriptionPausedError(TranscriptionError):
    """Raised when a transcription job is paused by the user."""


class GeminiAuthError(TranscriptionError):
    """Raised when the Gemini API key is missing, invalid, or unauthorized."""


class GeminiQuotaError(TranscriptionError):
    """Raised when all fallback Gemini models fail due to quota / rate limits."""


ARABIC_VERBATIM_PROMPT = """أنت خبير محترف ومتقن في تفريغ المحاضرات الجامعية والعلمية والهندسية والطبية والشرعية (سواء باللغة العربية الخالصة أو العربية المختلطة بمصطلحات إنجليزية) بدقة متناهية.
المطلوب:
تفريغ هذا المقطع الصوتي كاملاً تفريغاً نصياً حرفياً (Full Verbatim) من الثانية الأولى حتى الثانية الأخيرة دون أي نقص.

تعليمات صارمة جداً:
1. الاكتمال التام (ممنوع التوقف المبكر أو التلخيص): فرّغ كل جملة وكل كلمة من بداية المقطع الصوتي إلى نهايته الكاملة. يُمنع منعاً باتاً التوقف في منتصف التسجيل أو حذف أي جزء أو اختصار الشرح أو تلخيصه.
2. المصطلحات الإنجليزية والتقنية والهندسية (مهم جداً): اكتب جميع المصطلحات الإنجليزية والاختصارات وأسماء المكونات والأجهزة والبرمجيات والبروتوكولات والعبارات الإنجليزية التي ينطقها المحاضر بالحروف الإنجليزية الصحيحة داخل النص العربي (مثل: Embedded System, Microcontroller, ADC, DAC, I2C, RS-232, CPU, RAM, ROM, Sensor, Actuator, Protocol, PIC16F84) — لا تقم أبداً بتعريب نطق الكلمات الإنجليزية أو كتابتها بحروف عربية!
3. الطوابع الزمنية المستمرة حتى النهاية: أدرج طابعاً زمنياً منتظماً بصيغة [MM:SS] في بداية كل فقرة (كل 30 إلى 60 ثانية تقريباً) واستمر في وضع الطوابع الزمنية دون انقطاع حتى نهاية الملف الصوتي بالكامل؛ لا تتوقف عن وضع الطوابع الزمنية بعد الدقائق الأولى، ولا تدمج أكثر من 90 ثانية من الكلام في فقرة واحدة، وافصل كل فقرة بسطر فارغ.
4. الدقة العلمية والشرعية: التزم بالدقة التامة في كتابة المصطلحات العلمية والشرعية والعقدية والآيات القرآنية الكريمة والأحاديث النبوية وأسماء الأعلام والكتب.
5. منع التكرار والمقدمات: تجنب تماماً تكرار الجمل أو الفقرات؛ وإذا كان هناك صمت طويل فانتقل مباشرة إلى الكلام التالي. ابدأ مباشرة بنص الكلام المفرغ دون أي مقدمات أو تعليقات أو علامات كود من قبلك.
"""

ENGLISH_VERBATIM_PROMPT = """You are an expert academic and technical lecture transcriber.
Task: Produce a complete, accurate, full-verbatim transcript of this audio recording from the very first second to the very last second.

Strict rules:
1. Complete verbatim coverage: Transcribe every word from start to finish with NO early stopping, NO skipping, NO summarization, NO omission, and NO paraphrasing.
2. Continuous timestamps to the end: Insert regular timestamps in [MM:SS] format at the start of every paragraph (roughly every 30-60 seconds) ALL THE WAY TO THE END of the audio, separated by blank lines. Never stop inserting timestamps partway through, and never lump more than 90 seconds into a single paragraph.
3. Preserve all technical, engineering, medical, academic, and proper-noun terminology accurately in proper English spelling.
4. Do NOT repeat sentences or enter repetition loops; skip prolonged silence cleanly.
5. Output ONLY the transcript text with no conversational intro, outro, or markdown code fences.
"""

AUTO_VERBATIM_PROMPT = """You are an expert multilingual academic and technical lecture transcriber.
Task: Produce a complete, full-verbatim transcript of this audio from start to finish in the spoken language(s) without translating or truncating.

Strict rules:
1. Complete verbatim coverage: Transcribe every word from start to finish with NO early stopping, NO skipping, NO summarization, or paraphrasing.
2. Continuous timestamps to the end: Insert regular timestamps in [MM:SS] format at the start of every paragraph (roughly every 30-60 seconds) ALL THE WAY TO THE END of the audio, separated by blank lines. Never stop inserting timestamps partway through, and never lump more than 90 seconds into one paragraph.
3. Bilingual technical clarity: In mixed Arabic/English lectures, write ALL spoken English technical terms, acronyms, and hardware/software names in proper English script (e.g., Microcontroller, Embedded System, ADC, I2C, CPU, RAM) rather than transliterating them into Arabic letters, while keeping Quranic/Islamic and Arabic terminology accurate.
4. Do NOT repeat phrases or enter hallucination loops; skip prolonged silence cleanly.
5. Output ONLY the transcript text with no conversational intro, outro, or markdown code fences.
"""


def build_transcription_prompt(
    language_hint: str = "ar",
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> str:
    """Return the high-accuracy verbatim prompt for a given language and chunk position."""
    lang = (language_hint or "ar").strip().lower()
    if lang.startswith("ar"):
        base = ARABIC_VERBATIM_PROMPT.strip()
        if total_chunks > 1:
            base += (
                f"\n\nملاحظة: هذا هو الجزء ({chunk_index + 1} من {total_chunks}) من المحاضرة. "
                "ابدأ الطوابع الزمنية لهذا الجزء من [00:00] بدقة متزامنة مع صوت هذا الملف فقط."
            )
    elif lang.startswith("en"):
        base = ENGLISH_VERBATIM_PROMPT.strip()
        if total_chunks > 1:
            base += (
                f"\n\nNote: This is part {chunk_index + 1} of {total_chunks} of the lecture. "
                "Start timestamps at [00:00] relative to this audio file only."
            )
    else:
        base = AUTO_VERBATIM_PROMPT.strip()
        if total_chunks > 1:
            base += (
                f"\n\nNote: Part {chunk_index + 1} of {total_chunks}. "
                "Keep timestamps relative to [00:00] of this audio file."
            )
    return base


def _get_state_name(file_obj: Any) -> str:
    state = getattr(file_obj, "state", None)
    if state is None:
        return "ACTIVE"
    name = getattr(state, "name", None)
    if name is not None:
        return str(name).upper()
    text = str(state).upper()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text


def _classify_gemini_error(exc: BaseException) -> str:
    """Classify a Gemini SDK / HTTP exception into: auth | not_found | rate_limit | transient | fatal."""
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = str(exc).lower()

    if status_code in (401, 403) or any(
        tok in msg
        for tok in (
            "api_key_invalid",
            "api key not valid",
            "invalid api key",
            "permission_denied",
            "unauthenticated",
            "401",
            "403 forbidden",
        )
    ):
        return "auth"

    if status_code == 404 or any(
        tok in msg for tok in ("404", "not_found", "is not found for api version", "model not found")
    ):
        return "not_found"

    if status_code == 429 or any(
        tok in msg
        for tok in (
            "429",
            "resource_exhausted",
            "quota",
            "rate limit",
            "too many requests",
        )
    ):
        return "rate_limit"

    if status_code in (408, 500, 502, 503, 504) or isinstance(
        exc, (ConnectionError, TimeoutError, OSError)
    ) or any(
        tok in msg
        for tok in (
            "503",
            "500",
            "502",
            "504",
            "unavailable",
            "overloaded",
            "deadline_exceeded",
            "internal",
            "timed out",
            "timeout",
            "connection",
            "remote",
            "reset by peer",
            "ssl",
            "eof",
            "dns",
            "getaddrinfo",
        )
    ):
        return "transient"

    return "transient"


def _interruptible_sleep(
    seconds: float,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> None:
    """Sleep in small slices while remaining immediately responsive to cancel/pause."""
    deadline = time.monotonic() + max(0.0, float(seconds))
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise TranscriptionCancelledError("Transcription cancelled")
        if pause_event is not None and pause_event.is_set():
            raise TranscriptionPausedError("Transcription paused")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        step = min(0.25, remaining)
        if cancel_event is not None:
            if cancel_event.wait(timeout=step):
                raise TranscriptionCancelledError("Transcription cancelled")
        else:
            time.sleep(step)


def _call_progress_cb(
    progress_cb: Callable[..., Any] | None,
    progress_pct: float,
    chunk_index: int,
    total_chunks: int,
    message: str,
) -> None:
    if progress_cb is None:
        return
    try:
        sig = inspect.signature(progress_cb)
        params = list(sig.parameters.values())
        positional = [
            p
            for p in params
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        if has_varargs or len(positional) >= 4:
            progress_cb(progress_pct, chunk_index, total_chunks, message)
        elif len(positional) == 3:
            progress_cb(progress_pct, chunk_index, total_chunks)
        elif len(positional) == 2:
            progress_cb(progress_pct, message)
        elif len(positional) == 1:
            progress_cb(progress_pct)
        else:
            progress_cb()
    except Exception:
        pass


def _call_status_cb(
    on_status: Callable[..., Any] | None,
    message: str,
    level: str = "info",
) -> None:
    if on_status is None:
        return
    try:
        sig = inspect.signature(on_status)
        params = list(sig.parameters.values())
        positional = [
            p
            for p in params
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
        if has_varargs or len(positional) >= 2:
            on_status(message, level)
        elif len(positional) == 1:
            on_status(message)
        else:
            on_status()
    except Exception:
        pass


_AUDIO_MIME_TYPES: dict[str, str] = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".webm": "audio/webm",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
}


def _resolve_audio_mime_type(path_obj: Path) -> str | None:
    """Return an explicit Gemini-compatible MIME type for audio files (fixes Windows registry gaps)."""
    ext = path_obj.suffix.lower()
    if ext in _AUDIO_MIME_TYPES:
        return _AUDIO_MIME_TYPES[ext]
    guessed, _ = mimetypes.guess_type(str(path_obj))
    if guessed and guessed.startswith("audio/"):
        return guessed
    return None


def _upload_supports_config(upload_fn: Any) -> bool:
    """Return True if ``client.files.upload`` accepts a ``config`` keyword argument."""
    try:
        sig = inspect.signature(upload_fn)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD or param.name == "config":
            return True
    return False


def _generate_supports_config(generate_fn: Any) -> bool:
    """Return True if ``client.models.generate_content`` accepts a ``config`` keyword argument."""
    try:
        sig = inspect.signature(generate_fn)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD or param.name == "config":
            return True
    return False


def _build_generation_config() -> Any:
    """Build deterministic generation config with 65,536 max output tokens."""
    try:
        from google.genai import types

        return types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=65536,
        )
    except Exception:
        return {"temperature": 0.0, "max_output_tokens": 65536}


_TS_ITEM_RE = re.compile(
    r"\[(\d{1,3}:\d{2}(?::\d{2})?)(?:\s*[-–—]\s*(\d{1,3}:\d{2}(?::\d{2})?))?\]"
)
_RETRY_IN_RE = re.compile(r"retry\s+in\s+(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_RETRY_DELAY_RE = re.compile(
    r"retry_?delay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s?",
    re.IGNORECASE,
)


def _parse_ts_seconds(ts: str) -> int:
    parts = [int(p) for p in ts.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return 0


def _extract_max_timestamp_seconds(text: str, chunk_duration_seconds: float) -> float:
    """Return the largest plausible timestamp (in seconds) within *text*."""
    if not text:
        return 0.0
    upper_bound = (
        float(chunk_duration_seconds) * 1.25
        if chunk_duration_seconds and chunk_duration_seconds > 0
        else float("inf")
    )
    max_sec = 0.0
    for match in _TS_ITEM_RE.finditer(text):
        for group_idx in (1, 2):
            raw = match.group(group_idx)
            if not raw:
                continue
            sec = float(_parse_ts_seconds(raw))
            if sec <= upper_bound and sec > max_sec:
                max_sec = sec
    return max_sec


def _extract_finish_reason(response: Any) -> str:
    """Safely extract candidate[0].finish_reason as a string."""
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""
    try:
        first = candidates[0]
    except (IndexError, TypeError):
        return ""
    reason = getattr(first, "finish_reason", None)
    if reason is None:
        return ""
    name = getattr(reason, "name", None)
    if name is not None:
        return str(name)
    return str(reason)


def _is_503_unavailable(exc: BaseException) -> bool:
    """Return True if an exception indicates Gemini 503/504 unavailability or high demand."""
    status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status_code in (503, 504):
        return True
    msg = str(exc).lower()
    return any(
        tok in msg
        for tok in ("503", "504", "unavailable", "high demand", "overloaded", "deadline_exceeded")
    )


def _is_daily_quota_error(exc: BaseException, parsed_retry: float | None = None) -> bool:
    """Return True when a 429 error reflects daily (RPD) quota exhaustion rather than RPM."""
    if parsed_retry is not None and parsed_retry >= 600.0:
        return True
    msg = str(exc or "").lower()
    return any(
        tok in msg
        for tok in (
            "perday",
            "per_day",
            "requestsperday",
            "daily",
            "per day",
        )
    )


def _parse_retry_delay_seconds(exc: BaseException) -> float | None:
    """Extract server-suggested retry delay in seconds from 429 error messages."""
    msg = str(exc or "")
    for pat in (_RETRY_IN_RE, _RETRY_DELAY_RE):
        m = pat.search(msg)
        if m:
            try:
                val = float(m.group(1))
                if val > 0:
                    return min(120.0, val)
            except (TypeError, ValueError):
                pass
    return None


class GeminiTranscriber:
    """Resilient Google Gemini audio transcriber supporting chunked multi-hour lectures.

    Features:
    - Network-aware file upload with retry and automatic offline wait.
    - Remote file ``PROCESSING`` state polling with timeout protection.
    - Multi-model fallback across ``config.models`` with 429 rate-limit backoff and 503 retries.
    - Guaranteed ``finally:`` cleanup of remote Gemini File API objects.
    - Automatic timestamp shifting per chunk so stitched transcripts are seamless.
    """

    def __init__(
        self,
        config: AppConfig | dict[str, Any] | None = None,
        *,
        client: Any = None,
        poll_interval: float = 1.5,
        poll_timeout: float = 300.0,
        base_backoff: float = 2.0,
        max_backoff: float = 60.0,
        network_check_interval: float = 3.0,
    ) -> None:
        if isinstance(config, AppConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = AppConfig.from_dict(config)
        elif config is None:
            self.config = AppConfig()
        else:
            # Duck-typed config object
            self.config = config  # type: ignore[assignment]

        self._client = client
        self._client_lock = threading.RLock()
        self.poll_interval = max(0.05, float(poll_interval))
        self.poll_timeout = max(5.0, float(poll_timeout))
        self.base_backoff = max(0.05, float(base_backoff))
        self.max_backoff = max(self.base_backoff, float(max_backoff))
        self.network_check_interval = max(0.05, float(network_check_interval))

    def _get_api_key(self) -> str:
        eff = getattr(self.config, "effective_api_key", None)
        if callable(eff):
            return str(eff() or "").strip()
        return str(getattr(self.config, "gemini_api_key", "") or "").strip()

    def _get_models(self) -> list[str]:
        models = getattr(self.config, "models", None)
        if isinstance(models, (list, tuple)) and models:
            cleaned = [str(m).strip() for m in models if str(m).strip()]
            if cleaned:
                return cleaned
        return list(DEFAULT_MODELS)

    def _get_max_retries(self) -> int:
        val = getattr(self.config, "max_retries", 8)
        try:
            return max(1, int(val))
        except (TypeError, ValueError):
            return 8

    def _get_client(self) -> Any:
        with self._client_lock:
            if self._client is not None:
                return self._client

            api_key = self._get_api_key()
            if not api_key:
                raise GeminiAuthError(
                    "مفتاح Gemini API غير مضبوط. يرجى إدخال المفتاح في الإعدادات أو عبر المتغير GEMINI_API_KEY. "
                    "(Missing Gemini API key.)"
                )

            from google import genai
            from google.genai import types

            try:
                self._client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(timeout=180000),
                )
            except TypeError:
                self._client = genai.Client(api_key=api_key)
            return self._client

    def _check_control(
        self,
        cancel_event: threading.Event | None,
        pause_event: threading.Event | None,
    ) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise TranscriptionCancelledError("Transcription cancelled")
        if pause_event is not None and pause_event.is_set():
            raise TranscriptionPausedError("Transcription paused")

    def _wait_if_offline(
        self,
        cancel_event: threading.Event | None,
        pause_event: threading.Event | None,
        on_status: Callable[..., Any] | None = None,
    ) -> None:
        """If offline, wait gracefully until connectivity returns or cancel/pause fires."""
        self._check_control(cancel_event, pause_event)
        if is_online():
            return

        _call_status_cb(
            on_status,
            "انقطع الاتصال بالإنترنت أثناء التفريغ — بانتظار عودة الاتصال...",
            "warning",
        )

        # Combine cancel and pause into a single stop_event for wait_for_internet
        combined_stop = threading.Event()

        def _watch_control() -> None:
            while not combined_stop.is_set():
                if (cancel_event is not None and cancel_event.is_set()) or (
                    pause_event is not None and pause_event.is_set()
                ):
                    combined_stop.set()
                    return
                combined_stop.wait(timeout=0.15)

        watcher = threading.Thread(target=_watch_control, daemon=True)
        watcher.start()
        try:
            reconnected = wait_for_internet(
                stop_event=combined_stop,
                check_interval=self.network_check_interval,
            )
        finally:
            combined_stop.set()
            watcher.join(timeout=0.5)

        self._check_control(cancel_event, pause_event)
        if reconnected:
            _call_status_cb(
                on_status,
                "عاد الاتصال بالإنترنت — استئناف التفريغ...",
                "info",
            )

    def _upload_file_with_retry(
        self,
        client: Any,
        audio_path: str,
        cancel_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
        on_status: Callable[..., Any] | None = None,
    ) -> Any:
        """Upload an audio chunk to Gemini File API with network-outage wait & retries."""
        path_obj = Path(audio_path)
        if not path_obj.is_file():
            raise TranscriptionError(f"Audio chunk file not found: {audio_path}")

        max_attempts = max(3, self._get_max_retries())
        last_exc: BaseException | None = None

        mime_type = _resolve_audio_mime_type(path_obj)
        supports_config = _upload_supports_config(getattr(client.files, "upload", None))

        for attempt in range(1, max_attempts + 1):
            self._check_control(cancel_event, pause_event)
            self._wait_if_offline(cancel_event, pause_event, on_status=on_status)

            try:
                if mime_type and supports_config:
                    return client.files.upload(
                        file=str(path_obj),
                        config={"mime_type": mime_type},
                    )
                return client.files.upload(file=str(path_obj))
            except (TranscriptionCancelledError, TranscriptionPausedError):
                raise
            except Exception as exc:
                last_exc = exc
                kind = _classify_gemini_error(exc)
                if kind == "auth":
                    raise GeminiAuthError(f"Gemini authentication failed during upload: {exc}") from exc

                if attempt >= max_attempts:
                    break

                if not is_online():
                    self._wait_if_offline(cancel_event, pause_event, on_status=on_status)

                delay = min(self.max_backoff, self.base_backoff * (2 ** (attempt - 1)))
                delay *= random.uniform(0.6, 1.0)
                _call_status_cb(
                    on_status,
                    f"فشل رفع الصوت (محاولة {attempt}/{max_attempts}) — إعادة المحاولة بعد {delay:.1f}ث: {exc}",
                    "warning",
                )
                _interruptible_sleep(delay, cancel_event, pause_event)

        raise TranscriptionError(f"Failed to upload audio file after {max_attempts} attempts: {last_exc}")

    def _wait_for_file_active(
        self,
        client: Any,
        uploaded_file: Any,
        cancel_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
        on_status: Callable[..., Any] | None = None,
    ) -> Any:
        """Poll Gemini remote file until state transitions out of PROCESSING."""
        started = time.monotonic()
        current = uploaded_file

        while True:
            self._check_control(cancel_event, pause_event)
            state_name = _get_state_name(current)

            if state_name == "ACTIVE":
                return current
            if state_name == "FAILED":
                err_info = getattr(current, "error", None)
                raise TranscriptionError(f"Gemini remote file processing failed: {err_info}")
            if state_name != "PROCESSING":
                # Unknown non-processing state (treat as ready if not FAILED)
                return current

            elapsed = time.monotonic() - started
            if elapsed >= self.poll_timeout:
                raise TranscriptionError(
                    f"Timed out after {self.poll_timeout:.0f}s waiting for Gemini file processing."
                )

            _interruptible_sleep(self.poll_interval, cancel_event, pause_event)

            # Fetch updated status with brief network resilience
            try:
                current = client.files.get(name=current.name)
            except Exception as exc:
                kind = _classify_gemini_error(exc)
                if kind == "auth":
                    raise GeminiAuthError(f"Gemini auth error while polling file: {exc}") from exc
                if not is_online():
                    self._wait_if_offline(cancel_event, pause_event, on_status=on_status)
                # Keep polling until poll_timeout expires
                _call_status_cb(
                    on_status,
                    f"تأخر مؤقت أثناء فحص حالة الملف على Gemini: {exc}",
                    "warning",
                )

    def _safe_delete_remote_file(self, client: Any, uploaded_file: Any) -> None:
        """Best-effort remote file deletion so Gemini File API quota never fills up."""
        name = getattr(uploaded_file, "name", None)
        if not name:
            return
        try:
            client.files.delete(name=name)
        except Exception:
            pass

    def _generate_with_fallback(
        self,
        client: Any,
        uploaded_file: Any,
        prompt: str,
        cancel_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
        on_status: Callable[..., Any] | None = None,
        chunk_duration_seconds: float = 0.0,
    ) -> str:
        """Invoke ``generate_content`` across fallback models with completeness guard & smart 429/503 handling."""
        models = self._get_models()
        unavailable_models: set[str] = set()
        incomplete_models: set[str] = set()
        max_rounds = max(2, min(6, (self._get_max_retries() + 1) // 2 + 1))

        generate_fn = getattr(client.models, "generate_content", None)
        supports_config = _generate_supports_config(generate_fn)
        gen_config = _build_generation_config() if supports_config else None

        last_error: BaseException | None = None
        saw_rate_limit = False
        best_partial_text = ""
        best_partial_score: tuple[float, int] = (-1.0, -1)

        duration_sec = max(0.0, float(chunk_duration_seconds or 0.0))

        for round_idx in range(1, max_rounds + 1):
            active_models = [
                m
                for m in models
                if m not in unavailable_models and m not in incomplete_models
            ]
            if not active_models:
                break

            round_retry_hints: list[float] = []

            for model_name in active_models:
                self._check_control(cancel_event, pause_event)
                self._wait_if_offline(cancel_event, pause_event, on_status=on_status)

                try:
                    if supports_config and gen_config is not None:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[uploaded_file, prompt],
                            config=gen_config,
                        )
                    else:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[uploaded_file, prompt],
                        )
                    text = ""
                    try:
                        text = (getattr(response, "text", None) or "").strip()
                    except Exception:
                        text = ""

                    if not text:
                        last_error = TranscriptionError(f"Empty response returned by model '{model_name}'")
                        _call_status_cb(
                            on_status,
                            f"استجابة فارغة من النموذج {model_name} — تجربة النموذج البديل...",
                            "warning",
                        )
                        _interruptible_sleep(min(1.0, self.base_backoff), cancel_event, pause_event)
                        continue

                    finish_reason = _extract_finish_reason(response)
                    hit_max_tokens = "MAX_TOKENS" in finish_reason.upper()

                    max_ts_sec = _extract_max_timestamp_seconds(text, duration_sec)
                    low_coverage = False
                    if duration_sec >= 180.0:
                        coverage_ratio = min(1.0, max_ts_sec / duration_sec)
                        if max_ts_sec < duration_sec * 0.72:
                            low_coverage = True
                    elif duration_sec > 0.0:
                        coverage_ratio = min(1.0, max_ts_sec / duration_sec)
                    else:
                        coverage_ratio = 0.0 if hit_max_tokens else 1.0

                    if hit_max_tokens or low_coverage:
                        candidate_score = (coverage_ratio, len(text))
                        if candidate_score > best_partial_score:
                            best_partial_score = candidate_score
                            best_partial_text = text

                        incomplete_models.add(model_name)
                        reasons: list[str] = []
                        if hit_max_tokens:
                            reasons.append(f"finish_reason={finish_reason}")
                        if low_coverage:
                            reasons.append(
                                f"تغطية الطوابع الزمنية {max_ts_sec:.0f}ث/{duration_sec:.0f}ث ({coverage_ratio * 100:.0f}%)"
                            )
                        reason_text = "، ".join(reasons)
                        last_error = TranscriptionError(
                            f"Incomplete transcript from '{model_name}' ({reason_text})"
                        )
                        _call_status_cb(
                            on_status,
                            f"تحذير: تفريغ غير مكتمل من {model_name} ({reason_text}) — تجربة النموذج البديل فوراً... "
                            f"(Incomplete/truncated output from {model_name}, trying next model)",
                            "warning",
                        )
                        continue

                    return text

                except (TranscriptionCancelledError, TranscriptionPausedError):
                    raise
                except Exception as exc:
                    last_error = exc
                    kind = _classify_gemini_error(exc)

                    if kind == "auth":
                        raise GeminiAuthError(f"Gemini authentication error: {exc}") from exc

                    if kind == "not_found":
                        unavailable_models.add(model_name)
                        _call_status_cb(
                            on_status,
                            f"النموذج {model_name} غير متاح — الانتقال للنموذج البديل...",
                            "warning",
                        )
                        continue

                    if kind == "rate_limit":
                        saw_rate_limit = True
                        parsed_retry = _parse_retry_delay_seconds(exc)
                        if _is_daily_quota_error(exc, parsed_retry):
                            # Daily (RPD) bucket exhausted on this model -> skip it permanently this call & advance immediately!
                            unavailable_models.add(model_name)
                            _call_status_cb(
                                on_status,
                                f"انتهت الحصة اليومية (RPD 429) للنموذج {model_name} — الانتقال فوراً للنموذج التالي...",
                                "warning",
                            )
                            continue

                        if parsed_retry is not None:
                            round_retry_hints.append(parsed_retry)
                        # Different Gemini models have independent Free-Tier buckets: switch fast within round!
                        switch_delay = min(self.max_backoff, min(0.25, self.base_backoff * 0.25))
                        hint_msg = f" (retry~{parsed_retry:.0f}s)" if parsed_retry is not None else ""
                        _call_status_cb(
                            on_status,
                            f"تجاوز حد الطلبات (429{hint_msg}) على {model_name} — التبديل للنموذج البديل...",
                            "warning",
                        )
                        if switch_delay > 0:
                            _interruptible_sleep(switch_delay, cancel_event, pause_event)
                        continue

                    if _is_503_unavailable(exc):
                        # Fast failover on 503 high demand so another ready model runs right away
                        fast_delay = (
                            min(0.5, self.base_backoff * 0.25)
                            if round_idx == 1
                            else min(1.5, self.base_backoff * 0.5)
                        )
                        fast_delay = min(self.max_backoff, fast_delay)
                        _call_status_cb(
                            on_status,
                            f"ضغط مرتفع (503) على {model_name} — الانتقال الفوري للنموذج البديل...",
                            "warning",
                        )
                        if fast_delay > 0:
                            _interruptible_sleep(fast_delay, cancel_event, pause_event)
                        continue

                    # Other transient error (500 / timeout / network)
                    if not is_online():
                        self._wait_if_offline(cancel_event, pause_event, on_status=on_status)

                    delay = min(self.max_backoff, self.base_backoff * (2 ** (round_idx - 1)))
                    delay *= random.uniform(0.6, 1.0)
                    _call_status_cb(
                        on_status,
                        f"خطأ مؤقت من {model_name} (جولة {round_idx}/{max_rounds}) — إعادة المحاولة بعد {delay:.1f}ث: {exc}",
                        "warning",
                    )
                    _interruptible_sleep(delay, cancel_event, pause_event)

            # Only sleep between rounds if eligible models still remain
            remaining_models = [
                m
                for m in models
                if m not in unavailable_models and m not in incomplete_models
            ]
            if not remaining_models:
                break

            if round_idx < max_rounds:
                round_delay = min(self.max_backoff, self.base_backoff * (2 ** round_idx)) * 0.5
                if round_retry_hints:
                    hinted = min(self.max_backoff, min(round_retry_hints))
                    round_delay = max(round_delay, hinted)
                if round_delay > 0:
                    _interruptible_sleep(round_delay, cancel_event, pause_event)

        if best_partial_text:
            _call_status_cb(
                on_status,
                "تحذير: تعذر الحصول على تفريغ مكتمل من جميع النماذج — سيتم استخدام أفضل تفريغ جزئي متاح "
                "(Exhausted all models; returning best partial transcript).",
                "warning",
            )
            return best_partial_text

        if saw_rate_limit:
            raise GeminiQuotaError(
                f"تم تجاوز حصة Gemini API على جميع النماذج الاحتياطية (429): {last_error}"
            )
        raise TranscriptionError(
            f"فشل تفريغ المقطع الصوتي عبر جميع نماذج Gemini ({', '.join(models)}): {last_error}"
        )

    def transcribe_single_file(
        self,
        audio_path: str,
        *,
        language_hint: str = "ar",
        offset_seconds: int = 0,
        chunk_index: int = 0,
        total_chunks: int = 1,
        chunk_duration_seconds: float = 0.0,
        cancel_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
        on_status: Callable[..., Any] | None = None,
    ) -> str:
        """Upload, poll, transcribe, shift timestamps, and ALWAYS delete remote file."""
        self._check_control(cancel_event, pause_event)
        client = self._get_client()
        uploaded_file: Any = None
        try:
            uploaded_file = self._upload_file_with_retry(
                client,
                audio_path,
                cancel_event=cancel_event,
                pause_event=pause_event,
                on_status=on_status,
            )
            uploaded_file = self._wait_for_file_active(
                client,
                uploaded_file,
                cancel_event=cancel_event,
                pause_event=pause_event,
                on_status=on_status,
            )
            prompt = build_transcription_prompt(
                language_hint=language_hint,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
            )
            raw_text = self._generate_with_fallback(
                client,
                uploaded_file,
                prompt,
                cancel_event=cancel_event,
                pause_event=pause_event,
                on_status=on_status,
                chunk_duration_seconds=chunk_duration_seconds,
            )
            if offset_seconds:
                raw_text = shift_timestamps(raw_text, int(round(offset_seconds)))
            return raw_text.strip()
        finally:
            if uploaded_file is not None:
                self._safe_delete_remote_file(client, uploaded_file)

    def transcribe_chunks(
        self,
        chunks: Sequence[dict[str, Any] | str],
        language_hint: str = "ar",
        cancel_event: threading.Event | None = None,
        progress_cb: Callable[..., Any] | None = None,
        pause_event: threading.Event | None = None,
        on_status: Callable[..., Any] | None = None,
    ) -> str:
        """Transcribe an ordered list of audio chunks and stitch them with continuous timestamps.

        Args:
            chunks: List of chunk dicts ``{"path": str, "offset_seconds": int, "duration_seconds": float}``
                (or plain file paths, treated as offset 0).
            language_hint: Language code hint (default ``"ar"``).
            cancel_event: Optional ``threading.Event`` to abort transcription immediately.
            progress_cb: Optional progress callback receiving ``(pct, chunk_index, total_chunks, message)``
                (auto-adapts to 1/2/3/4-arg callables).
            pause_event: Optional ``threading.Event`` to pause cleanly between/during operations.
            on_status: Optional status/log callback ``(message, level)``.

        Returns:
            Continuous raw transcript text stitched across all chunks with shifted timestamps.
        """
        chunk_list = list(chunks or [])
        if not chunk_list:
            raise TranscriptionError("No audio chunks provided for transcription.")

        total = len(chunk_list)
        stitched_parts: list[str] = []

        for idx, chunk in enumerate(chunk_list):
            self._check_control(cancel_event, pause_event)

            if isinstance(chunk, dict):
                chunk_path = str(chunk.get("path", ""))
                offset_sec = int(round(float(chunk.get("offset_seconds", 0) or 0)))
                chunk_duration_sec = float(chunk.get("duration_seconds", 0.0) or 0.0)
            else:
                chunk_path = str(chunk)
                offset_sec = 0
                chunk_duration_sec = 0.0

            start_pct = (idx / total) * 100.0
            _call_progress_cb(
                progress_cb,
                start_pct,
                idx + 1,
                total,
                f"جاري تفريغ الجزء {idx + 1} من {total}...",
            )

            part_text = self.transcribe_single_file(
                chunk_path,
                language_hint=language_hint,
                offset_seconds=offset_sec,
                chunk_index=idx,
                total_chunks=total,
                chunk_duration_seconds=chunk_duration_sec,
                cancel_event=cancel_event,
                pause_event=pause_event,
                on_status=on_status,
            )
            if part_text:
                stitched_parts.append(part_text)

            done_pct = ((idx + 1) / total) * 100.0
            _call_progress_cb(
                progress_cb,
                done_pct,
                idx + 1,
                total,
                f"اكتمل تفريغ الجزء {idx + 1} من {total}",
            )

        return "\n\n".join(stitched_parts).strip()
