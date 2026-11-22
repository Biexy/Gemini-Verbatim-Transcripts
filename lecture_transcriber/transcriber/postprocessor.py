"""Post-processing utilities for Arabic/academic lecture transcripts.

Provides:
- Timestamp shifting across chunked audio transcripts (`shift_timestamps`)
- Transcript cleaning & multi-paragraph AI hallucination loop elimination (`clean_transcript_body`)
- Structured NotebookLM-ready Markdown formatting (`format_lecture_markdown`)
- Course-level artifact generation (`rebuild_course_artifacts`):
  - `_Summary_Report.md`
  - `000_Complete_<clean_course_name>_NotebookLM.md`
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Iterable, Mapping
from urllib.parse import quote

try:
    from lecture_transcriber.config import (
        clean_filename as _clean_filename,
        strip_media_extension as _strip_media_extension,
    )
except Exception:
    _clean_filename = None
    _strip_media_extension = None


def _safe_clean_filename(name: str, max_length: int = 120) -> str:
    """Fallback Windows-safe Arabic filename sanitizer if config is unavailable."""
    if _clean_filename is not None:
        return _clean_filename(name)
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip(" ._")
    if not cleaned:
        cleaned = "untitled"
    return cleaned[:max_length]


# Matches [MM:SS], [HH:MM:SS], and ranges like [MM:SS - MM:SS] or [HH:MM:SS - HH:MM:SS]
_TS_TOKEN = r"\d{1,3}:\d{2}(?::\d{2})?"
_BRACKETED_TS_RANGE_RE = re.compile(
    rf"\[({_TS_TOKEN})(?:\s*[-–—]\s*({_TS_TOKEN}))?\]"
)
_BRACKETED_TS_RE = re.compile(rf"\[{_TS_TOKEN}(?:\s*[-–—]\s*{_TS_TOKEN})?\]")


def _parse_ts_to_seconds(ts: str) -> int:
    parts = [int(p) for p in ts.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return 0


def _format_seconds_ts(total_seconds: int, force_hours: bool = False) -> str:
    total_seconds = max(0, int(total_seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0 or force_hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def shift_timestamps(text: str, offset_seconds: int) -> str:
    """Shift all bracketed `[MM:SS]` and `[HH:MM:SS]` timestamps in *text* by *offset_seconds*.

    Supports single timestamps (`[00:10]`, `[01:05:10]`) as well as ranges (`[00:10 - 00:45]`).
    Automatically rolls minutes over into `[HH:MM:SS]` when the shifted value reaches >= 3600s.
    """
    if not text or not offset_seconds:
        return text or ""

    offset = int(round(offset_seconds))
    if offset == 0:
        return text

    def _replace(match: re.Match[str]) -> str:
        start_raw = match.group(1)
        end_raw = match.group(2)

        start_sec = max(0, _parse_ts_to_seconds(start_raw) + offset)
        start_force_h = start_raw.count(":") == 2 or start_sec >= 3600
        start_fmt = _format_seconds_ts(start_sec, force_hours=start_force_h)

        if end_raw:
            end_sec = max(0, _parse_ts_to_seconds(end_raw) + offset)
            end_force_h = end_raw.count(":") == 2 or end_sec >= 3600 or start_force_h
            end_fmt = _format_seconds_ts(end_sec, force_hours=end_force_h)
            return f"[{start_fmt} - {end_fmt}]"

        return f"[{start_fmt}]"

    return _BRACKETED_TS_RANGE_RE.sub(_replace, text)


# Patterns for conversational AI intros / outros that Gemini occasionally prepends/appends
_INTRO_LINE_PATTERNS = [
    re.compile(
        r"^(?:بالتأكيد|بالطبع|حسناً|حسنا|نعم|أهلاً|تفضل|إليك|فيما يلي|هذا هو|ها هو|إليكم)\b.*(?:تفريغ|النص|المقطع|المحاضرة|التسجيل|الصوتي)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:تفريغ|النص المفرغ|التفريغ النصي|التفريغ الحرفي)\s*[:：\-]?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:sure|certainly|of course|here(?:'s| is)|below is|the following is)\b.*(?:transcript|transcription|verbatim|audio|lecture)",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:transcript|transcription|verbatim transcript)\s*[:：\-]?\s*$", re.IGNORECASE),
]

_OUTRO_LINE_PATTERNS = [
    re.compile(
        r"^(?:آمل|أتمنى|أرجو|نتمنى)\s+(?:أن يكون|لك|لكم).*(?:مفيد|التوفيق|ساعد)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:هل تريد|إذا أردت|إذا كنت ترغب|لا تتردد|إن احتجت).*(?:تلخيص|تعديل|مساعدة|شيء آخر|توضيح)",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:انتهى التفريغ|نهاية التفريغ|تم التفريغ|نهاية المقطع|انتهى المقطع)\b.*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:i hope this helps|let me know if|feel free to|end of transcript|transcription complete)\b.*$",
        re.IGNORECASE,
    ),
]

# Arabic diacritics (tashkeel) and tatweel for normalization during loop comparison
_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670\u0640]")
_NON_WORD_RE = re.compile(r"[^\w\u0600-\u06ff\s]+")


def _strip_code_fences(text: str) -> str:
    """Remove wrapping or stray markdown code fences (``` or ```markdown)."""
    text = text.strip()
    # If the entire response is wrapped in a single fenced block, unwrap it
    wrapped = re.match(r"^```[a-zA-Z0-9_-]*\s*\n(.*?)\n```\s*$", text, flags=re.DOTALL)
    if wrapped:
        text = wrapped.group(1)

    # Also drop any standalone fence lines that may remain
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*```[a-zA-Z0-9_-]*\s*$", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _strip_ai_chat_remarks(text: str) -> str:
    """Strip conversational AI intro/outro lines without touching actual transcript lines."""
    lines = text.splitlines()

    # Strip leading blank lines & conversational intro lines (up to first 5 non-empty lines)
    checked = 0
    while lines and checked < 6:
        stripped = lines[0].strip()
        if not stripped:
            lines.pop(0)
            continue
        # Never strip a line that contains a timestamp
        if _BRACKETED_TS_RE.search(stripped):
            break
        cleaned_heading = stripped.lstrip("#*>- ").rstrip(":：* ").strip()
        if any(p.search(stripped) or p.search(cleaned_heading) for p in _INTRO_LINE_PATTERNS):
            lines.pop(0)
            checked += 1
            continue
        # Also strip a standalone horizontal rule immediately following an intro
        if checked > 0 and re.match(r"^[-_*]{3,}$", stripped):
            lines.pop(0)
            checked += 1
            continue
        break

    # Strip trailing blank lines & conversational outro lines
    checked_tail = 0
    while lines and checked_tail < 6:
        stripped = lines[-1].strip()
        if not stripped:
            lines.pop()
            continue
        if _BRACKETED_TS_RE.search(stripped):
            break
        cleaned_tail = stripped.lstrip("#*>- ").rstrip(".*!؟? ").strip()
        if any(p.search(stripped) or p.search(cleaned_tail) for p in _OUTRO_LINE_PATTERNS):
            lines.pop()
            checked_tail += 1
            continue
        if checked_tail > 0 and re.match(r"^[-_*]{3,}$", stripped):
            lines.pop()
            checked_tail += 1
            continue
        break

    return "\n".join(lines).strip()


def _separate_inline_timestamps(text: str) -> str:
    """Ensure timestamps packed mid-line start their own clean paragraphs."""
    # Insert a paragraph break before any bracketed timestamp preceded by non-newline text
    # e.g., "[00:00] كلام أول [00:30] كلام ثاني" -> "[00:00] كلام أول\n\n[00:30] كلام ثاني"
    out = re.sub(
        rf"([^\n])([ \t]*)(\[{_TS_TOKEN}(?:\s*[-–—]\s*{_TS_TOKEN})?\])",
        lambda m: f"{m.group(1).rstrip()}\n\n{m.group(3)}",
        text,
    )
    # Normalize spaces right after a timestamp at the start of a line
    out = re.sub(
        rf"(^|\n)(\[{_TS_TOKEN}(?:\s*[-–—]\s*{_TS_TOKEN})?\])[ \t]*",
        r"\1\2 ",
        out,
    )
    # Also ensure consecutive single-newline timestamp lines become clean paragraphs
    out = re.sub(
        rf"\n(\[{_TS_TOKEN}(?:\s*[-–—]\s*{_TS_TOKEN})?\])",
        r"\n\n\1",
        out,
    )
    # Collapse 3+ newlines into double newline
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _normalize_for_compare(paragraph: str) -> str:
    """Normalize paragraph text for hallucination/loop comparison (ignores timestamps & diacritics)."""
    no_ts = _BRACKETED_TS_RE.sub(" ", paragraph)
    no_diac = _DIACRITICS_RE.sub("", no_ts)
    # Unify Arabic alef / teh marbuta / alef maksura variants lightly
    no_diac = (
        no_diac.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )
    no_punct = _NON_WORD_RE.sub(" ", no_diac)
    return re.sub(r"\s+", " ", no_punct).strip().lower()


def _paragraphs_similar(norm_a: str, norm_b: str) -> bool:
    """Return True if two normalized paragraphs are identical or near-identical."""
    if not norm_a or not norm_b:
        return norm_a == norm_b
    if norm_a == norm_b:
        return True
    # Different numbers indicate distinct enumerated items, not a hallucination loop
    if re.findall(r"\d+", norm_a) != re.findall(r"\d+", norm_b):
        return False
    # Only use fuzzy matching for longer strings to avoid false positives on short/similar sentences
    min_len = min(len(norm_a), len(norm_b))
    max_len = max(len(norm_a), len(norm_b))
    if min_len < 30:
        return False
    if min_len / max_len < 0.85:
        return False
    return SequenceMatcher(None, norm_a, norm_b).ratio() >= 0.93


def _dedupe_intra_paragraph_loops(paragraph: str) -> str:
    """Collapse runaway word/phrase repetition loops inside a single paragraph."""
    # Separate optional leading timestamp so we don't disturb it
    ts_prefix = ""
    body = paragraph
    m = re.match(rf"^(\[{_TS_TOKEN}(?:\s*[-–—]\s*{_TS_TOKEN})?\]\s*)(.*)$", paragraph, flags=re.DOTALL)
    if m:
        ts_prefix = m.group(1)
        body = m.group(2)

    words = body.split()
    if len(words) < 6:
        return paragraph

    norm_words = [_normalize_for_compare(w) for w in words]

    result_words: list[str] = []
    i = 0
    n = len(words)
    while i < n:
        collapsed = False
        max_ngram = min(30, (n - i) // 3)
        for k in range(1, max_ngram + 1):
            pattern = norm_words[i : i + k]
            if not any(pattern):
                continue
            # Count consecutive repeats of this k-word pattern
            reps = 1
            pos = i + k
            while pos + k <= n and norm_words[pos : pos + k] == pattern:
                reps += 1
                pos += k
            # Short 1-2 word phrases need >= 4 consecutive repeats to be treated as a loop;
            # longer phrases (>= 3 words) need >= 3 consecutive repeats.
            min_reps = 4 if k <= 2 else 3
            if reps >= min_reps:
                result_words.extend(words[i : i + k])
                # Also skip any partial trailing match of the same pattern
                rem_pos = pos
                for offset in range(k):
                    if rem_pos < n and norm_words[rem_pos] == pattern[offset]:
                        rem_pos += 1
                    else:
                        break
                i = rem_pos
                collapsed = True
                break
        if not collapsed:
            result_words.append(words[i])
            i += 1

    return f"{ts_prefix}{' '.join(result_words)}".strip()


def _remove_paragraph_cycles(paragraphs: list[str], max_cycle_len: int = 25) -> list[str]:
    """Detect and eliminate single- and multi-paragraph repeating hallucination cycles."""
    if len(paragraphs) <= 1:
        return paragraphs

    norms = [_normalize_for_compare(p) for p in paragraphs]
    cleaned: list[str] = []
    cleaned_norms: list[str] = []

    i = 0
    n = len(paragraphs)
    while i < n:
        # Skip completely empty/timestamp-only duplicate noise if identical to previous
        found_cycle = False
        limit_k = min(max_cycle_len, (n - i) // 2)

        for k in range(1, limit_k + 1):
            block_norms = norms[i : i + k]
            combined_len = sum(len(x) for x in block_norms)

            # Verify block i..i+k matches block i+k..i+2k
            if not all(
                _paragraphs_similar(block_norms[j], norms[i + k + j])
                for j in range(k)
            ):
                continue

            # Count consecutive full repetitions of the k-paragraph cycle
            reps = 2
            pos = i + 2 * k
            while pos + k <= n and all(
                _paragraphs_similar(block_norms[j], norms[pos + j])
                for j in range(k)
            ):
                reps += 1
                pos += k

            # Thresholds to avoid false positives on genuine short refrains
            if k == 1:
                # Substantial single paragraph repeated >=2 times, OR short phrase repeated >=3 times
                if combined_len < 15 and reps < 3:
                    continue
            else:
                # Multi-paragraph block repeated >=2 times must carry meaningful text
                if combined_len < 20 and reps < 3:
                    continue

            # Keep only the first occurrence of the cycle
            for j in range(k):
                cleaned.append(paragraphs[i + j])
                cleaned_norms.append(block_norms[j])

            # Advance past all repeated copies AND any partial trailing prefix of the cycle
            tail_pos = pos
            for j in range(k):
                if tail_pos < n and _paragraphs_similar(block_norms[j], norms[tail_pos]):
                    tail_pos += 1
                else:
                    break

            i = tail_pos
            found_cycle = True
            break

        if not found_cycle:
            # Secondary guard against drifting/non-adjacent tail hallucination loops:
            # If a substantial paragraph (>= 30 chars) has already appeared >= 3 times
            # within the recent 30 kept paragraphs, drop the excess repetition.
            cur_norm = norms[i]
            if len(cur_norm) >= 30:
                recent = cleaned_norms[-30:]
                matches_in_recent = sum(1 for r in recent if _paragraphs_similar(cur_norm, r))
                if matches_in_recent >= 3:
                    i += 1
                    continue

            cleaned.append(paragraphs[i])
            cleaned_norms.append(cur_norm)
            i += 1

    return cleaned


def clean_transcript_body(text: str) -> str:
    """Clean raw Gemini transcription output into clean, loop-free Arabic/academic paragraphs.

    Steps:
    1. Strip markdown code fences (```` ``` ````).
    2. Strip conversational AI intro/outro remarks.
    3. Separate inline packed timestamps onto individual paragraphs.
    4. Collapse intra-paragraph runaway phrase repetitions.
    5. Eliminate single- and multi-paragraph repeating hallucination cycles (up to 25 paragraphs).
    """
    if not text or not text.strip():
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _strip_code_fences(cleaned)
    cleaned = _strip_ai_chat_remarks(cleaned)
    cleaned = _separate_inline_timestamps(cleaned)

    # Split into non-empty paragraphs
    raw_paragraphs = [
        re.sub(r"[ \t]+", " ", p).strip()
        for p in re.split(r"\n{2,}", cleaned)
        if p.strip()
    ]

    # Clean intra-paragraph repetition loops first
    paragraphs = [_dedupe_intra_paragraph_loops(p) for p in raw_paragraphs]

    # Eliminate multi-paragraph hallucination cycles (run up to 2 passes for nested cycles)
    paragraphs = _remove_paragraph_cycles(paragraphs)
    paragraphs = _remove_paragraph_cycles(paragraphs)

    return "\n\n".join(p for p in paragraphs if p.strip()).strip()


def format_lecture_markdown(
    title: str,
    course_name: str,
    seq_num: int | str,
    source_type: str,
    source_id: str,
    url: str,
    cleaned_body: str,
) -> str:
    """Format a cleaned lecture transcript into structured, NotebookLM-friendly Markdown."""
    try:
        seq_display = f"{int(seq_num):03d}"
    except (TypeError, ValueError):
        seq_display = str(seq_num or "001")

    source_map = {
        "youtube": "YouTube",
        "gdrive": "Google Drive",
    }
    source_label = source_map.get(str(source_type or "").lower(), str(source_type or "Unknown"))

    cleaned_title = _strip_media_extension(title) if _strip_media_extension else (title or "").strip()
    safe_title = (cleaned_title or title or f"Lecture {seq_display}").strip()
    safe_course = (course_name or "General").strip()
    body = (cleaned_body or "").strip()

    return (
        f"# {safe_title}\n\n"
        f"- **المادة / Course**: {safe_course}\n"
        f"- **رقم المحاضرة / Lecture**: {seq_display}\n"
        f"- **المصدر / Source**: {source_label}\n"
        f"- **المعرف / ID**: `{source_id}`\n"
        f"- **الرابط / URL**: {url}\n"
        f"- **طريقة التفريغ / Engine**: Google Gemini API (Full Verbatim with Timestamps)\n\n"
        f"---\n\n"
        f"## النص الكامل المفرغ مع الطوابع الزمنية\n\n"
        f"{body}\n"
    )


def _get_field(job: Any, key: str, default: Any = None) -> Any:
    if isinstance(job, Mapping):
        return job.get(key, default)
    return getattr(job, key, default)


def _format_size(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "-"
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def _file_uri(path: str) -> str:
    abs_path = os.path.abspath(path).replace("\\", "/")
    if not abs_path.startswith("/"):
        # Windows drive path like C:/...
        encoded = quote(abs_path, safe="/:")
        return f"file:///{encoded}"
    return f"file://{quote(abs_path, safe='/')}"


def _atomic_write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp_path, path)


_STATUS_LABELS = {
    "done": "🟢 مكتمل (Done)",
    "queued": "⏳ قيد الانتظار (Queued)",
    "downloading": "⬇️ جاري التحميل (Downloading)",
    "converting": "🔄 جاري التحويل (Converting)",
    "transcribing": "🎙️ جاري التفريغ (Transcribing)",
    "cleaning": "✨ جاري التنظيف (Cleaning)",
    "paused": "⏸️ متوقف مؤقتاً (Paused)",
    "failed": "❌ فشل (Failed)",
    "cancelled": "🚫 ملغي (Cancelled)",
}


def rebuild_course_artifacts(
    course_name: str,
    output_dir: str,
    jobs_list: Iterable[Any],
) -> dict[str, Any]:
    """Regenerate `_Summary_Report.md` and `000_Complete_<clean_course_name>_NotebookLM.md`.

    Args:
        course_name: Human-readable course name.
        output_dir: Directory where the course `.md` files are saved.
        jobs_list: Iterable of job dicts or job objects belonging to the course.

    Returns:
        Dict with keys `summary_report`, `notebooklm_file`, `completed_count`, `total_count`.
    """
    os.makedirs(output_dir, exist_ok=True)

    jobs = sorted(
        list(jobs_list or []),
        key=lambda j: (
            int(_get_field(j, "seq_num", 0) or 0),
            str(_get_field(j, "title", "") or ""),
        ),
    )

    clean_course = _safe_clean_filename(course_name or "Course")
    summary_path = os.path.join(output_dir, "_Summary_Report.md")
    combined_filename = f"000_Complete_{clean_course}_NotebookLM.md"
    combined_path = os.path.join(output_dir, combined_filename)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_count = len(jobs)
    completed_jobs: list[tuple[Any, str, int]] = []

    table_rows: list[str] = []
    for idx, job in enumerate(jobs, start=1):
        seq_val = _get_field(job, "seq_num", idx)
        try:
            seq_str = f"{int(seq_val):03d}"
        except (TypeError, ValueError):
            seq_str = f"{idx:03d}"

        title = str(_get_field(job, "title", f"Lecture {seq_str}") or f"Lecture {seq_str}")
        source_type = str(_get_field(job, "source_type", "youtube") or "youtube")
        source_id = str(_get_field(job, "source_id", "") or "")
        url = str(_get_field(job, "url", "") or "")
        status = str(_get_field(job, "status", "queued") or "queued").lower()
        output_path = str(_get_field(job, "output_path", "") or "")

        file_exists = bool(output_path and os.path.isfile(output_path) and os.path.getsize(output_path) > 0)
        file_size = os.path.getsize(output_path) if file_exists else 0

        if status == "done" and file_exists:
            completed_jobs.append((job, output_path, file_size))
        elif status == "done" and not file_exists:
            # Output file missing despite done status
            file_size = 0

        status_badge = _STATUS_LABELS.get(status, status)
        size_str = _format_size(file_size)

        if file_exists:
            md_name = os.path.basename(output_path)
            file_link = f"[{md_name}]({_file_uri(output_path)})"
        else:
            file_link = "-"

        src_label = "YouTube" if source_type == "youtube" else ("Google Drive" if source_type == "gdrive" else source_type)
        url_cell = f"[{src_label}]({url})" if url else src_label
        safe_title_cell = title.replace("|", "\\|")

        table_rows.append(
            f"| {seq_str} | {safe_title_cell} | {url_cell} | `{source_id}` | {status_badge} | {size_str} | {file_link} |"
        )

    completed_count = len(completed_jobs)

    # 1. Build _Summary_Report.md
    summary_lines = [
        f"# تقرير تفريغ مادة: {course_name}",
        "",
        f"- **إجمالي المحاضرات / Total Lectures**: {total_count}",
        f"- **المحاضرات المكتملة / Completed**: {completed_count} / {total_count}",
        f"- **آخر تحديث / Last Updated**: {now_str}",
        f"- **الملف المجمع لـ NotebookLM**: [{combined_filename}]({_file_uri(combined_path)})",
        "",
        "## قائمة المحاضرات والتفريغات",
        "",
        "| # | عنوان المحاضرة | المصدر | المعرف | الحالة | الحجم | ملف التفريغ |",
        "|---|----------------|--------|--------|--------|-------|-------------|",
        *table_rows,
        "",
    ]
    _atomic_write_text(summary_path, "\n".join(summary_lines))

    # 2. Build 000_Complete_<clean_course_name>_NotebookLM.md
    toc_lines: list[str] = []
    sections: list[str] = []

    for idx, (job, output_path, file_size) in enumerate(completed_jobs, start=1):
        seq_val = _get_field(job, "seq_num", idx)
        try:
            seq_str = f"{int(seq_val):03d}"
        except (TypeError, ValueError):
            seq_str = f"{idx:03d}"
        title = str(_get_field(job, "title", f"Lecture {seq_str}") or f"Lecture {seq_str}")
        toc_lines.append(f"{idx}. **{seq_str} — {title}** ({_format_size(file_size)})")

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                lecture_md = f.read().strip()
        except OSError:
            continue

        if lecture_md:
            sections.append(lecture_md)

    combined_header = [
        f"# {course_name} — التفريغ الشامل (NotebookLM Master)",
        "",
        f"- **المادة / Course**: {course_name}",
        f"- **عدد المحاضرات المضمنة / Included Lectures**: {len(sections)} / {total_count}",
        f"- **تاريخ التجميع / Generated**: {now_str}",
        "",
        "## فهرس المحاضرات / Table of Contents",
        "",
    ]
    if toc_lines:
        combined_header.extend(toc_lines)
    else:
        combined_header.append("_لا توجد محاضرات مكتملة بعد / No completed lectures yet._")

    combined_parts = ["\n".join(combined_header)]
    if sections:
        combined_parts.append("\n\n---\n\n".join(sections))

    _atomic_write_text(combined_path, "\n\n---\n\n".join(combined_parts) + "\n")

    return {
        "summary_report": summary_path,
        "notebooklm_file": combined_path,
        "completed_count": completed_count,
        "total_count": total_count,
    }
