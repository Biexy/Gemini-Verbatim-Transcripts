"""FFmpeg/FFprobe audio inspection, lightweight MP3 conversion, and smart chunking."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable

from ..config import AppConfig, clean_filename
from .url_parser import (
    AudioProcessingError,
    DownloadCancelledError,
)

_PASSTHROUGH_AUDIO_EXTS = frozenset(
    {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".webm", ".wav", ".flac"}
)
_MAX_PASSTHROUGH_BYTES = 20 * 1024 * 1024
_MAX_PASSTHROUGH_KBPS = 80.0


def _subprocess_kwargs() -> dict[str, Any]:
    """Return platform-safe subprocess kwargs (suppresses console popups on Windows)."""
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return kwargs


def check_ffmpeg() -> tuple[bool, str]:
    """Check whether both ``ffmpeg`` and ``ffprobe`` are installed and executable.

    Returns:
        ``(True, version_summary)`` if ready, or ``(False, error_message)`` otherwise.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return (
            False,
            "FFmpeg غير مثبت أو غير موجود في PATH / FFmpeg not found in PATH.",
        )

    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return (
            False,
            "FFprobe غير موجود في PATH / FFprobe not found in PATH.",
        )

    try:
        proc = subprocess.run(
            [ffmpeg_bin, "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            **_subprocess_kwargs(),
        )
        if proc.returncode != 0:
            return (False, f"FFmpeg returned exit code {proc.returncode}")
        first_line = (proc.stdout or proc.stderr or "").splitlines()[0].strip()
        return (True, first_line or ffmpeg_bin)
    except Exception as exc:
        return (False, f"FFmpeg check failed: {exc}")


def probe_audio_info(file_path: str | Path) -> dict[str, Any]:
    """Inspect a media file using ``ffprobe`` and return duration & stream info."""
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        raise AudioProcessingError(
            f"الملف الصوتي غير موجود أو فارغ / Media file missing or empty: {path}"
        )

    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        raise AudioProcessingError(
            "FFprobe غير موجود في PATH / FFprobe not found in PATH."
        )

    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration,size:stream=codec_type,codec_name,channels,sample_rate,duration:stream_disposition=attached_pic",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            **_subprocess_kwargs(),
        )
    except Exception as exc:
        raise AudioProcessingError(f"فشل فحص الملف عبر ffprobe / ffprobe failed: {exc}") from exc

    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        raise AudioProcessingError(
            f"ملف الوسائط تالف أو غير صالح / Invalid media file ({path.name}): {err}"
        )

    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AudioProcessingError(f"تعذر قراءة مخرجات ffprobe / Invalid ffprobe JSON: {exc}") from exc

    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    video_streams = [
        s
        for s in streams
        if s.get("codec_type") == "video"
        and not (s.get("disposition") or {}).get("attached_pic")
    ]
    duration = _parse_duration(fmt.get("duration"))
    if duration <= 0.0:
        for s in audio_streams or streams:
            sd = _parse_duration(s.get("duration"))
            if sd > duration:
                duration = sd

    return {
        "duration_seconds": duration,
        "has_audio": bool(audio_streams),
        "has_video": bool(video_streams),
        "streams": streams,
        "size_bytes": path.stat().st_size,
    }


def _parse_duration(raw: Any) -> float:
    if raw is None:
        return 0.0
    try:
        val = float(raw)
        return val if val >= 0.0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _run_ffmpeg(cmd: list[str], cancel_event: threading.Event | None = None) -> None:
    """Run an ffmpeg command, polling cancel_event so long conversions can be aborted cleanly."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        **_subprocess_kwargs(),
    )
    try:
        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise DownloadCancelledError("تم إلغاء التحويل / Conversion cancelled")
            try:
                proc.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                continue

        stderr_text = proc.stderr.read() if proc.stderr else ""
        if proc.returncode != 0:
            raise AudioProcessingError(
                f"فشل تحويل الصوت عبر FFmpeg (code {proc.returncode}) / "
                f"FFmpeg conversion failed: {stderr_text.strip()[:400]}"
            )
    finally:
        if proc.stderr:
            try:
                proc.stderr.close()
            except Exception:
                pass


def prepare_audio_for_transcription(
    raw_path: str | Path,
    temp_dir: str | Path,
    source_id: str,
    config: AppConfig | None = None,
    progress_cb: Callable[[float], Any] | None = None,
    cancel_event: threading.Event | None = None,
    max_chunk_seconds: int | None = None,
    keep_audio: bool | None = None,
) -> list[dict[str, Any]]:
    """Convert raw audio/video to mono 16kHz MP3 and split into safe transcription chunks.

    - In ``eco`` and ``strict_audio`` modes, performs zero-reencode direct upload
      pass-through when the input is already pure audio, within the chunk duration limit,
      <= 20 MB, <= 80 kbps, and ``keep_audio`` is False.
    - Otherwise converts using ``-vn -ac 1 -ar 16000 -b:a <effective_bitrate>`` (32k by
      default in ``eco``/``strict_audio``; ``config.audio_bitrate`` in ``balanced``) and
      splits lectures longer than ``config.max_chunk_minutes * 60`` seconds into sequential
      chunks via ``ffmpeg -f segment -c copy``.
    - Deletes ``raw_path`` after successful conversion/move to conserve disk space.

    Returns:
        List of chunk dicts: ``[{"path": str, "offset_seconds": int, "duration_seconds": float}, ...]``.
    """
    cfg = config or AppConfig()
    ok, msg = check_ffmpeg()
    if not ok:
        raise AudioProcessingError(msg)

    raw_file = Path(raw_path).resolve()
    out_dir = Path(temp_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if cancel_event is not None and cancel_event.is_set():
        raise DownloadCancelledError("تم إلغاء التحويل / Conversion cancelled")

    # Step 1: Probe input media
    info = probe_audio_info(raw_file)
    if not info["has_audio"]:
        raise AudioProcessingError(
            f"الملف لا يحتوي على مسار صوتي / Media file has no audio stream: {raw_file.name}"
        )

    safe_id = clean_filename(source_id or raw_file.stem, max_length=80)
    mode = str(getattr(cfg, "download_mode", "eco") or "eco").strip().lower()
    if mode not in ("eco", "balanced", "strict_audio"):
        mode = "eco"

    eff_bitrate_fn = getattr(cfg, "effective_audio_bitrate", None)
    if callable(eff_bitrate_fn):
        bitrate = str(eff_bitrate_fn()).strip()
    else:
        raw_br = (getattr(cfg, "audio_bitrate", "64k") or "64k").strip()
        bitrate = "32k" if (mode in ("eco", "strict_audio") and raw_br == "64k") else raw_br

    effective_keep_audio = (
        bool(getattr(cfg, "keep_audio", False))
        if keep_audio is None
        else bool(keep_audio)
    )

    chunk_limit = (
        max(1, int(max_chunk_seconds))
        if max_chunk_seconds is not None
        else max(60, int(cfg.max_chunk_minutes) * 60)
    )

    if progress_cb:
        try:
            progress_cb(10.0)
        except Exception:
            pass

    # Clean up stale chunks for this source_id from earlier failed runs
    for stale in out_dir.glob(f"{safe_id}_chunk_*"):
        if stale.is_file() and stale.resolve() != raw_file:
            try:
                stale.unlink()
            except OSError:
                pass

    # Step 1b: Direct Upload Pass-Through in eco / strict_audio modes
    raw_ext = raw_file.suffix.lower()
    duration = float(info.get("duration_seconds") or 0.0)
    size_bytes = int(info.get("size_bytes") or raw_file.stat().st_size)
    approx_kbps = (
        (size_bytes * 8.0) / (duration * 1000.0)
        if duration > 0.0
        else float("inf")
    )

    can_passthrough = (
        mode in ("eco", "strict_audio")
        and not effective_keep_audio
        and not bool(info.get("has_video"))
        and raw_ext in _PASSTHROUGH_AUDIO_EXTS
        and 0.0 < duration <= (chunk_limit + 2.0)
        and size_bytes <= _MAX_PASSTHROUGH_BYTES
        and approx_kbps <= _MAX_PASSTHROUGH_KBPS
    )
    if can_passthrough:
        final_chunk = out_dir / f"{safe_id}_chunk_000{raw_ext}"
        try:
            os.replace(raw_file, final_chunk)
        except OSError:
            shutil.move(str(raw_file), str(final_chunk))
        if progress_cb:
            try:
                progress_cb(100.0)
            except Exception:
                pass
        return [
            {
                "path": str(final_chunk),
                "offset_seconds": 0,
                "duration_seconds": round(duration, 2),
                "direct_passthrough": True,
            }
        ]

    converted_mp3 = out_dir / f"{safe_id}_converted.mp3"
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    # Step 2: Convert to lightweight mono 16kHz MP3
    convert_cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(raw_file),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        bitrate,
        "-f",
        "mp3",
        str(converted_mp3),
    ]
    _run_ffmpeg(convert_cmd, cancel_event=cancel_event)

    if not converted_mp3.exists() or converted_mp3.stat().st_size == 0:
        raise AudioProcessingError(
            f"فشل إنشاء ملف MP3 / Converted MP3 file missing or empty: {converted_mp3}"
        )

    if progress_cb:
        try:
            progress_cb(65.0)
        except Exception:
            pass

    # Re-probe converted MP3 for exact duration
    mp3_info = probe_audio_info(converted_mp3)
    total_duration = mp3_info["duration_seconds"] or info["duration_seconds"]

    chunks: list[dict[str, Any]] = []

    # Step 3: Split if longer than chunk_limit (with small 2s tolerance to avoid tiny tail slivers)
    if total_duration > (chunk_limit + 2.0):
        segment_pattern = str(out_dir / f"{safe_id}_chunk_%03d.mp3")
        split_cmd = [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(converted_mp3),
            "-f",
            "segment",
            "-segment_time",
            str(chunk_limit),
            "-c",
            "copy",
            "-reset_timestamps",
            "1",
            segment_pattern,
        ]
        _run_ffmpeg(split_cmd, cancel_event=cancel_event)

        segment_files = sorted(out_dir.glob(f"{safe_id}_chunk_*.mp3"))
        if not segment_files:
            raise AudioProcessingError(
                f"فشل تقسيم الصوت إلى مقاطع / FFmpeg produced no segment files for {safe_id}"
            )

        cumulative_offset = 0.0
        for seg_path in segment_files:
            if not seg_path.exists() or seg_path.stat().st_size == 0:
                try:
                    seg_path.unlink()
                except OSError:
                    pass
                continue
            try:
                seg_info = probe_audio_info(seg_path)
                seg_dur = seg_info["duration_seconds"]
            except AudioProcessingError:
                seg_dur = 0.0
            # Discard negligible empty trailing frames (< 0.15s) if we already have chunks
            if seg_dur < 0.15 and chunks:
                try:
                    seg_path.unlink()
                except OSError:
                    pass
                continue
            chunks.append(
                {
                    "path": str(seg_path),
                    "offset_seconds": int(round(cumulative_offset)),
                    "duration_seconds": round(seg_dur, 2),
                }
            )
            cumulative_offset += seg_dur

        # Remove intermediate unsplit MP3 now that segments are ready
        try:
            converted_mp3.unlink()
        except OSError:
            pass
    else:
        final_chunk = out_dir / f"{safe_id}_chunk_000.mp3"
        os.replace(converted_mp3, final_chunk)
        chunks.append(
            {
                "path": str(final_chunk),
                "offset_seconds": 0,
                "duration_seconds": round(total_duration, 2),
            }
        )

    # Step 4: Delete raw downloaded media file to save disk space
    chunk_resolved = {Path(c["path"]).resolve() for c in chunks}
    if raw_file.exists() and raw_file not in chunk_resolved:
        try:
            raw_file.unlink()
        except OSError:
            pass

    if progress_cb:
        try:
            progress_cb(100.0)
        except Exception:
            pass

    return chunks


def merge_audio_chunks_to_mp3(
    chunks: list[dict[str, Any]] | list[str],
    dest_path: str | Path,
    bitrate: str = "32k",
    cancel_event: threading.Event | None = None,
) -> str:
    """Save or concatenate audio chunk(s) into a single clean MP3 at ``dest_path``."""
    chunk_paths: list[Path] = []
    for item in chunks or []:
        raw = item.get("path") if isinstance(item, dict) else item
        if not raw:
            continue
        p = Path(raw)
        if p.is_file() and p.stat().st_size > 0:
            chunk_paths.append(p)

    if not chunk_paths:
        raise AudioProcessingError(
            "لا توجد ملفات صوتية صالحة للحفظ / No valid audio chunks found to save."
        )

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_name(f"{dest.stem}.tmp{dest.suffix or '.mp3'}")
    ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

    try:
        if len(chunk_paths) == 1:
            src = chunk_paths[0]
            if src.suffix.lower() == ".mp3":
                shutil.copy2(src, tmp_dest)
                os.replace(tmp_dest, dest)
                return str(dest)
            cmd = [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                (bitrate or "32k").strip(),
                "-f",
                "mp3",
                str(tmp_dest),
            ]
            _run_ffmpeg(cmd, cancel_event=cancel_event)
            os.replace(tmp_dest, dest)
            return str(dest)

        # Multiple chunks: concatenate cleanly via FFmpeg concat demuxer
        concat_list = dest.parent / f".{dest.stem}_concat.txt"
        try:
            lines = []
            for p in chunk_paths:
                escaped = str(p.resolve()).replace("\\", "/").replace("'", "'\\''")
                lines.append(f"file '{escaped}'")
            concat_list.write_text("\n".join(lines), encoding="utf-8")

            all_mp3 = all(p.suffix.lower() == ".mp3" for p in chunk_paths)
            if all_mp3:
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_list),
                    "-c",
                    "copy",
                    str(tmp_dest),
                ]
            else:
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_list),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-b:a",
                    (bitrate or "32k").strip(),
                    "-f",
                    "mp3",
                    str(tmp_dest),
                ]
            _run_ffmpeg(cmd, cancel_event=cancel_event)
            os.replace(tmp_dest, dest)
            return str(dest)
        finally:
            if concat_list.exists():
                try:
                    concat_list.unlink()
                except OSError:
                    pass
    except Exception:
        if tmp_dest.exists():
            try:
                tmp_dest.unlink()
            except OSError:
                pass
        raise


def cleanup_audio_files(items: list[str] | list[dict[str, Any]]) -> int:
    """Delete a list of audio file paths or chunk dicts, returning count deleted."""
    deleted = 0
    for item in items or []:
        raw = item.get("path") if isinstance(item, dict) else item
        if not raw:
            continue
        p = Path(raw)
        if p.exists():
            try:
                p.unlink()
                deleted += 1
            except OSError:
                pass
    return deleted
