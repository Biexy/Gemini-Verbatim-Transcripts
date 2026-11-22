"""Headless CLI interface for Lecture Transcriber with live terminal progress."""

from __future__ import annotations

import argparse
import os
import queue
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

TERMINAL_STATUSES = {"done", "failed", "cancelled"}


def _format_speed(bps: float) -> str:
    if not bps or bps <= 0:
        return ""
    if bps >= 1048576:
        return f"{bps / 1048576:.2f} MB/s"
    if bps >= 1024:
        return f"{bps / 1024:.1f} KB/s"
    return f"{bps:.0f} B/s"


def _format_eta(sec: Any) -> str:
    if sec is None:
        return ""
    try:
        s = int(sec)
    except (TypeError, ValueError):
        return ""
    if s < 0:
        return ""
    h, rem = divmod(s, 3600)
    m, r = divmod(rem, 60)
    return f"{h}:{m:02d}:{r:02d}" if h > 0 else f"{m:02d}:{r:02d}"


def _read_urls_from_file(path: str) -> list[str]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Links file not found: {path}")
    urls: list[str] = []
    for line in file_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            urls.append(s)
    return urls


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lecture-transcriber-cli",
        description="Headless CLI for downloading & verbatim-transcribing YouTube and Google Drive lectures.",
    )
    parser.add_argument(
        "--course",
        "-c",
        type=str,
        default="",
        help="Course / folder name where transcripts will be saved.",
    )
    parser.add_argument(
        "--urls",
        "-u",
        nargs="+",
        default=[],
        help="One or more YouTube / Google Drive URLs (videos, playlists, files, or folders).",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default="",
        help="Path to a text file containing one URL per line.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="",
        help="Override output base directory for transcripts.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="",
        help="Override Gemini API key for this session.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=None,
        help="Number of concurrent download workers.",
    )
    parser.add_argument(
        "--transcribe-workers",
        type=int,
        default=None,
        help="Number of concurrent Gemini transcription workers.",
    )
    parser.add_argument(
        "--bitrate",
        type=str,
        default="",
        help="Audio conversion bitrate (e.g. 32k, 48k, 64k, 96k).",
    )
    parser.add_argument(
        "--mode",
        choices=["eco", "balanced", "strict_audio"],
        default=None,
        help="Download & bandwidth-saver mode: eco (default), balanced, or strict_audio (no video fallback).",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        default=False,
        help="Save a lightweight MP3 copy in <course>/audio/ alongside each transcript.",
    )
    parser.add_argument(
        "--proxy",
        type=str,
        default=None,
        help="Optional HTTP/SOCKS5 proxy URL.",
    )
    parser.add_argument(
        "--resume-only",
        action="store_true",
        help="Resume any pending/interrupted jobs already in the database without adding new URLs.",
    )
    return parser


def _to_dict(obj: Any) -> dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {}


class _SessionConfigManager:
    """In-memory overlay over ConfigManager for per-invocation CLI overrides."""

    def __init__(self, base: Any, overrides: dict[str, Any]) -> None:
        self._base = base
        self._overrides = dict(overrides)

    @property
    def config(self) -> Any:
        return self.get()

    def get(self) -> Any:
        cfg = self._base.get() if hasattr(self._base, "get") else self._base.load()
        for k, v in self._overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def load(self) -> Any:
        return self.get()

    def ensure_directories(self) -> None:
        cfg = self.get()
        Path(cfg.temp_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.output_base_dir).mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    from lecture_transcriber.config import ConfigManager, ensure_utf8_stdio
    from lecture_transcriber.database import Database
    from lecture_transcriber.engine import TranscriptionEngine

    ensure_utf8_stdio()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    urls: list[str] = [u.strip() for u in (args.urls or []) if u.strip()]
    if args.file:
        try:
            urls.extend(_read_urls_from_file(args.file))
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2

    if not args.resume_only:
        if not urls:
            parser.error("Provide --urls and/or --file (or pass --resume-only).")
        if not args.course.strip():
            parser.error("--course is required when adding URLs.")

    base_config_manager = ConfigManager()

    # Apply session-scoped CLI overrides without clobbering saved Web UI config.json
    overrides: dict[str, Any] = {}
    if args.output:
        overrides["output_base_dir"] = os.path.abspath(args.output)
    if args.api_key:
        overrides["gemini_api_key"] = args.api_key.strip()
    if args.download_workers is not None:
        overrides["download_workers"] = max(1, args.download_workers)
    if args.transcribe_workers is not None:
        overrides["transcribe_workers"] = max(1, args.transcribe_workers)
    if args.bitrate:
        overrides["audio_bitrate"] = args.bitrate.strip()
    if args.mode:
        overrides["download_mode"] = args.mode
    if args.keep_audio:
        overrides["keep_audio"] = True
    if args.proxy is not None:
        overrides["proxy"] = args.proxy.strip()

    config_manager: Any = (
        _SessionConfigManager(base_config_manager, overrides) if overrides else base_config_manager
    )

    db = Database()
    engine = TranscriptionEngine(config_manager, db)

    ev_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=512)

    def _on_event(ev: dict[str, Any]) -> None:
        try:
            ev_queue.put_nowait(ev)
        except queue.Full:
            pass

    sub_token: Any = None
    use_queue = ev_queue
    if hasattr(engine, "subscribe"):
        try:
            ret = engine.subscribe()
            if isinstance(ret, queue.Queue):
                use_queue = ret
                sub_token = ret
            else:
                sub_token = ret
        except TypeError:
            sub_token = engine.subscribe(_on_event)

    engine.start()
    tracked_ids: set[str] = set()

    try:
        if urls:
            print(f"Course: {args.course}")
            print(f"Expanding {len(urls)} URL(s)...")
            result = engine.enqueue_urls(args.course.strip(), urls)
            added_jobs = result.get("jobs", []) if isinstance(result, dict) else []
            added_count = result.get("added", len(added_jobs)) if isinstance(result, dict) else 0
            skipped_count = result.get("skipped", 0) if isinstance(result, dict) else 0
            print(f"Enqueued: {added_count} | Skipped: {skipped_count}")
            for j in added_jobs:
                jd = _to_dict(j)
                if jd.get("id"):
                    tracked_ids.add(jd["id"])
            if not tracked_ids and isinstance(result, dict) and result.get("job_ids"):
                tracked_ids.update(result["job_ids"])

        if args.resume_only or not tracked_ids:
            for j in db.list_jobs():
                jd = _to_dict(j)
                if jd.get("status") not in TERMINAL_STATUSES and jd.get("id"):
                    tracked_ids.add(jd["id"])

        if not tracked_ids:
            print("No pending jobs to process.")
            return 0

        print(f"Processing {len(tracked_ids)} job(s)... Press Ctrl+C to pause & exit safely.\n")
        last_print: dict[str, float] = {}
        last_status: dict[str, str] = {}

        while True:
            # Drain queued events for crisp live output
            while True:
                try:
                    ev = use_queue.get(timeout=0.25)
                except queue.Empty:
                    break
                etype = ev.get("type", "")
                data = ev.get("data", ev) if isinstance(ev, dict) else {}
                if etype in ("offline", "online", "network", "network_status"):
                    online = data.get("online", etype == "online")
                    msg = data.get("message") or ("Back online — resuming!" if online else "Internet offline — waiting...")
                    print(f"\n[NETWORK] {msg}")
                elif etype == "log":
                    lvl = str(data.get("level", "info")).upper()
                    if lvl in ("WARNING", "ERROR"):
                        print(f"[{lvl}] {data.get('message', '')}")
                else:
                    job_obj = data.get("job") if isinstance(data, dict) else None
                    if not job_obj and isinstance(data, dict) and (data.get("id") or data.get("job_id")):
                        job_obj = data
                    if isinstance(job_obj, dict):
                        jid = job_obj.get("id") or job_obj.get("job_id")
                        if jid and jid in tracked_ids:
                            st = str(job_obj.get("status", ""))
                            pct = float(job_obj.get("progress", 0.0) or 0.0)
                            spd = _format_speed(float(job_obj.get("speed_bps", 0.0) or 0.0))
                            eta = _format_eta(job_obj.get("eta_seconds"))
                            title = str(job_obj.get("title") or jid)[:48]
                            now = time.monotonic()
                            if st != last_status.get(jid) or (now - last_print.get(jid, 0.0)) >= 1.5 or st in TERMINAL_STATUSES:
                                extras = " | ".join(x for x in (spd, f"ETA {eta}" if eta else "") if x)
                                extra_str = f" ({extras})" if extras else ""
                                print(f"  [{st.upper():12s}] {pct:5.1f}%  {title}{extra_str}")
                                last_print[jid] = now
                                last_status[jid] = st

            # Check DB completion state
            done_cnt = 0
            failed_cnt = 0
            remaining = 0
            for jid in tracked_ids:
                job = db.get_job(jid)
                if not job:
                    continue
                jd = _to_dict(job)
                st = jd.get("status", "")
                if st == "done":
                    done_cnt += 1
                elif st in ("failed", "cancelled"):
                    failed_cnt += 1
                elif st == "paused":
                    # Paused jobs won't finish by themselves in CLI
                    failed_cnt += 1
                else:
                    remaining += 1

            if remaining == 0:
                print("\n" + "=" * 60)
                print(f"Finished! Completed: {done_cnt} | Failed/Cancelled: {failed_cnt} | Total: {len(tracked_ids)}")
                print("=" * 60)
                return 0 if failed_cnt == 0 else 1

    except KeyboardInterrupt:
        print("\nInterrupted by user — pausing active jobs safely so downloads can resume later...")
        if hasattr(engine, "pause_all"):
            engine.pause_all()
        return 130
    finally:
        if hasattr(engine, "unsubscribe") and sub_token is not None:
            try:
                engine.unsubscribe(sub_token)
            except Exception:
                pass
        if hasattr(engine, "stop"):
            engine.stop()
        if hasattr(db, "close"):
            db.close()


def main() -> None:
    sys.exit(run_cli())


if __name__ == "__main__":
    main()
