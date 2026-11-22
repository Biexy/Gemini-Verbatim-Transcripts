"""Unified launcher for Lecture Transcriber (Web UI by default, or headless CLI via --cli).

Examples:
    python run.py
    python run.py --port 8765 --no-browser
    python run.py --cli --course "Quran Studies" --urls "https://youtu.be/..."
    python run.py --cli --course "Quran Studies" --file links.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

# Ensure parent directory is importable whether invoked from inside or outside lecture_transcriber/
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))


def build_launcher_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Lecture Transcriber — lightweight local Web UI & headless CLI.",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in headless CLI mode instead of launching the local Web UI server.",
    )
    # Web UI options
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind the local Web UI server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for the local Web UI server (default: from config, 8765).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the Web UI in the default browser.",
    )
    # Forwarded CLI options (shown cleanly in --help)
    parser.add_argument("--course", "-c", type=str, default="", help="[CLI] Course / folder name.")
    parser.add_argument("--urls", "-u", nargs="+", default=[], help="[CLI] One or more YouTube / Google Drive URLs.")
    parser.add_argument("--file", "-f", type=str, default="", help="[CLI] Text file with one URL per line.")
    parser.add_argument("--output", "-o", type=str, default="", help="[CLI] Override output base directory.")
    parser.add_argument("--api-key", type=str, default="", help="[CLI] Override Gemini API key.")
    parser.add_argument("--download-workers", type=int, default=None, help="[CLI] Download worker count.")
    parser.add_argument("--transcribe-workers", type=int, default=None, help="[CLI] Transcribe worker count.")
    parser.add_argument("--bitrate", type=str, default="", help="[CLI] Audio bitrate (e.g. 32k, 64k).")
    parser.add_argument(
        "--mode",
        choices=["eco", "balanced", "strict_audio"],
        default=None,
        help="[CLI] Bandwidth-saver mode: eco (default), balanced, or strict_audio.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        default=False,
        help="[CLI] Keep a lightweight MP3 copy in <course>/audio/ alongside each transcript.",
    )
    parser.add_argument("--proxy", type=str, default=None, help="[CLI] Optional HTTP/SOCKS5 proxy URL.")
    parser.add_argument("--resume-only", action="store_true", help="[CLI] Resume pending jobs in database.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_launcher_parser()
    args = parser.parse_args(argv)

    if args.cli:
        from lecture_transcriber.cli import run_cli

        cli_args: list[str] = []
        if args.course:
            cli_args.extend(["--course", args.course])
        if args.urls:
            cli_args.append("--urls")
            cli_args.extend(args.urls)
        if args.file:
            cli_args.extend(["--file", args.file])
        if args.output:
            cli_args.extend(["--output", args.output])
        if args.api_key:
            cli_args.extend(["--api-key", args.api_key])
        if args.download_workers is not None:
            cli_args.extend(["--download-workers", str(args.download_workers)])
        if args.transcribe_workers is not None:
            cli_args.extend(["--transcribe-workers", str(args.transcribe_workers)])
        if args.bitrate:
            cli_args.extend(["--bitrate", args.bitrate])
        if args.mode:
            cli_args.extend(["--mode", args.mode])
        if args.keep_audio:
            cli_args.append("--keep-audio")
        if args.proxy is not None:
            cli_args.extend(["--proxy", args.proxy])
        if args.resume_only:
            cli_args.append("--resume-only")
        return run_cli(cli_args)

    from lecture_transcriber.web.server import run_server

    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
