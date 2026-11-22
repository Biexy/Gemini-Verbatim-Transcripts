#!/usr/bin/env python3
"""Top-level launcher for Gemini Verbatim Transcripts (Web UI & Headless CLI)."""

from __future__ import annotations

import sys

from lecture_transcriber.run import main

if __name__ == "__main__":
    sys.exit(main())
