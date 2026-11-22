"""Tests for URL parsing, Google Drive Range resume, YouTube fallback, and FFmpeg audio processing."""

from __future__ import annotations

import hashlib
import http.server
import math
import os
import re
import shutil
import socketserver
import struct
import subprocess
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlparse

from lecture_transcriber.config import AppConfig
from lecture_transcriber.downloader import (
    BALANCED_AUDIO_FORMAT,
    ECO_AUDIO_FORMAT,
    STRICT_AUDIO_FORMAT,
    AudioProcessingError,
    DownloadCancelledError,
    DownloadError,
    DownloadPausedError,
    GDrivePermissionError,
    GDriveQuotaError,
    URLParseError,
    check_ffmpeg,
    cleanup_audio_files,
    detect_extension_from_magic,
    detect_url_type,
    download_gdrive_file,
    download_youtube_audio,
    expand_urls,
    extract_gdrive_file_id,
    extract_gdrive_folder_id,
    extract_youtube_id,
    merge_audio_chunks_to_mp3,
    parse_gdrive_confirm_url,
    parse_gdrive_folder_html,
    prepare_audio_for_transcription,
    probe_audio_info,
)


def _generate_tone_wav(path: Path, duration_seconds: float, sample_rate: int = 16000) -> None:
    """Generate a valid mono PCM WAV file with a 440Hz sine tone (no external deps)."""
    n_samples = int(duration_seconds * sample_rate)
    amplitude = 12000
    freq = 440.0
    frames = bytearray()
    for i in range(n_samples):
        sample = int(amplitude * math.sin(2.0 * math.pi * freq * i / sample_rate))
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)


class TestURLParser(unittest.TestCase):
    def test_youtube_variants_detection(self) -> None:
        self.assertEqual(
            detect_url_type("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            ("youtube_video", "dQw4w9WgXcQ"),
        )
        self.assertEqual(
            detect_url_type("https://youtu.be/dQw4w9WgXcQ?t=42"),
            ("youtube_video", "dQw4w9WgXcQ"),
        )
        self.assertEqual(
            detect_url_type("https://www.youtube.com/shorts/abcDEF12345"),
            ("youtube_video", "abcDEF12345"),
        )
        self.assertEqual(
            detect_url_type("https://www.youtube.com/live/xyz98765432"),
            ("youtube_video", "xyz98765432"),
        )
        self.assertEqual(
            detect_url_type("https://www.youtube.com/playlist?list=PL1234567890abcdef"),
            ("youtube_playlist", "PL1234567890abcdef"),
        )
        # Video with list= defaults to single video unless include_playlist=True
        watch_with_list = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL1234567890abcdef"
        self.assertEqual(detect_url_type(watch_with_list), ("youtube_video", "dQw4w9WgXcQ"))
        self.assertEqual(
            detect_url_type(watch_with_list, include_playlist=True),
            ("youtube_playlist", "PL1234567890abcdef"),
        )
        self.assertEqual(extract_youtube_id("youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_gdrive_variants_detection(self) -> None:
        fid = "1AbCdEfGhIjKlMnOpQrStUvWxYz12345"
        self.assertEqual(
            detect_url_type(f"https://drive.google.com/file/d/{fid}/view?usp=sharing"),
            ("gdrive_file", fid),
        )
        self.assertEqual(
            detect_url_type(f"https://drive.google.com/open?id={fid}"),
            ("gdrive_file", fid),
        )
        self.assertEqual(
            detect_url_type(f"https://drive.google.com/uc?export=download&id={fid}"),
            ("gdrive_file", fid),
        )
        self.assertEqual(
            extract_gdrive_file_id(f"https://drive.google.com/file/u/0/d/{fid}/preview"),
            fid,
        )

        folder_id = "1FolderIdExample987654321"
        self.assertEqual(
            detect_url_type(f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"),
            ("gdrive_folder", folder_id),
        )
        self.assertEqual(
            extract_gdrive_folder_id(f"https://drive.google.com/embeddedfolderview?id={folder_id}#list"),
            folder_id,
        )

        with self.assertRaises(URLParseError):
            detect_url_type("https://example.com/lecture.mp3")

    def test_expand_urls_deduplication_and_ordering(self) -> None:
        fid = "1AbCdEfGhIjKlMnOpQrStUvWxYz12345"
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",  # duplicate of first
            f"https://drive.google.com/file/d/{fid}/view",
            f"https://drive.google.com/open?id={fid}",  # duplicate of third
            "https://www.youtube.com/shorts/abcDEF12345",
        ]
        items = expand_urls(urls, fetch_metadata=False)
        self.assertEqual(len(items), 3)
        self.assertEqual([(i.source_type, i.source_id, i.seq_num) for i in items], [
            ("youtube", "dQw4w9WgXcQ", 1),
            ("gdrive", fid, 2),
            ("youtube", "abcDEF12345", 3),
        ])

    def test_parse_gdrive_folder_html_embedded_and_ivd_and_private(self) -> None:
        # 1. embeddedfolderview HTML (skips PDFs & subfolders)
        embedded_html = """
        <div class="flip-entry" id="entry-fileAAA111">
          <a href="https://drive.google.com/file/d/fileAAA111/view">
            <div class="flip-entry-title">المحاضرة الأولى.mp4</div>
          </a>
        </div>
        <div class="flip-entry" id="entry-pdfSkip99">
          <a href="https://drive.google.com/file/d/pdfSkip99/view">
            <div class="flip-entry-title">ملخص.pdf</div>
          </a>
        </div>
        <div class="flip-entry" id="entry-subFold88">
          <a href="https://drive.google.com/drive/folders/subFold88">
            <div class="flip-entry-title">Subfolder</div>
          </a>
        </div>
        <div class="flip-entry" id="entry-fileBBB222">
          <a href="https://drive.google.com/file/d/fileBBB222/view">
            <div class="flip-entry-title">Lecture 02.m4a</div>
          </a>
        </div>
        """
        entries = parse_gdrive_folder_html(embedded_html, folder_id="rootFolder1")
        self.assertEqual(entries, [
            ("fileAAA111", "المحاضرة الأولى.mp4"),
            ("fileBBB222", "Lecture 02.m4a"),
        ])

        # 2. window['_DRIVE_ivd'] with literal Arabic UTF-8 + \x22 escapes
        ivd_html = (
            r"""<script>window['_DRIVE_ivd'] = '\x5b\x5b\x5b\x22fileIVD111\x22,null,"""
            r"""\x22محاضرة التفسير 1.mp3\x22,\x22audio/mpeg\x22\x5d\x5d\x5d';</script>"""
        )
        ivd_entries = parse_gdrive_folder_html(ivd_html, folder_id="rootFolder2")
        self.assertEqual(ivd_entries, [("fileIVD111", "محاضرة التفسير 1.mp3")])

        # 3. Private folder HTML -> GDrivePermissionError
        private_html = "<html><body>Sign in to continue to Google Drive - You need access</body></html>"
        with self.assertRaises(GDrivePermissionError):
            parse_gdrive_folder_html(private_html, folder_id="priv12345")

    def test_expand_youtube_playlist_mocked(self) -> None:
        fake_info = {
            "entries": [
                {"id": "vid00000001", "title": "الدرس الأول", "duration": 1800},
                {"id": "vid00000002", "title": "[Private video]", "duration": 0},
                {"id": "vid00000003", "title": "الدرس الثاني", "duration": 2400},
                {"id": "vid00000001", "title": "Duplicate", "duration": 1800},
            ]
        }

        class FakeYDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=False):
                return fake_info

        with mock.patch("yt_dlp.YoutubeDL", FakeYDL):
            items = expand_urls(["https://www.youtube.com/playlist?list=PLabcdef123456"])
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].source_id, "vid00000001")
        self.assertEqual(items[0].title, "الدرس الأول")
        self.assertEqual(items[1].source_id, "vid00000003")
        self.assertEqual(items[1].seq_num, 2)

    def test_live_youtube_metadata_smoke(self) -> None:
        """Non-destructive live smoke test of url_parser against a real public YouTube video."""
        # Classic short public video ("Me at the zoo" - jNQXAC9IVRw)
        items = expand_urls(["https://www.youtube.com/watch?v=jNQXAC9IVRw"], fetch_metadata=True)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source_type, "youtube")
        self.assertEqual(items[0].source_id, "jNQXAC9IVRw")
        self.assertTrue(len(items[0].title) > 0)


class _ThreadingTCPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class TestGDriveDownloaderMockServer(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        # Create a 640KB deterministic payload starting with MP3 ID3 magic bytes
        self.payload = b"ID3\x03\x00\x00\x00\x00\x00\x00" + os.urandom(640 * 1024 - 10)
        self.expected_sha256 = hashlib.sha256(self.payload).hexdigest()
        self.state = {
            "confirm_served": False,
            "drop_once_at_40pct": True,
            "range_headers_seen": [],
            "requests_count": 0,
        }

        payload = self.payload
        state = self.state

        class MockGDriveHandler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format, *args):  # noqa: A002
                pass

            def do_GET(self):  # noqa: N802
                state["requests_count"] += 1
                parsed = urlparse(self.path)
                qs = parse_qs(parsed.query)

                if parsed.path == "/private":
                    body = b"<html><body><h1>You need access</h1><p>Sign in to continue to Google Drive</p></body></html>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if parsed.path == "/quota":
                    body = b"<html><body><p>Sorry, you can't download this file at this time. Too many users have viewed or downloaded this file recently (quota exceeded).</p></body></html>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if parsed.path == "/uc":
                    # Serve modern virus-scan confirm HTML form on initial request
                    state["confirm_served"] = True
                    html_page = (
                        '<!DOCTYPE html><html><body>'
                        '<form id="download-form" action="/download" method="get">'
                        '<input type="hidden" name="id" value="file12345">'
                        '<input type="hidden" name="confirm" value="t">'
                        '<input type="hidden" name="uuid" value="uuid-9999">'
                        '</form></body></html>'
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html_page)))
                    self.end_headers()
                    self.wfile.write(html_page)
                    return

                if parsed.path == "/download":
                    # Verify confirm token & uuid were carried over
                    if qs.get("confirm") != ["t"] or qs.get("uuid") != ["uuid-9999"]:
                        self.send_response(400)
                        self.end_headers()
                        return

                    total = len(payload)
                    range_hdr = self.headers.get("Range")
                    if range_hdr:
                        state["range_headers_seen"].append(range_hdr)
                        m = re.match(r"bytes=(\d+)-(\d*)", range_hdr)
                        start = int(m.group(1)) if m else 0
                        end = int(m.group(2)) if (m and m.group(2)) else (total - 1)
                        chunk = payload[start : end + 1]
                        self.send_response(206)
                        self.send_header("Content-Type", "audio/mpeg")
                        self.send_header("Content-Disposition", 'attachment; filename="lecture_01.mp3"')
                        self.send_header("Accept-Ranges", "bytes")
                        self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                        self.send_header("Content-Length", str(len(chunk)))
                        self.end_headers()
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        return

                    # First media download: advertise full Content-Length, send only 40%, then cut socket!
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/mpeg")
                    self.send_header("Content-Disposition", 'attachment; filename="lecture_01.mp3"')
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(total))
                    self.end_headers()
                    if state["drop_once_at_40pct"]:
                        state["drop_once_at_40pct"] = False
                        cut_point = int(total * 0.40)
                        self.wfile.write(payload[:cut_point])
                        self.wfile.flush()
                        self.close_connection = True
                        return
                    self.wfile.write(payload)
                    self.wfile.flush()
                    return

                self.send_response(404)
                self.end_headers()

        self.server = _ThreadingTCPServer(("127.0.0.1", 0), MockGDriveHandler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self._tmpdir.cleanup()

    def test_virus_scan_confirm_and_range_resume_sha256_match(self) -> None:
        url = f"http://127.0.0.1:{self.port}/uc?export=download&id=file12345"
        progress_updates: list[tuple[float, int, int]] = []
        cfg = AppConfig(socket_timeout=10, max_retries=4)

        out_path = download_gdrive_file(
            url=url,
            source_id="file12345",
            temp_dir=self.tmp,
            config=cfg,
            progress_cb=lambda pct, spd, eta, dl, tot: progress_updates.append((pct, dl, tot)),
        )

        self.assertTrue(self.state["confirm_served"])
        self.assertTrue(
            len(self.state["range_headers_seen"]) >= 1,
            f"Expected Range header on resume, got {self.state['range_headers_seen']}",
        )
        expected_cut = int(len(self.payload) * 0.40)
        self.assertEqual(self.state["range_headers_seen"][0], f"bytes={expected_cut}-")

        result_path = Path(out_path)
        self.assertTrue(result_path.exists())
        self.assertEqual(result_path.suffix, ".mp3")
        actual_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
        self.assertEqual(actual_sha256, self.expected_sha256)
        self.assertFalse((self.tmp / "file12345.gdrive.part").exists())
        self.assertTrue(any(pct == 100.0 for pct, _, _ in progress_updates))

    def test_private_and_quota_html_errors(self) -> None:
        cfg = AppConfig(socket_timeout=5, max_retries=2)
        with self.assertRaises(GDrivePermissionError) as ctx_priv:
            download_gdrive_file(
                url=f"http://127.0.0.1:{self.port}/private",
                source_id="priv001",
                temp_dir=self.tmp,
                config=cfg,
            )
        self.assertIn("Anyone with the link", str(ctx_priv.exception))

        with self.assertRaises(GDriveQuotaError) as ctx_quota:
            download_gdrive_file(
                url=f"http://127.0.0.1:{self.port}/quota",
                source_id="quota001",
                temp_dir=self.tmp,
                config=cfg,
            )
        self.assertIn("quota", str(ctx_quota.exception).lower())

    def test_parse_confirm_url_variants_and_magic_bytes(self) -> None:
        classic = '<a id="uc-download-link" href="/uc?export=download&amp;confirm=abcd&amp;id=xyz">Download</a>'
        parsed = parse_gdrive_confirm_url(classic, "https://drive.google.com/uc?id=xyz")
        self.assertIn("confirm=abcd", parsed or "")

        wav_path = self.tmp / "sample.bin"
        _generate_tone_wav(wav_path, 0.2)
        self.assertEqual(detect_extension_from_magic(wav_path), ".wav")

    def test_gdrive_smart_saver_transcoded_and_range_fallback(self) -> None:
        cfg = AppConfig(download_mode="eco", socket_timeout=10, max_retries=2)
        seen_dl_formats: list[str] = []

        class FakeYDLTranscoded:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=False):
                if not download:
                    return {
                        "id": "realVid123",
                        "ext": "mp4",
                        "formats": [
                            {"format_id": "source", "format_note": "Original", "height": 1080, "vcodec": "h264", "acodec": "aac", "filesize": 500_000_000},
                            {"format_id": "18", "height": 360, "vcodec": "h264", "acodec": "aac", "filesize": 25_000_000},
                        ],
                    }
                seen_dl_formats.append(self.opts.get("format", ""))
                out_file = self.opts["outtmpl"].replace("%(ext)s", "mp4")
                Path(out_file).write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 512)
                return {"id": "realVid123", "ext": "mp4", "requested_downloads": [{"filepath": out_file}]}

        with mock.patch("lecture_transcriber.downloader.gdrive_dl.is_online", return_value=True), \
             mock.patch("yt_dlp.YoutubeDL", FakeYDLTranscoded), \
             mock.patch("requests.Session.get", side_effect=AssertionError("Should not hit Range HTTP when transcoded succeeds")):
            out = download_gdrive_file(
                url="https://drive.google.com/file/d/realVid123/view",
                source_id="realVid123",
                temp_dir=self.tmp,
                config=cfg,
            )
        self.assertTrue(Path(out).exists())
        self.assertEqual(len(seen_dl_formats), 1)
        self.assertIn("bestaudio", seen_dl_formats[0])
        self.assertIn("18", seen_dl_formats[0])

        # Fallback case 1: yt-dlp finds only 'source' format (no transcoded stream) -> seamless HTTP fallback!
        class FakeYDLOnlySource:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=False):
                if download:
                    raise AssertionError("Should not download via yt-dlp when only source exists")
                return {
                    "id": "audioSrc1",
                    "ext": "mp3",
                    "formats": [
                        {"format_id": "source", "vcodec": "none", "acodec": "mp3", "filesize": 8_000_000},
                    ],
                }

        fake_resp = mock.MagicMock()
        fake_resp.status_code = 200
        fake_resp.headers = {"Content-Type": "audio/mpeg", "Content-Length": str(len(self.payload))}
        fake_resp.raw = None
        fake_resp.iter_content.return_value = iter([self.payload])

        with mock.patch("lecture_transcriber.downloader.gdrive_dl.is_online", return_value=True), \
             mock.patch("yt_dlp.YoutubeDL", FakeYDLOnlySource), \
             mock.patch("requests.Session.get", return_value=fake_resp) as mock_get:
            out_fb1 = download_gdrive_file(
                url="https://drive.google.com/file/d/audioSrc1/view",
                source_id="audioSrc1",
                temp_dir=self.tmp,
                config=cfg,
            )
        self.assertEqual(mock_get.call_count, 1)
        self.assertTrue(Path(out_fb1).exists())
        self.assertEqual(Path(out_fb1).suffix, ".mp3")

        # Fallback case 2: yt-dlp raises error -> silent fallback to HTTP Range downloader
        class FakeYDLError:
            def __init__(self, opts):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=False):
                raise RuntimeError("yt-dlp extractor failed")

        fake_resp2 = mock.MagicMock()
        fake_resp2.status_code = 200
        fake_resp2.headers = {"Content-Type": "audio/mpeg", "Content-Length": str(len(self.payload))}
        fake_resp2.raw = None
        fake_resp2.iter_content.return_value = iter([self.payload])

        with mock.patch("lecture_transcriber.downloader.gdrive_dl.is_online", return_value=True), \
             mock.patch("yt_dlp.YoutubeDL", FakeYDLError), \
             mock.patch("requests.Session.get", return_value=fake_resp2) as mock_get2:
            out_fb2 = download_gdrive_file(
                url="https://drive.google.com/file/d/errSrc2/view",
                source_id="errSrc2",
                temp_dir=self.tmp,
                config=cfg,
            )
        self.assertEqual(mock_get2.call_count, 1)
        self.assertTrue(Path(out_fb2).exists())


class TestYouTubeAndAudioProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_check_ffmpeg_available(self) -> None:
        ok, msg = check_ffmpeg()
        self.assertTrue(ok, f"Expected FFmpeg installed in environment, got: {msg}")
        self.assertIn("ffmpeg", msg.lower())

    def test_ffmpeg_conversion_and_chunk_splitting_and_cleanup(self) -> None:
        raw_wav = self.tmp / "lecture_raw.wav"
        _generate_tone_wav(raw_wav, duration_seconds=12.0, sample_rate=16000)
        self.assertTrue(raw_wav.exists())

        info = probe_audio_info(raw_wav)
        self.assertTrue(info["has_audio"])
        self.assertFalse(info["has_video"])
        self.assertAlmostEqual(info["duration_seconds"], 12.0, delta=0.5)

        cfg = AppConfig(audio_bitrate="64k", max_chunk_minutes=35)
        # Split into 4-second chunks -> expect ~3 chunks covering 12s
        chunks = prepare_audio_for_transcription(
            raw_path=raw_wav,
            temp_dir=self.tmp,
            source_id="clip_test_12s",
            config=cfg,
            max_chunk_seconds=4,
        )

        # Raw input file must be deleted after conversion
        self.assertFalse(raw_wav.exists())
        self.assertGreaterEqual(len(chunks), 2)
        self.assertLessEqual(len(chunks), 4)
        self.assertEqual(chunks[0]["offset_seconds"], 0)

        total_chunk_dur = 0.0
        for idx, ch in enumerate(chunks):
            p = Path(ch["path"])
            self.assertTrue(p.exists())
            self.assertEqual(p.suffix, ".mp3")
            ch_info = probe_audio_info(p)
            self.assertTrue(ch_info["has_audio"])
            if idx > 0:
                self.assertGreater(ch["offset_seconds"], chunks[idx - 1]["offset_seconds"])
            total_chunk_dur += ch["duration_seconds"]

        self.assertAlmostEqual(total_chunk_dur, 12.0, delta=1.2)

        # Test merge_audio_chunks_to_mp3 concatenates multiple MP3 chunks cleanly
        merged_mp3 = self.tmp / "merged_out.mp3"
        merge_audio_chunks_to_mp3(chunks, merged_mp3, bitrate="32k")
        self.assertTrue(merged_mp3.exists())
        merged_info = probe_audio_info(merged_mp3)
        self.assertTrue(merged_info["has_audio"])
        self.assertAlmostEqual(merged_info["duration_seconds"], 12.0, delta=1.5)

        # Verify cleanup removes all generated chunks
        deleted = cleanup_audio_files(chunks)
        self.assertEqual(deleted, len(chunks))
        for ch in chunks:
            self.assertFalse(Path(ch["path"]).exists())

    def test_eco_direct_passthrough_vs_32k_transcode_and_video_detection(self) -> None:
        # Build a clean 32k pure-audio M4A via FFmpeg
        wav_src = self.tmp / "speech_src.wav"
        _generate_tone_wav(wav_src, duration_seconds=4.0, sample_rate=16000)
        m4a_file = self.tmp / "pure_audio.m4a"
        ffmpeg_bin = shutil.which("ffmpeg")
        assert ffmpeg_bin
        subprocess.run(
            [ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav_src), "-c:a", "aac", "-b:a", "32k", str(m4a_file)],
            check=True,
        )

        # 1. Eco mode with keep_audio=False -> zero-FFmpeg direct upload pass-through!
        eco_cfg = AppConfig(download_mode="eco", keep_audio=False)
        with mock.patch("lecture_transcriber.downloader.audio_processor._run_ffmpeg", side_effect=AssertionError("FFmpeg transcode should be skipped on direct passthrough")):
            chunks_pt = prepare_audio_for_transcription(
                raw_path=m4a_file,
                temp_dir=self.tmp,
                source_id="pt_m4a",
                config=eco_cfg,
            )
        self.assertEqual(len(chunks_pt), 1)
        self.assertTrue(chunks_pt[0].get("direct_passthrough"))
        self.assertTrue(Path(chunks_pt[0]["path"]).exists())
        self.assertEqual(Path(chunks_pt[0]["path"]).suffix, ".m4a")
        self.assertFalse(m4a_file.exists())

        # 2. Same M4A when keep_audio=True -> skips pass-through & transcodes to 32k MP3
        passthru_path = Path(chunks_pt[0]["path"])
        keep_cfg = AppConfig(download_mode="eco", keep_audio=True)
        seen_cmds: list[list[str]] = []
        orig_run = __import__("lecture_transcriber.downloader.audio_processor", fromlist=["_run_ffmpeg"])._run_ffmpeg

        def spy_run(cmd, cancel_event=None):
            seen_cmds.append(list(cmd))
            return orig_run(cmd, cancel_event=cancel_event)

        with mock.patch("lecture_transcriber.downloader.audio_processor._run_ffmpeg", side_effect=spy_run):
            chunks_keep = prepare_audio_for_transcription(
                raw_path=passthru_path,
                temp_dir=self.tmp,
                source_id="keep_m4a",
                config=keep_cfg,
            )
        self.assertEqual(len(chunks_keep), 1)
        self.assertFalse(chunks_keep[0].get("direct_passthrough", False))
        self.assertEqual(Path(chunks_keep[0]["path"]).suffix, ".mp3")
        self.assertTrue(any("-b:a" in cmd and "32k" in cmd for cmd in seen_cmds))

        # 3. Balanced mode -> uses 64k instead of 32k and does not pass-through
        wav2 = self.tmp / "bal_src.wav"
        _generate_tone_wav(wav2, duration_seconds=2.0, sample_rate=16000)
        bal_cfg = AppConfig(download_mode="balanced", audio_bitrate="64k")
        seen_bal: list[list[str]] = []
        with mock.patch("lecture_transcriber.downloader.audio_processor._run_ffmpeg", side_effect=lambda cmd, cancel_event=None: (seen_bal.append(list(cmd)), orig_run(cmd, cancel_event=cancel_event))[1]):
            chunks_bal = prepare_audio_for_transcription(
                raw_path=wav2,
                temp_dir=self.tmp,
                source_id="bal_test",
                config=bal_cfg,
            )
        self.assertEqual(Path(chunks_bal[0]["path"]).suffix, ".mp3")
        self.assertTrue(any("-b:a" in cmd and "64k" in cmd for cmd in seen_bal))

        # 4. MP4 with video stream -> probe_audio_info detects has_video=True -> strips video (-vn) at 32k
        vid_mp4 = self.tmp / "with_video.mp4"
        subprocess.run(
            [
                ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=black:s=160x120:d=2",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                "-c:v", "mpeg4", "-c:a", "aac", "-b:a", "32k",
                str(vid_mp4),
            ],
            check=True,
        )
        v_info = probe_audio_info(vid_mp4)
        self.assertTrue(v_info["has_audio"])
        self.assertTrue(v_info["has_video"])
        chunks_vid = prepare_audio_for_transcription(
            raw_path=vid_mp4,
            temp_dir=self.tmp,
            source_id="vid_mp4",
            config=eco_cfg,
        )
        self.assertEqual(Path(chunks_vid[0]["path"]).suffix, ".mp3")
        self.assertFalse(chunks_vid[0].get("direct_passthrough", False))

    def test_youtube_fallback_pause_and_cancel_part_handling(self) -> None:
        cfg = AppConfig(download_mode="eco", max_retries=5, socket_timeout=10)
        seen_formats: list[str] = []
        seen_clients: list[Any] = []

        class FakeYDLFallback:
            def __init__(self, opts):
                self.opts = opts
                seen_formats.append(opts.get("format", ""))
                seen_clients.append(opts.get("extractor_args"))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=True):
                # Fail first 3 audio-only attempts (android/ios, tv_embedded/web_safari/web, default)
                # so last-resort fallback (4th attempt) runs only after all audio-only profiles fail!
                if len(seen_formats) < 4:
                    raise RuntimeError("HTTP Error 403: Forbidden (SABR)")
                out_file = self.opts["outtmpl"].replace("%(ext)s", "mp4")
                Path(out_file).write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1024)
                for hook in self.opts.get("progress_hooks", []):
                    hook({"status": "finished", "filename": out_file, "downloaded_bytes": 1036, "total_bytes": 1036})
                return {"id": "yt123456789"}

        with mock.patch("lecture_transcriber.downloader.youtube_dl.is_online", return_value=True), \
             mock.patch("lecture_transcriber.downloader.youtube_dl._wait_interruptible"), \
             mock.patch("yt_dlp.YoutubeDL", FakeYDLFallback):
            out = download_youtube_audio(
                url="https://www.youtube.com/watch?v=yt123456789",
                source_id="yt123456789",
                temp_dir=self.tmp,
                config=cfg,
            )

        self.assertTrue(Path(out).exists())
        self.assertEqual(len(seen_formats), 4)
        self.assertEqual(seen_formats[0], ECO_AUDIO_FORMAT)
        self.assertEqual(seen_formats[1], ECO_AUDIO_FORMAT)
        self.assertEqual(seen_formats[2], ECO_AUDIO_FORMAT)
        self.assertEqual(seen_formats[3], "18/ba/b")
        self.assertIsNone(seen_clients[0])
        self.assertEqual(seen_clients[1], {"youtube": {"player_client": ["android", "ios"]}})
        self.assertEqual(seen_clients[2], {"youtube": {"player_client": ["tv_embedded", "web_safari", "web"]}})
        self.assertEqual(seen_clients[3], {"youtube": {"player_client": ["android", "web"]}})

        # Pause preserves .part file; cancel deletes .part file
        part_file = self.tmp / "ytpause1234.m4a.part"
        part_file.write_bytes(b"partial-data")

        class FakeYDLPauseCancel:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=True):
                for hook in self.opts.get("progress_hooks", []):
                    hook({"status": "downloading", "downloaded_bytes": 100, "total_bytes": 1000})
                return {}

        pause_ev = threading.Event()
        pause_ev.set()
        with mock.patch("lecture_transcriber.downloader.youtube_dl.is_online", return_value=True), \
             mock.patch("yt_dlp.YoutubeDL", FakeYDLPauseCancel):
            with self.assertRaises(DownloadPausedError):
                download_youtube_audio(
                    url="https://www.youtube.com/watch?v=ytpause1234",
                    source_id="ytpause1234",
                    temp_dir=self.tmp,
                    config=cfg,
                    pause_event=pause_ev,
                )
        self.assertTrue(part_file.exists(), ".part file must be preserved on pause")

        cancel_ev = threading.Event()
        cancel_ev.set()
        with mock.patch("lecture_transcriber.downloader.youtube_dl.is_online", return_value=True), \
             mock.patch("yt_dlp.YoutubeDL", FakeYDLPauseCancel):
            with self.assertRaises(DownloadCancelledError):
                download_youtube_audio(
                    url="https://www.youtube.com/watch?v=ytpause1234",
                    source_id="ytpause1234",
                    temp_dir=self.tmp,
                    config=cfg,
                    cancel_event=cancel_ev,
                )
        self.assertFalse(part_file.exists(), ".part file must be cleaned up on cancel")

    def test_youtube_balanced_and_strict_audio_refusal(self) -> None:
        # Balanced mode uses BALANCED_AUDIO_FORMAT on audio-only clients before 18/ba/b
        bal_formats: list[str] = []

        class FakeYDLBalanced:
            def __init__(self, opts):
                self.opts = opts
                bal_formats.append(opts.get("format", ""))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=True):
                if len(bal_formats) == 1:
                    raise RuntimeError("403 Forbidden")
                out_file = self.opts["outtmpl"].replace("%(ext)s", "m4a")
                Path(out_file).write_bytes(b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 512)
                return {"id": "bal001", "requested_downloads": [{"filepath": out_file}]}

        with mock.patch("lecture_transcriber.downloader.youtube_dl.is_online", return_value=True), \
             mock.patch("lecture_transcriber.downloader.youtube_dl._wait_interruptible"), \
             mock.patch("yt_dlp.YoutubeDL", FakeYDLBalanced):
            out_bal = download_youtube_audio(
                url="https://www.youtube.com/watch?v=bal001",
                source_id="bal001",
                temp_dir=self.tmp,
                config=AppConfig(download_mode="balanced", max_retries=4),
            )
        self.assertTrue(Path(out_bal).exists())
        self.assertEqual(bal_formats, [BALANCED_AUDIO_FORMAT, BALANCED_AUDIO_FORMAT])

        # Strict Audio mode: tries all 3 audio-only profiles, NEVER tries 18/ba/b, and raises DownloadError
        strict_formats: list[str] = []

        class FakeYDLStrictFail:
            def __init__(self, opts):
                strict_formats.append(opts.get("format", ""))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=True):
                raise RuntimeError("ERROR: Requested format is not available")

        with mock.patch("lecture_transcriber.downloader.youtube_dl.is_online", return_value=True), \
             mock.patch("lecture_transcriber.downloader.youtube_dl._wait_interruptible"), \
             mock.patch("yt_dlp.YoutubeDL", FakeYDLStrictFail):
            with self.assertRaises(DownloadError) as ctx:
                download_youtube_audio(
                    url="https://www.youtube.com/watch?v=strict001",
                    source_id="strict001",
                    temp_dir=self.tmp,
                    config=AppConfig(download_mode="strict_audio", max_retries=8),
                )
        self.assertEqual(len(strict_formats), 3)
        self.assertTrue(all(fmt == STRICT_AUDIO_FORMAT for fmt in strict_formats))
        self.assertFalse(any("18" in fmt.split("/") or "b" in fmt.split("/") for fmt in strict_formats))
        self.assertIn("Strict Audio", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
