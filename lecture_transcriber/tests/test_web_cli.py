"""Integration tests for the zero-dependency Web UI server, SSE stream, and CLI entrypoints."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import requests

from lecture_transcriber.config import ConfigManager
from lecture_transcriber.database import Database
from lecture_transcriber.downloader import DownloadPausedError, ParsedMediaItem
from lecture_transcriber.engine import TranscriptionEngine
from lecture_transcriber.network import NetworkStatusMonitor
from lecture_transcriber.web.server import create_server

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = Path(__file__).resolve().parents[1]


class TestWebServerAndSSE(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.secret_key = "TEST_DUMMY_INTEGRATION_KEY_987654321"

        self.cfg_mgr = ConfigManager(self.tmp / "config.json")
        self.cfg_mgr.update(
            gemini_api_key=self.secret_key,
            output_base_dir=str(self.tmp / "output"),
            temp_dir=str(self.tmp / "temp"),
            download_workers=2,
            transcribe_workers=2,
        )
        self.db = Database(self.tmp / "state.db")

        self.dl_gate = threading.Event()
        self.dl_started = threading.Event()

        def fake_expand(urls, config=None, fetch_metadata=True, include_playlist=False):
            items = []
            for idx, u in enumerate(urls, start=1):
                sid = u.rstrip("/").split("/")[-1].replace("watch?v=", "")
                items.append(
                    ParsedMediaItem(
                        source_type="gdrive" if "drive.google.com" in u else "youtube",
                        url=u,
                        source_id=sid,
                        title=f"محاضرة {sid}",
                        seq_num=idx,
                    )
                )
            return items

        def fake_download(source_type, url, source_id, temp_dir, config=None, progress_cb=None, cancel_event=None, pause_event=None, on_status=None):
            self.dl_started.set()
            if progress_cb:
                progress_cb(25.0, 128_000.0, 10.0, 256, 1024)
            # Wait for gate or pause/cancel
            deadline = time.monotonic() + 5.0
            while not self.dl_gate.is_set() and time.monotonic() < deadline:
                if pause_event and pause_event.is_set():
                    raise DownloadPausedError("paused")
                if cancel_event and cancel_event.is_set():
                    from lecture_transcriber.downloader import DownloadCancelledError
                    raise DownloadCancelledError("cancelled")
                time.sleep(0.02)
            raw = Path(temp_dir) / f"{source_id}.mp3"
            raw.write_bytes(b"ID3" + b"\x00" * 512)
            if progress_cb:
                progress_cb(100.0, 128_000.0, 0.0, 1024, 1024)
            return str(raw)

        def fake_prepare(raw_path, temp_dir, source_id, config=None, cancel_event=None):
            if os.path.isfile(raw_path):
                os.remove(raw_path)
            chunk = Path(temp_dir) / f"{source_id}_chunk_000.mp3"
            chunk.write_bytes(b"ID3" + b"\x00" * 512)
            return [{"path": str(chunk), "offset_seconds": 0, "duration_seconds": 45.0}]

        class FakeTranscriber:
            def transcribe_chunks(self, chunks, **kwargs):
                return "[00:00] بسم الله الرحمن الرحيم، هذا نص التفريغ التجريبي."

        self._net_patch1 = mock.patch("lecture_transcriber.network.is_online", return_value=True)
        self._net_patch2 = mock.patch("lecture_transcriber.engine.is_online", return_value=True)
        self._net_patch1.start()
        self._net_patch2.start()

        self.monitor = NetworkStatusMonitor(check_interval=10.0)
        self.engine = TranscriptionEngine(
            self.cfg_mgr,
            self.db,
            transcriber_factory=lambda cfg: FakeTranscriber(),
            network_monitor=self.monitor,
            download_fn=fake_download,
            prepare_audio_fn=fake_prepare,
            expand_urls_fn=fake_expand,
        )
        self.engine.start()

        self.server = create_server(
            self.engine,
            self.cfg_mgr,
            self.db,
            host="127.0.0.1",
            port=0,
            ffmpeg_status_fn=lambda: {"available": True, "version": "ffmpeg test-build"},
        )
        self.port = self.server.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.1},
            daemon=True,
        )
        self.server_thread.start()

    def tearDown(self) -> None:
        self.dl_gate.set()
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=3.0)
        self.engine.stop()
        self.db.close()
        self._net_patch2.stop()
        self._net_patch1.stop()
        self._tmpdir.cleanup()

    def test_index_html_offline_arabic_rtl(self) -> None:
        resp = requests.get(f"{self.base_url}/", timeout=5)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers.get("Content-Type", ""))
        html = resp.text
        self.assertIn('dir="rtl"', html)
        self.assertIn('lang="ar"', html)
        self.assertIn("quickModeSelect", html)
        self.assertIn("cfgDownloadMode", html)
        self.assertIn("cfgKeepAudio", html)
        self.assertIn("cfgModels", html)
        self.assertIn("ترتيب نماذج الذكاء الاصطناعي", html)
        self.assertIn("محفوظ تلقائياً على جهازك", html)
        self.assertIn("وضع التوفير", html)
        self.assertIn("الاحتفاظ بملفات الصوت", html)
        # Ensure zero external CDN scripts or stylesheets
        self.assertIsNone(re.search(r'<script[^>]+src=["\']https?://', html, re.IGNORECASE))
        self.assertIsNone(re.search(r'<link[^>]+href=["\']https?://', html, re.IGNORECASE))

    def test_state_and_settings_endpoints_mask_api_key(self) -> None:
        r = requests.get(f"{self.base_url}/api/state", timeout=5)
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["online"])
        self.assertTrue(data["ffmpeg"]["available"])
        self.assertTrue(data["config"]["has_api_key"])
        self.assertEqual(data["config"]["download_mode"], "eco")
        self.assertFalse(data["config"]["keep_audio"])
        self.assertNotIn(self.secret_key, r.text)

        # Invalid settings rejected cleanly with 400
        bad = requests.post(
            f"{self.base_url}/api/settings",
            json={"audio_bitrate": "not_a_bitrate"},
            timeout=5,
        )
        self.assertEqual(bad.status_code, 400)

        bad_mode = requests.post(
            f"{self.base_url}/api/settings",
            json={"download_mode": "turbo_invalid"},
            timeout=5,
        )
        self.assertEqual(bad_mode.status_code, 400)

        bad_keep = requests.post(
            f"{self.base_url}/api/settings",
            json={"keep_audio": "maybe"},
            timeout=5,
        )
        self.assertEqual(bad_keep.status_code, 400)

        bad_models = requests.post(
            f"{self.base_url}/api/settings",
            json={"models": "   , \n  "},
            timeout=5,
        )
        self.assertEqual(bad_models.status_code, 400)

        # Valid settings update persists & masks new key
        new_key = "TEST_DUMMY_UPDATED_KEY_555555555"
        upd = requests.post(
            f"{self.base_url}/api/settings",
            json={
                "gemini_api_key": new_key,
                "download_workers": 4,
                "transcribe_workers": 3,
                "audio_bitrate": "96k",
                "download_mode": "strict_audio",
                "keep_audio": True,
                "proxy": "http://127.0.0.1:9090",
                "models": "gemini-3.6-flash, gemini-3.5-flash\ngemini-3.8-flash",
            },
            timeout=5,
        )
        self.assertEqual(upd.status_code, 200)
        upd_data = upd.json()
        self.assertTrue(upd_data["ok"])
        self.assertNotIn(new_key, upd.text)
        self.assertEqual(upd_data["config"]["download_workers"], 4)
        self.assertEqual(upd_data["config"]["audio_bitrate"], "96k")
        self.assertEqual(upd_data["config"]["download_mode"], "strict_audio")
        self.assertTrue(upd_data["config"]["keep_audio"])
        self.assertEqual(
            upd_data["config"]["models"],
            ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.8-flash"],
        )
        self.assertEqual(self.cfg_mgr.get().gemini_api_key, new_key)
        self.assertEqual(self.cfg_mgr.get().download_mode, "strict_audio")
        self.assertTrue(self.cfg_mgr.get().keep_audio)
        self.assertEqual(
            self.cfg_mgr.get().models,
            ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.8-flash"],
        )

        # Updating models as a JSON list also works
        upd_list = requests.post(
            f"{self.base_url}/api/settings",
            json={"models": ["gemini-3.6-flash", "gemini-3.1-flash-lite", "gemini-3.6-flash"]},
            timeout=5,
        )
        self.assertEqual(upd_list.status_code, 200)
        self.assertEqual(
            upd_list.json()["config"]["models"],
            ["gemini-3.6-flash", "gemini-3.1-flash-lite"],
        )

    def test_enqueue_pause_resume_cancel_transcript_open_folder_and_delete(self) -> None:
        # Enqueue while download is gated so we can test pause/resume/pause_all/resume_all
        enq = requests.post(
            f"{self.base_url}/api/enqueue",
            json={
                "course_name": "علوم القرآن",
                "urls_text": "https://youtu.be/lec001\nhttps://drive.google.com/file/d/lec002/view",
            },
            timeout=5,
        )
        self.assertEqual(enq.status_code, 200)
        enq_data = enq.json()
        self.assertTrue(enq_data["ok"])
        self.assertEqual(enq_data["added"], 2)
        jid1, jid2 = enq_data["job_ids"]

        self.assertTrue(self.dl_started.wait(timeout=3.0))

        # Pause job 1
        p1 = requests.post(f"{self.base_url}/api/jobs/{jid1}/pause", json={}, timeout=5)
        self.assertEqual(p1.status_code, 200)
        self.assertTrue(p1.json()["ok"])

        # Pause all & Resume all
        pa = requests.post(f"{self.base_url}/api/pause_all", json={}, timeout=5)
        self.assertEqual(pa.status_code, 200)
        self.assertTrue(pa.json()["ok"])

        # Cancel job 2 while paused
        c2 = requests.post(f"{self.base_url}/api/jobs/{jid2}/cancel", json={}, timeout=5)
        self.assertEqual(c2.status_code, 200)
        self.assertEqual(c2.json()["job"]["status"], "cancelled")

        # Transcript not ready yet -> 404
        t_miss = requests.get(f"{self.base_url}/api/jobs/{jid1}/transcript", timeout=5)
        self.assertEqual(t_miss.status_code, 404)

        # Open gate and resume job 1 -> finishes!
        self.dl_gate.set()
        res1 = requests.post(f"{self.base_url}/api/jobs/{jid1}/resume", json={}, timeout=5)
        self.assertEqual(res1.status_code, 200)

        # Wait for job 1 to finish
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if self.db.get_job(jid1)["status"] == "done":
                break
            time.sleep(0.05)
        self.assertEqual(self.db.get_job(jid1)["status"], "done")

        # Fetch transcript preview via REST API
        tr = requests.get(f"{self.base_url}/api/jobs/{jid1}/transcript", timeout=5)
        self.assertEqual(tr.status_code, 200)
        tr_data = tr.json()
        self.assertTrue(tr_data["ok"])
        self.assertIn("بسم الله الرحمن الرحيم", tr_data["content"])

        # Open folder safely (mock OS launcher)
        opened_paths: list[str] = []
        with mock.patch(
            "lecture_transcriber.web.server._open_folder_in_os",
            side_effect=lambda p: opened_paths.append(p),
        ):
            of = requests.post(
                f"{self.base_url}/api/open_folder",
                json={"course_id": enq_data["course"]["id"]},
                timeout=5,
            )
        self.assertEqual(of.status_code, 200)
        self.assertEqual(len(opened_paths), 1)

        # Delete job 2
        d2 = requests.delete(f"{self.base_url}/api/jobs/{jid2}", timeout=5)
        self.assertEqual(d2.status_code, 200)
        self.assertIsNone(self.db.get_job(jid2))

    def test_sse_events_headers_snapshot_and_live_updates(self) -> None:
        import http.client

        self.dl_gate.set()
        received_events: list[dict] = []
        stream_ready = threading.Event()
        stop_reading = threading.Event()
        content_type_holder: list[str] = []

        def _sse_reader() -> None:
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=6.0)
            try:
                conn.request("GET", "/api/events")
                resp = conn.getresponse()
                content_type_holder.append(resp.getheader("Content-Type") or "")
                while not stop_reading.is_set():
                    raw_line = resp.fp.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        payload = json.loads(line[5:].strip())
                        received_events.append(payload)
                        if payload.get("type") == "snapshot":
                            stream_ready.set()
                        if (
                            payload.get("type") == "job_update"
                            and payload.get("data", {}).get("job", {}).get("status") == "done"
                        ):
                            break
            finally:
                conn.close()

        reader = threading.Thread(target=_sse_reader, daemon=True)
        reader.start()

        self.assertTrue(stream_ready.wait(timeout=3.0), "Did not receive initial SSE snapshot")
        self.assertTrue(content_type_holder and "text/event-stream" in content_type_holder[0])

        # Trigger live events via enqueue
        r = requests.post(
            f"{self.base_url}/api/enqueue",
            json={"course_name": "SSE Course", "urls_text": "https://youtu.be/ssevid01"},
            timeout=5,
        )
        self.assertEqual(r.status_code, 200)

        reader.join(timeout=6.0)
        stop_reading.set()

        types_seen = [ev.get("type") for ev in received_events]
        self.assertIn("snapshot", types_seen)
        self.assertIn("job_update", types_seen)
        self.assertTrue(
            any(
                ev.get("type") == "job_update" and ev.get("data", {}).get("job", {}).get("status") == "done"
                for ev in received_events
            ),
            f"Expected SSE job_update with status=done, got types={types_seen}",
        )


class TestCLIEntrypoints(unittest.TestCase):
    def test_run_py_help_from_package_dir_and_repo_root(self) -> None:
        proc1 = subprocess.run(
            [sys.executable, "run.py", "--help"],
            cwd=str(PKG_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        self.assertEqual(proc1.returncode, 0, proc1.stderr)
        self.assertIn("--cli", proc1.stdout)
        self.assertIn("--course", proc1.stdout)
        self.assertIn("--urls", proc1.stdout)
        self.assertIn("--mode", proc1.stdout)
        self.assertIn("--keep-audio", proc1.stdout)

        proc2 = subprocess.run(
            [sys.executable, str(PKG_DIR / "run.py"), "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        self.assertEqual(proc2.returncode, 0, proc2.stderr)
        self.assertIn("--cli", proc2.stdout)

    def test_cli_module_help_and_mode_flags(self) -> None:
        from lecture_transcriber.cli import build_arg_parser

        proc = subprocess.run(
            [sys.executable, "-m", "lecture_transcriber.cli", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--course", proc.stdout)
        self.assertIn("--file", proc.stdout)
        self.assertIn("--resume-only", proc.stdout)
        self.assertIn("--mode", proc.stdout)
        self.assertIn("--keep-audio", proc.stdout)

        parser = build_arg_parser()
        parsed = parser.parse_args(
            ["--course", "Fiqh", "--urls", "https://youtu.be/abc", "--mode", "strict_audio", "--keep-audio"]
        )
        self.assertEqual(parsed.mode, "strict_audio")
        self.assertTrue(parsed.keep_audio)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--course", "Fiqh", "--urls", "https://youtu.be/abc", "--mode", "bad_mode"])


if __name__ == "__main__":
    unittest.main()
