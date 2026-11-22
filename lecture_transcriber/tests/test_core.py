"""Tests for core persistence, config, filename sanitization, and network watchdog."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from lecture_transcriber.config import (
    DEFAULT_MODELS,
    VALID_DOWNLOAD_MODES,
    AppConfig,
    ConfigManager,
    clean_filename,
    ensure_utf8_stdio,
)
from lecture_transcriber.database import ACTIVE_STATUSES, VALID_STATUSES, Database
from lecture_transcriber.network import (
    NetworkStatusMonitor,
    call_with_retry,
    is_online,
    retry_with_backoff,
    wait_for_internet,
)


class TestCleanFilename(unittest.TestCase):
    def test_arabic_and_illegal_windows_chars(self) -> None:
        raw = 'المحاضرة 1: شرح "المقدمة" <فقه> | *مهم*؟ / \\'
        cleaned = clean_filename(raw)
        for ch in '<>:"/\\|?*':
            self.assertNotIn(ch, cleaned)
        self.assertIn("المحاضرة", cleaned)
        self.assertIn("المقدمة", cleaned)
        self.assertIn("فقه", cleaned)
        self.assertIn("مهم", cleaned)

    def test_windows_reserved_names(self) -> None:
        for reserved in ("CON", "con", "PRN", "AUX", "NUL", "COM1", "COM9", "LPT1", "LPT9"):
            cleaned = clean_filename(reserved)
            self.assertNotEqual(cleaned.upper(), reserved.upper())
            self.assertTrue(cleaned.startswith("_"))

        # Reserved names with extensions
        self.assertTrue(clean_filename("CON.txt").startswith("_"))
        self.assertTrue(clean_filename("nul.md").startswith("_"))
        self.assertTrue(clean_filename("LPT3.mp3").startswith("_"))

    def test_length_limit_and_trailing_dots(self) -> None:
        long_arabic = "محاضرة_تفسير_" * 30
        cleaned = clean_filename(long_arabic, max_length=80)
        self.assertLessEqual(len(cleaned), 80)
        self.assertFalse(cleaned.endswith((" ", ".", "_", "-")))

        self.assertEqual(clean_filename("...  ___  ..."), "untitled")
        self.assertEqual(clean_filename(""), "untitled")
        self.assertEqual(clean_filename(None), "untitled")  # type: ignore[arg-type]

    def test_control_and_bidi_characters_removed(self) -> None:
        raw = "درس\u200f\u202e\x00\x1fالأول"
        cleaned = clean_filename(raw)
        self.assertNotIn("\u200f", cleaned)
        self.assertNotIn("\u202e", cleaned)
        self.assertNotIn("\x00", cleaned)
        self.assertIn("درس", cleaned)
        self.assertIn("الأول", cleaned)

    def test_ensure_utf8_stdio_safe(self) -> None:
        ensure_utf8_stdio()
        self.assertEqual(os.environ.get("PYTHONIOENCODING"), "utf-8")


class TestConfigManager(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.cfg_path = self.tmp / "data" / "config.json"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_defaults_and_persistence(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            mgr = ConfigManager(self.cfg_path)
            cfg = mgr.get()
            self.assertEqual(cfg.download_workers, 3)
            self.assertEqual(cfg.transcribe_workers, 1)
            self.assertEqual(cfg.audio_bitrate, "64k")
            self.assertEqual(cfg.download_mode, "eco")
            self.assertFalse(cfg.keep_audio)
            self.assertEqual(cfg.effective_audio_bitrate(), "32k")
            self.assertEqual(cfg.max_chunk_minutes, 12)
            self.assertEqual(cfg.models, DEFAULT_MODELS)
            self.assertEqual(cfg.port, 8765)

            updated = mgr.update(
                gemini_api_key="AIzaSyTestKey1234567890",
                download_workers=4,
                download_mode="strict_audio",
                keep_audio=True,
                output_base_dir=str(self.tmp / "out"),
                temp_dir=str(self.tmp / "temp"),
                proxy="http://127.0.0.1:8080",
            )
            self.assertEqual(updated.download_workers, 4)
            self.assertEqual(updated.download_mode, "strict_audio")
            self.assertTrue(updated.keep_audio)
            self.assertEqual(updated.proxy, "http://127.0.0.1:8080")
            self.assertTrue(self.cfg_path.exists())

            # Reload from disk in a fresh manager
            mgr2 = ConfigManager(self.cfg_path)
            reloaded = mgr2.get()
            self.assertEqual(reloaded.gemini_api_key, "AIzaSyTestKey1234567890")
            self.assertEqual(reloaded.download_workers, 4)
            self.assertEqual(reloaded.download_mode, "strict_audio")
            self.assertTrue(reloaded.keep_audio)
            self.assertEqual(reloaded.proxy, "http://127.0.0.1:8080")

            # Masked key serialization and preservation on update
            masked = reloaded.to_dict(mask_api_key=True)
            self.assertTrue(masked["has_api_key"])
            self.assertIn("*", masked["gemini_api_key"])
            self.assertNotEqual(masked["gemini_api_key"], "AIzaSyTestKey1234567890")

            mgr2.update(gemini_api_key=masked["gemini_api_key"], port=9000)
            after = mgr2.get()
            self.assertEqual(after.gemini_api_key, "AIzaSyTestKey1234567890")
            self.assertEqual(after.port, 9000)

            mgr2.ensure_directories()
            self.assertTrue((self.tmp / "out").is_dir())
            self.assertTrue((self.tmp / "temp").is_dir())

    def test_env_fallback_and_corrupt_json_recovery(self) -> None:
        self.cfg_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg_path.write_text("{corrupt json", encoding="utf-8")
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "env-secret-key-9999"}):
            mgr = ConfigManager(self.cfg_path)
            cfg = mgr.get()
            self.assertEqual(cfg.effective_api_key(), "env-secret-key-9999")
            self.assertEqual(cfg.gemini_api_key, "env-secret-key-9999")

    def test_unknown_field_raises(self) -> None:
        mgr = ConfigManager(self.cfg_path)
        with self.assertRaises(ValueError):
            mgr.update(nonexistent_field=123)

    def test_download_mode_and_keep_audio_validation(self) -> None:
        self.assertEqual(VALID_DOWNLOAD_MODES, ("eco", "balanced", "strict_audio"))
        self.assertEqual(AppConfig(download_mode="INVALID_MODE").download_mode, "eco")
        self.assertEqual(AppConfig(download_mode="BALANCED").download_mode, "balanced")
        self.assertEqual(AppConfig(download_mode="balanced").effective_audio_bitrate(), "64k")
        self.assertEqual(AppConfig(download_mode="eco", audio_bitrate="24k").effective_audio_bitrate(), "24k")
        self.assertEqual(AppConfig(download_mode="eco", audio_bitrate="48k").effective_audio_bitrate(), "48k")
        self.assertFalse(AppConfig(keep_audio="false").keep_audio)
        self.assertTrue(AppConfig(keep_audio="true").keep_audio)

        mgr = ConfigManager(self.cfg_path)
        with self.assertRaises(ValueError):
            mgr.update(download_mode="ultra_invalid")


class TestDatabase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "state.db"
        self.db = Database(self.db_path)

    def tearDown(self) -> None:
        self.db.close()
        self._tmpdir.cleanup()

    def test_wal_mode_enabled(self) -> None:
        row = self.db._conn.execute("PRAGMA journal_mode;").fetchone()
        self.assertEqual(row[0].lower(), "wal")

    def test_course_and_job_crud_and_state_transitions(self) -> None:
        c1 = self.db.create_or_get_course("فقه المعاملات", "C:/out/fiqh")
        c2 = self.db.create_or_get_course("فقه المعاملات", "C:/out/other")
        self.assertEqual(c1["id"], c2["id"])
        self.assertEqual(len(self.db.list_courses()), 1)

        job = self.db.add_job(
            course_id=c1["id"],
            source_type="youtube",
            url="https://youtu.be/abc12345678",
            source_id="abc12345678",
            title="الدرس الأول",
            seq_num=1,
        )
        self.assertEqual(job["status"], "queued")
        self.assertEqual(self.db.get_job(job["id"])["title"], "الدرس الأول")
        self.assertIsNotNone(
            self.db.find_job_by_source(c1["id"], "youtube", "abc12345678")
        )

        # Update progress & speed
        updated = self.db.update_job(
            job["id"],
            status="downloading",
            stage="downloading",
            progress=42.5,
            speed_bps=125000.0,
            eta_seconds=60,
            bytes_downloaded=500000,
            total_bytes=1200000,
        )
        self.assertEqual(updated["status"], "downloading")
        self.assertAlmostEqual(updated["progress"], 42.5)

        # Pause -> Resume
        paused = self.db.pause_job(job["id"])
        self.assertEqual(paused["status"], "paused")
        self.assertEqual(paused["speed_bps"], 0.0)
        resumed = self.db.resume_job(job["id"])
        self.assertEqual(resumed["status"], "queued")

        # Fail -> Retry
        self.db.update_job(job["id"], status="failed", error_message="timeout", retries=3)
        retried = self.db.retry_job(job["id"])
        self.assertEqual(retried["status"], "queued")
        self.assertEqual(retried["retries"], 0)
        self.assertEqual(retried["error_message"], "")

        # Cancel -> Done protection
        cancelled = self.db.cancel_job(job["id"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.db.update_job(job["id"], status="done", progress=100.0)
        self.assertEqual(self.db.pause_job(job["id"])["status"], "done")
        self.assertEqual(self.db.cancel_job(job["id"])["status"], "done")
        self.assertEqual(self.db.retry_job(job["id"])["status"], "done")

        # Delete
        self.assertTrue(self.db.delete_job(job["id"]))
        self.assertIsNone(self.db.get_job(job["id"]))
        self.assertFalse(self.db.delete_job(job["id"]))

    def test_reset_interrupted_jobs_crash_recovery(self) -> None:
        course = self.db.create_or_get_course("CrashTest", "C:/out/crash")
        created_by_status: dict[str, str] = {}
        for idx, st in enumerate(VALID_STATUSES, start=1):
            j = self.db.add_job(
                course_id=course["id"],
                source_type="gdrive",
                url=f"https://drive.google.com/file/d/id{idx}/view",
                source_id=f"id{idx}",
                title=f"Lecture {idx}",
                seq_num=idx,
                status=st,
                speed_bps=9999.0,
                eta_seconds=120,
            )
            created_by_status[st] = j["id"]

        reset_count = self.db.reset_interrupted_jobs()
        self.assertEqual(reset_count, len(ACTIVE_STATUSES))

        for st, jid in created_by_status.items():
            job = self.db.get_job(jid)
            if st in ACTIVE_STATUSES:
                self.assertEqual(job["status"], "queued")
                self.assertEqual(job["stage"], "queued")
                self.assertEqual(job["speed_bps"], 0.0)
                self.assertIsNone(job["eta_seconds"])
            else:
                self.assertEqual(job["status"], st)

        stats = self.db.get_stats(course["id"])
        self.assertEqual(stats["total"], len(VALID_STATUSES))
        self.assertEqual(stats["active"], 0)
        self.assertEqual(stats["queued"], 1 + len(ACTIVE_STATUSES))

    def test_wal_multithreaded_concurrency(self) -> None:
        course = self.db.create_or_get_course("ConcurrencyCourse", "C:/out/conc")
        errors: list[Exception] = []

        def worker(worker_idx: int) -> None:
            try:
                # Mix shared Database instance and independent WAL connection
                local_db = Database(self.db_path) if worker_idx % 2 == 0 else None
                target = local_db or self.db
                try:
                    for i in range(15):
                        j = target.add_job(
                            course_id=course["id"],
                            source_type="youtube",
                            url=f"https://youtu.be/w{worker_idx}_{i}",
                            source_id=f"w{worker_idx}_{i}",
                            title=f"Worker {worker_idx} Item {i}",
                            seq_num=worker_idx * 100 + i,
                        )
                        target.update_job(j["id"], status="downloading", progress=50.0)
                        target.update_job(j["id"], status="done", progress=100.0)
                        _ = target.list_jobs(course_id=course["id"])
                finally:
                    if local_db is not None:
                        local_db.close()
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker, i) for i in range(8)]
            for f in futures:
                f.result()

        self.assertEqual(errors, [])
        self.assertEqual(len(self.db.list_jobs(course_id=course["id"])), 120)
        self.assertEqual(self.db.get_stats(course["id"])["done"], 120)


class TestNetworkUtilities(unittest.TestCase):
    def test_is_online_handles_socket_errors(self) -> None:
        with mock.patch("socket.create_connection", side_effect=OSError("network down")):
            self.assertFalse(is_online(timeout=0.5))

        dummy_sock = mock.MagicMock()
        dummy_sock.__enter__.return_value = dummy_sock
        with mock.patch("socket.create_connection", return_value=dummy_sock):
            self.assertTrue(is_online(timeout=0.5))

    def test_wait_for_internet_outage_then_recovery(self) -> None:
        states = iter([False, False, True])
        statuses: list[bool] = []

        with mock.patch(
            "lecture_transcriber.network.is_online",
            side_effect=lambda timeout=3.0: next(states),
        ):
            ok = wait_for_internet(
                check_interval=0.05,
                on_status=lambda online, elapsed: statuses.append(online),
            )
        self.assertTrue(ok)
        self.assertEqual(statuses, [False, False, True])

    def test_wait_for_internet_aborts_on_stop_event(self) -> None:
        stop = threading.Event()
        timer = threading.Timer(0.12, stop.set)
        timer.start()
        try:
            with mock.patch("lecture_transcriber.network.is_online", return_value=False):
                ok = wait_for_internet(stop_event=stop, check_interval=0.05)
            self.assertFalse(ok)
        finally:
            timer.cancel()

    def test_retry_with_backoff_simulated_disconnect_and_recovery(self) -> None:
        attempts = {"count": 0}
        retry_events: list[tuple[int, str]] = []
        online_checks = iter([False, False, True, True])

        @retry_with_backoff(
            max_attempts=4,
            base_delay=0.01,
            max_delay=0.02,
            retryable_exceptions=(ConnectionError, TimeoutError),
            check_interval=0.02,
            on_retry=lambda attempt, exc, delay: retry_events.append((attempt, str(exc))),
        )
        def flaky() -> str:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ConnectionError("socket reset")
            if attempts["count"] == 2:
                raise TimeoutError("read timeout")
            return "recovered"

        with mock.patch(
            "lecture_transcriber.network.is_online",
            side_effect=lambda timeout=3.0: next(online_checks, True),
        ):
            result = flaky()

        self.assertEqual(result, "recovered")
        self.assertEqual(attempts["count"], 3)
        self.assertEqual(len(retry_events), 2)

    def test_non_retryable_exception_raises_immediately(self) -> None:
        calls = {"count": 0}

        def bad() -> None:
            calls["count"] += 1
            raise ValueError("permission denied")

        with self.assertRaises(ValueError):
            call_with_retry(
                bad,
                max_attempts=5,
                base_delay=0.01,
                retryable_exceptions=(Exception,),
                non_retryable_exceptions=(ValueError,),
            )
        self.assertEqual(calls["count"], 1)

    def test_network_status_monitor_transitions(self) -> None:
        states = iter([True, False, False, True])
        transitions: list[bool] = []
        done_evt = threading.Event()

        def on_change(online: bool) -> None:
            transitions.append(online)
            if len(transitions) >= 2:
                done_evt.set()

        with mock.patch(
            "lecture_transcriber.network.is_online",
            side_effect=lambda timeout=3.0: next(states, True),
        ):
            mon = NetworkStatusMonitor(check_interval=0.05, timeout=0.1, on_change=on_change)
            mon.start()
            try:
                self.assertTrue(done_evt.wait(timeout=2.0))
                self.assertEqual(transitions[:2], [False, True])
                self.assertTrue(mon.is_online)
                snap = mon.status_dict()
                self.assertTrue(snap["online"])
                self.assertIsNone(snap["offline_since"])
            finally:
                mon.stop()


if __name__ == "__main__":
    unittest.main()
