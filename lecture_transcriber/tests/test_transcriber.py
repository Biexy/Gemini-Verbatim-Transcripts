"""Tests for GeminiTranscriber, postprocessor, and TranscriptionEngine."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from lecture_transcriber.config import AppConfig, ConfigManager
from lecture_transcriber.database import Database
from lecture_transcriber.downloader import DownloadPausedError, ParsedMediaItem, probe_audio_info
from lecture_transcriber.engine import TranscriptionEngine
from lecture_transcriber.network import NetworkStatusMonitor
from lecture_transcriber.transcriber import (
    GeminiAuthError,
    GeminiQuotaError,
    GeminiTranscriber,
    TranscriptionCancelledError,
    clean_transcript_body,
    format_lecture_markdown,
    rebuild_course_artifacts,
    shift_timestamps,
)


class TestPostprocessor(unittest.TestCase):
    def test_shift_timestamps_minutes_hours_and_ranges(self) -> None:
        text = (
            "[00:10] مقدمة الدرس الأول.\n\n"
            "[05:45 - 06:20] شرح المسألة الأولى.\n\n"
            "[34:50] ختام المقطع."
        )
        # +2100s (35 minutes)
        shifted_35 = shift_timestamps(text, 2100)
        self.assertIn("[35:10]", shifted_35)
        self.assertIn("[40:45 - 41:20]", shifted_35)
        self.assertIn("[01:09:50]", shifted_35)

        # +3900s (65 minutes -> 01:05:10)
        shifted_65 = shift_timestamps("[00:10] بداية الجزء الثالث", 3900)
        self.assertEqual(shifted_65, "[01:05:10] بداية الجزء الثالث")

        # Zero offset leaves text unchanged
        self.assertEqual(shift_timestamps("[00:10] نص", 0), "[00:10] نص")

    def test_hallucination_loop_cleaner_code_fences_intros_and_inline_ts(self) -> None:
        cycle_5 = [
            "الحمد لله رب العالمين والصلاة والسلام على أشرف المرسلين سيدنا محمد وعلى آله وصحبه أجمعين.",
            "نبدأ اليوم بعون الله تعالى في شرح باب الطهارة من كتاب الفقه الميسر بالتفصيل الكامل.",
            "قال المؤلف رحمه الله تعالى: المياه التي يجوز التطهير بها سبع مياه معروفة عند الفقهاء.",
            "الأول ماء السماء والثاني ماء البحر والثالث ماء النهر والرابع ماء البئر والخامس ماء العين.",
            "وهذه الأقسام السبعة يجمعها قولنا ما نزل من السماء أو نبع من الأرض على أصل خلقته.",
        ]

        repeated_blocks = []
        sec = 120
        for rep in range(10):
            for idx, para in enumerate(cycle_5):
                mm, ss = divmod(sec, 60)
                # Add minor diacritic variation on alternate cycles to test Arabic normalization
                text_variant = para.replace("الفقه", "الفِقه") if rep % 2 == 1 else para
                repeated_blocks.append(f"[{mm:02d}:{ss:02d}] {text_variant}")
                sec += 30

        raw_gemini_output = (
            "```markdown\n"
            "بالتأكيد، إليك التفريغ الحرفي الكامل للمحاضرة:\n\n"
            "[00:00] بسم الله الرحمن الرحيم [00:30] أهلاً بكم أيها الطلاب الكرام في هذه المحاضرة "
            "[01:00] وسنتحدث اليوم عن مسألة مهمة جداً جداً جداً جداً جداً جداً في أصول الفقه.\n\n"
            + "\n\n".join(repeated_blocks)
            + "\n\n[30:00] وبهذا نكون قد انتهينا من درس اليوم ونلقاكم في المحاضرة القادمة.\n\n"
            "آمل أن يكون هذا التفريغ مفيداً لك!\n"
            "```"
        )

        cleaned = clean_transcript_body(raw_gemini_output)

        # No code fences or conversational intro/outro
        self.assertNotIn("```", cleaned)
        self.assertNotIn("بالتأكيد", cleaned)
        self.assertNotIn("آمل أن يكون", cleaned)

        # Inline timestamps separated into individual paragraphs
        paragraphs = [p for p in cleaned.split("\n\n") if p.strip()]
        self.assertTrue(paragraphs[0].startswith("[00:00]"))
        self.assertTrue(paragraphs[1].startswith("[00:30]"))
        self.assertTrue(paragraphs[2].startswith("[01:00]"))

        # Intra-paragraph runaway word repetition collapsed
        self.assertNotIn("جداً جداً جداً جداً", cleaned)

        # 5-paragraph cycle repeated 10x collapsed to exactly ONE occurrence!
        self.assertEqual(cleaned.count("باب الطهارة من كتاب"), 1)
        self.assertEqual(cleaned.count("المياه التي يجوز التطهير بها"), 1)
        self.assertEqual(cleaned.count("ما نزل من السماء أو نبع من الأرض"), 1)

        # Closing genuine paragraph preserved
        self.assertIn("[30:00] وبهذا نكون قد انتهينا", cleaned)
        # Expect 3 intro paragraphs + 5 unique cycle paragraphs + 1 closing = 9 paragraphs
        self.assertEqual(len(paragraphs), 9)

    def test_format_markdown_and_course_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            md1 = format_lecture_markdown(
                title="المحاضرة الأولى: المقدمة",
                course_name="الفقه الميسر",
                seq_num=1,
                source_type="youtube",
                source_id="yt001",
                url="https://youtu.be/yt001",
                cleaned_body="[00:00] نص المحاضرة الأولى.",
            )
            md2 = format_lecture_markdown(
                title="المحاضرة الثانية: الطهارة",
                course_name="الفقه الميسر",
                seq_num=2,
                source_type="gdrive",
                source_id="gd002",
                url="https://drive.google.com/file/d/gd002/view",
                cleaned_body="[00:00] نص المحاضرة الثانية.",
            )
            p1 = os.path.join(tmpdir, "001_intro.md")
            p2 = os.path.join(tmpdir, "002_tahara.md")
            Path(p1).write_text(md1, encoding="utf-8")
            Path(p2).write_text(md2, encoding="utf-8")

            jobs = [
                {
                    "seq_num": 1,
                    "title": "المحاضرة الأولى: المقدمة",
                    "source_type": "youtube",
                    "source_id": "yt001",
                    "url": "https://youtu.be/yt001",
                    "status": "done",
                    "output_path": p1,
                },
                {
                    "seq_num": 2,
                    "title": "المحاضرة الثانية: الطهارة",
                    "source_type": "gdrive",
                    "source_id": "gd002",
                    "url": "https://drive.google.com/file/d/gd002/view",
                    "status": "done",
                    "output_path": p2,
                },
                {
                    "seq_num": 3,
                    "title": "المحاضرة الثالثة",
                    "source_type": "youtube",
                    "source_id": "yt003",
                    "url": "https://youtu.be/yt003",
                    "status": "failed",
                    "output_path": "",
                },
            ]

            artifacts = rebuild_course_artifacts("الفقه الميسر", tmpdir, jobs)
            self.assertEqual(artifacts["completed_count"], 2)
            self.assertEqual(artifacts["total_count"], 3)

            summary_text = Path(artifacts["summary_report"]).read_text(encoding="utf-8")
            self.assertIn("2 / 3", summary_text)
            self.assertIn("001_intro.md", summary_text)
            self.assertIn("002_tahara.md", summary_text)
            self.assertIn("فشل (Failed)", summary_text)

            master_text = Path(artifacts["notebooklm_file"]).read_text(encoding="utf-8")
            self.assertIn("Table of Contents", master_text)
            self.assertIn("نص المحاضرة الأولى.", master_text)
            self.assertIn("نص المحاضرة الثانية.", master_text)


class TestGeminiTranscriberMocked(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.chunk1 = self.tmp / "c0.mp3"
        self.chunk2 = self.tmp / "c1.mp3"
        self.chunk1.write_bytes(b"ID3" + b"\x00" * 2048)
        self.chunk2.write_bytes(b"ID3" + b"\x00" * 2048)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_processing_poll_fallback_429_503_timestamp_shift_and_finally_delete(self) -> None:
        deleted_files: list[str] = []
        poll_counts: dict[str, int] = {}
        gen_calls: list[str] = []

        class FakeFilesAPI:
            def __init__(self) -> None:
                self._n = 0

            def upload(self, file: str):
                self._n += 1
                name = f"files/remote_{self._n}"
                poll_counts[name] = 0
                return types.SimpleNamespace(name=name, state=types.SimpleNamespace(name="PROCESSING"))

            def get(self, name: str):
                poll_counts[name] += 1
                state = "ACTIVE" if poll_counts[name] >= 2 else "PROCESSING"
                return types.SimpleNamespace(name=name, state=types.SimpleNamespace(name=state))

            def delete(self, name: str):
                deleted_files.append(name)

        class FakeModelsAPI:
            def generate_content(self, model: str, contents: list):
                gen_calls.append(model)
                remote_name = contents[0].name
                # On first chunk: model 0 -> 429, model 1 -> 503, model 2 -> success!
                if remote_name == "files/remote_1":
                    if model == "m1":
                        raise RuntimeError("429 RESOURCE_EXHAUSTED quota")
                    if model == "m2":
                        raise RuntimeError("503 Service Unavailable")
                    return types.SimpleNamespace(text="[00:10] بداية الجزء الأول من المحاضرة.")
                # On second chunk: succeeds immediately
                return types.SimpleNamespace(text="[00:10] بداية الجزء الثاني من المحاضرة.")

        fake_client = types.SimpleNamespace(files=FakeFilesAPI(), models=FakeModelsAPI())
        cfg = AppConfig(gemini_api_key="test-key", models=["m1", "m2", "m3"], max_retries=4)
        transcriber = GeminiTranscriber(
            cfg,
            client=fake_client,
            poll_interval=0.05,
            base_backoff=0.05,
            max_backoff=0.1,
        )

        progress_marks: list[float] = []
        with mock.patch("lecture_transcriber.transcriber.gemini_client.is_online", return_value=True):
            stitched = transcriber.transcribe_chunks(
                [
                    {"path": str(self.chunk1), "offset_seconds": 0, "duration_seconds": 2100},
                    {"path": str(self.chunk2), "offset_seconds": 2100, "duration_seconds": 1800},
                ],
                progress_cb=lambda pct: progress_marks.append(pct),
            )

        self.assertIn("[00:10] بداية الجزء الأول", stitched)
        self.assertIn("[35:10] بداية الجزء الثاني", stitched)
        self.assertEqual(deleted_files, ["files/remote_1", "files/remote_2"])
        self.assertEqual(gen_calls[:3], ["m1", "m2", "m3"])
        self.assertEqual(progress_marks[-1], 100.0)

    def test_remote_file_deleted_even_on_auth_error(self) -> None:
        deleted: list[str] = []

        class FakeFilesAPI:
            def upload(self, file: str):
                return types.SimpleNamespace(name="files/must_delete", state="ACTIVE")

            def delete(self, name: str):
                deleted.append(name)

        class FakeModelsAPI:
            def generate_content(self, model: str, contents: list):
                raise RuntimeError("401 API_KEY_INVALID")

        fake_client = types.SimpleNamespace(files=FakeFilesAPI(), models=FakeModelsAPI())
        cfg = AppConfig(gemini_api_key="bad-key", models=["m1"])
        transcriber = GeminiTranscriber(cfg, client=fake_client, base_backoff=0.05)

        with mock.patch("lecture_transcriber.transcriber.gemini_client.is_online", return_value=True):
            with self.assertRaises(GeminiAuthError):
                transcriber.transcribe_chunks([{"path": str(self.chunk1), "offset_seconds": 0}])

        self.assertEqual(deleted, ["files/must_delete"])

    def test_upload_passes_explicit_audio_mime_type_when_supported(self) -> None:
        recorded_uploads: list[tuple[str, Any]] = []

        class FakeFilesWithConfig:
            def upload(self, *, file: str, config: dict | None = None):
                recorded_uploads.append((Path(file).suffix.lower(), config))
                return types.SimpleNamespace(name="files/mime_ok", state="ACTIVE")

            def delete(self, name: str):
                pass

        class FakeModelsOK:
            def generate_content(self, model: str, contents: list):
                return types.SimpleNamespace(text="[00:00] نص سليم.")

        fake_client = types.SimpleNamespace(files=FakeFilesWithConfig(), models=FakeModelsOK())
        transcriber = GeminiTranscriber(
            AppConfig(gemini_api_key="test-key", models=["m1"]),
            client=fake_client,
            base_backoff=0.05,
        )

        expected = [
            (".webm", "audio/webm"),
            (".opus", "audio/ogg"),
            (".m4a", "audio/mp4"),
            (".ogg", "audio/ogg"),
            (".mp3", "audio/mpeg"),
        ]
        with mock.patch("lecture_transcriber.transcriber.gemini_client.is_online", return_value=True):
            for ext, mime in expected:
                f = self.tmp / f"sample{ext}"
                f.write_bytes(b"\x00" * 256)
                out = transcriber.transcribe_single_file(str(f))
                self.assertIn("نص سليم", out)
                self.assertEqual(recorded_uploads[-1], (ext, {"mime_type": mime}))

    def test_max_tokens_finish_reason_triggers_fallback_and_passes_config(self) -> None:
        recorded_calls: list[tuple[str, Any]] = []

        class FakeFiles:
            def upload(self, file: str):
                return types.SimpleNamespace(name="files/max_tok", state="ACTIVE")

            def delete(self, name: str):
                pass

        class FakeModels:
            def generate_content(self, model: str, contents: list, config: Any = None):
                recorded_calls.append((model, config))
                if model == "m1":
                    return types.SimpleNamespace(
                        text="[00:00] بداية الشرح ثم انقطع النص...",
                        candidates=[types.SimpleNamespace(finish_reason="MAX_TOKENS")],
                    )
                return types.SimpleNamespace(
                    text="[00:00] بداية الشرح [00:45] نهاية الشرح كاملة.",
                    candidates=[types.SimpleNamespace(finish_reason="STOP")],
                )

        fake_client = types.SimpleNamespace(files=FakeFiles(), models=FakeModels())
        cfg = AppConfig(gemini_api_key="test-key", models=["m1", "m2"], max_retries=2)
        transcriber = GeminiTranscriber(cfg, client=fake_client, base_backoff=0.05)

        with mock.patch("lecture_transcriber.transcriber.gemini_client.is_online", return_value=True):
            result = transcriber.transcribe_single_file(str(self.chunk1))

        self.assertEqual([m for m, _ in recorded_calls], ["m1", "m2"])
        for _, gen_cfg in recorded_calls:
            self.assertIsNotNone(gen_cfg)
            self.assertEqual(getattr(gen_cfg, "temperature", None), 0.0)
            self.assertEqual(getattr(gen_cfg, "max_output_tokens", None), 65536)
        self.assertIn("نهاية الشرح كاملة", result)

    def test_coverage_guard_rejects_partial_and_accepts_full_or_best_partial(self) -> None:
        class FakeFiles:
            def upload(self, file: str):
                return types.SimpleNamespace(name="files/cov_test", state="ACTIVE")

            def delete(self, name: str):
                pass

        # Case 1: m1 stops at 04:00 (with bogus [99:00] ignored) on 720s chunk -> m2 returns 11:45!
        calls_1: list[str] = []

        class FakeModelsRecover:
            def generate_content(self, model: str, contents: list, config: Any = None):
                calls_1.append(model)
                if model == "m1":
                    return types.SimpleNamespace(
                        text="[00:00] البداية [04:00] توقف مبكر [99:00] رقم خاطئ",
                        candidates=[types.SimpleNamespace(finish_reason="STOP")],
                    )
                return types.SimpleNamespace(
                    text="[00:00] البداية [06:00] المنتصف [11:45] نهاية المحاضرة.",
                    candidates=[types.SimpleNamespace(finish_reason="STOP")],
                )

        t1 = GeminiTranscriber(
            AppConfig(gemini_api_key="test-key", models=["m1", "m2"], max_retries=2),
            client=types.SimpleNamespace(files=FakeFiles(), models=FakeModelsRecover()),
            base_backoff=0.05,
        )
        with mock.patch("lecture_transcriber.transcriber.gemini_client.is_online", return_value=True):
            out1 = t1.transcribe_chunks(
                [{"path": str(self.chunk1), "offset_seconds": 0, "duration_seconds": 720.0}]
            )
        self.assertEqual(calls_1, ["m1", "m2"])
        self.assertIn("[11:45] نهاية المحاضرة", out1)

        # Case 2: All models return partial coverage -> returns highest-coverage best_partial!
        class FakeModelsAllPartial:
            def generate_content(self, model: str, contents: list, config: Any = None):
                if model == "m1":
                    return types.SimpleNamespace(text="[00:00] أ [03:00] ب")
                if model == "m2":
                    return types.SimpleNamespace(text="[00:00] أ [07:30] أفضل تغطية")
                return types.SimpleNamespace(text="[00:00] أ [05:00] ج")

        t2 = GeminiTranscriber(
            AppConfig(gemini_api_key="test-key", models=["m1", "m2", "m3"], max_retries=2),
            client=types.SimpleNamespace(files=FakeFiles(), models=FakeModelsAllPartial()),
            base_backoff=0.05,
        )
        with mock.patch("lecture_transcriber.transcriber.gemini_client.is_online", return_value=True):
            out2 = t2.transcribe_chunks(
                [{"path": str(self.chunk1), "offset_seconds": 0, "duration_seconds": 720.0}]
            )
        self.assertIn("[07:30] أفضل تغطية", out2)

    def test_fast_switch_on_503_and_429_retry_delay_parsing(self) -> None:
        sleep_calls: list[float] = []

        class FakeFiles:
            def upload(self, file: str):
                return types.SimpleNamespace(name="files/fast_503", state="ACTIVE")

            def delete(self, name: str):
                pass

        class FakeModels:
            def generate_content(self, model: str, contents: list, config: Any = None):
                if model == "m1":
                    raise RuntimeError("503 UNAVAILABLE: This model is currently experiencing high demand.")
                if model == "m2":
                    raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 25.5s.")
                return types.SimpleNamespace(text="[00:00] نجح البديل الثالث بسرعة.")

        t = GeminiTranscriber(
            AppConfig(gemini_api_key="test-key", models=["m1", "m2", "m3"], max_retries=2),
            client=types.SimpleNamespace(files=FakeFiles(), models=FakeModels()),
            base_backoff=2.0,
            max_backoff=60.0,
        )

        with mock.patch("lecture_transcriber.transcriber.gemini_client.is_online", return_value=True), \
             mock.patch(
                 "lecture_transcriber.transcriber.gemini_client._interruptible_sleep",
                 side_effect=lambda sec, *a, **kw: sleep_calls.append(sec),
             ):
            out = t.transcribe_single_file(str(self.chunk1))

        self.assertIn("نجح البديل الثالث", out)
        self.assertEqual(len(sleep_calls), 2)
        self.assertLessEqual(sleep_calls[0], 0.5)
        self.assertLessEqual(sleep_calls[1], 0.25)

    def test_daily_quota_429_skips_model_immediately_without_stalling(self) -> None:
        sleep_calls: list[float] = []
        calls: list[str] = []

        class FakeFiles:
            def upload(self, file: str):
                return types.SimpleNamespace(name="files/daily_429", state="ACTIVE")

            def delete(self, name: str):
                pass

        class FakeModels:
            def generate_content(self, model: str, contents: list, config: Any = None):
                calls.append(model)
                if model == "m1":
                    raise RuntimeError(
                        "429 RESOURCE_EXHAUSTED: GenerateRequestsPerDayPerProjectPerModel-FreeTier quota exceeded"
                    )
                if model == "m2" and calls.count("m2") == 1:
                    raise RuntimeError("503 UNAVAILABLE")
                return types.SimpleNamespace(text="[00:00] تم التفريغ عبر الحصة المستقلة للنموذج الثاني.")

        t = GeminiTranscriber(
            AppConfig(gemini_api_key="test-key", models=["m1", "m2"], max_retries=4),
            client=types.SimpleNamespace(files=FakeFiles(), models=FakeModels()),
            base_backoff=2.0,
            max_backoff=60.0,
        )

        with mock.patch("lecture_transcriber.transcriber.gemini_client.is_online", return_value=True), \
             mock.patch(
                 "lecture_transcriber.transcriber.gemini_client._interruptible_sleep",
                 side_effect=lambda sec, *a, **kw: sleep_calls.append(sec),
             ):
            out = t.transcribe_single_file(str(self.chunk1))

        self.assertIn("الحصة المستقلة", out)
        # m1 hit daily RPD exhaustion in round 1 -> MUST NOT be retried in round 2!
        self.assertEqual(calls, ["m1", "m2", "m2"])
        # No sleep when skipping daily-exhausted m1; only the fast 503 switch & inter-round pause
        self.assertEqual(len(sleep_calls), 2)


class TestTranscriptionEngineEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.cfg_mgr = ConfigManager(self.tmp / "config.json")
        self.cfg_mgr.update(
            gemini_api_key="test-key",
            output_base_dir=str(self.tmp / "output"),
            temp_dir=str(self.tmp / "temp"),
            download_workers=2,
            transcribe_workers=2,
            max_retries=2,
        )
        self.db = Database(self.tmp / "state.db")

    def tearDown(self) -> None:
        self.db.close()
        self._tmpdir.cleanup()

    def _wait_for_status(self, job_id: str, expected: str | set[str], timeout: float = 8.0) -> dict:
        targets = {expected} if isinstance(expected, str) else set(expected)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.db.get_job(job_id)
            if job and job["status"] in targets:
                return job
            time.sleep(0.05)
        job = self.db.get_job(job_id)
        self.fail(f"Timed out waiting for job {job_id} to reach {targets}; current={job}")

    def test_full_pipeline_enqueue_to_done_and_cleanup(self) -> None:
        created_chunks: list[Path] = []

        def fake_expand(urls, config=None, fetch_metadata=True, include_playlist=False):
            return [
                ParsedMediaItem(
                    source_type="youtube",
                    url="https://www.youtube.com/watch?v=vid_e2e_01",
                    source_id="vid_e2e_01",
                    title="المحاضرة الأولى",
                    seq_num=1,
                )
            ]

        def fake_download(source_type, url, source_id, temp_dir, config=None, progress_cb=None, cancel_event=None, pause_event=None, on_status=None):
            raw = Path(temp_dir) / f"{source_id}.m4a"
            raw.write_bytes(b"raw-audio-bytes")
            if progress_cb:
                progress_cb(50.0, 200_000.0, 2.0, 500, 1000)
                progress_cb(100.0, 200_000.0, 0.0, 1000, 1000)
            return str(raw)

        def fake_prepare(raw_path, temp_dir, source_id, config=None, cancel_event=None):
            if os.path.isfile(raw_path):
                os.remove(raw_path)
            c0 = Path(temp_dir) / f"{source_id}_chunk_000.mp3"
            c0.write_bytes(b"ID3" + b"\x00" * 512)
            created_chunks.append(c0)
            return [{"path": str(c0), "offset_seconds": 0, "duration_seconds": 60.0}]

        class FakeTranscriber:
            def transcribe_chunks(self, chunks, language_hint="ar", cancel_event=None, pause_event=None, progress_cb=None, on_status=None):
                if progress_cb:
                    progress_cb(100.0, 1, 1, "done")
                return "```markdown\n[00:00] بسم الله الرحمن الرحيم [00:25] هذا تفريغ تجريبي ناجح.\n```"

        events: list[dict] = []
        with mock.patch("lecture_transcriber.network.is_online", return_value=True), \
             mock.patch("lecture_transcriber.engine.is_online", return_value=True):
            mon = NetworkStatusMonitor(check_interval=10.0)
            engine = TranscriptionEngine(
                self.cfg_mgr,
                self.db,
                transcriber_factory=lambda cfg: FakeTranscriber(),
                network_monitor=mon,
                download_fn=fake_download,
                prepare_audio_fn=fake_prepare,
                expand_urls_fn=fake_expand,
            )
            q = engine.subscribe()
            engine.start()
            try:
                res = engine.enqueue_urls("فقه العبادات", ["https://youtu.be/vid_e2e_01"])
                self.assertEqual(res["added"], 1)
                job_id = res["job_ids"][0]

                done_job = self._wait_for_status(job_id, "done")
                self.assertEqual(done_job["progress"], 100.0)
                out_md = Path(done_job["output_path"])
                self.assertTrue(out_md.exists())
                content = out_md.read_text(encoding="utf-8")
                self.assertIn("[00:00] بسم الله الرحمن الرحيم", content)
                self.assertIn("[00:25] هذا تفريغ تجريبي ناجح.", content)
                self.assertNotIn("```", content)

                # Temp audio chunks must be cleaned up after completion
                self.assertTrue(created_chunks)
                for cp in created_chunks:
                    self.assertFalse(cp.exists(), f"Chunk {cp} was not cleaned up after done")

                # Course summary + NotebookLM master generated
                course_dir = Path(res["course"]["output_dir"])
                self.assertTrue((course_dir / "_Summary_Report.md").exists())
                master_files = list(course_dir.glob("000_Complete_*_NotebookLM.md"))
                self.assertEqual(len(master_files), 1)

                # Re-enqueueing same URL skips as duplicate
                dup_res = engine.enqueue_urls("فقه العبادات", ["https://youtu.be/vid_e2e_01"])
                self.assertEqual(dup_res["added"], 0)
                self.assertEqual(dup_res["skipped_duplicates"], 1)

                # Drain pub-sub queue and verify job_update stages seen
                while not q.empty():
                    events.append(q.get_nowait())
                stages_seen = {
                    e["data"]["job"]["status"]
                    for e in events
                    if e["type"] == "job_update" and e["data"]["job_id"] == job_id
                }
                self.assertIn("downloading", stages_seen)
                self.assertIn("converting", stages_seen)
                self.assertIn("transcribing", stages_seen)
                self.assertIn("cleaning", stages_seen)
                self.assertIn("done", stages_seen)
            finally:
                engine.stop()

    def test_pause_resume_and_cancel_cleanup(self) -> None:
        dl_started = threading.Event()
        dl_calls = {"count": 0}

        def fake_expand(urls, config=None, fetch_metadata=True, include_playlist=False):
            return [
                ParsedMediaItem(source_type="gdrive", url=urls[0], source_id=urls[0].split("/")[-1], title=f"Title_{urls[0].split('/')[-1]}", seq_num=1)
            ]

        def fake_download(source_type, url, source_id, temp_dir, config=None, progress_cb=None, cancel_event=None, pause_event=None, on_status=None):
            dl_calls["count"] += 1
            part = Path(temp_dir) / f"{source_id}.gdrive.part"
            part.write_bytes(b"partial-bytes")
            dl_started.set()
            # Wait briefly on first attempt so test can pause it
            if dl_calls["count"] == 1:
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline:
                    if pause_event and pause_event.is_set():
                        raise DownloadPausedError("paused")
                    time.sleep(0.02)
            raw = Path(temp_dir) / f"{source_id}.mp3"
            raw.write_bytes(b"full-audio")
            if part.exists():
                part.unlink()
            return str(raw)

        def fake_prepare(raw_path, temp_dir, source_id, config=None, cancel_event=None):
            if os.path.isfile(raw_path):
                os.remove(raw_path)
            c0 = Path(temp_dir) / f"{source_id}_chunk_000.mp3"
            c0.write_bytes(b"ID3" + b"\x00" * 512)
            return [{"path": str(c0), "offset_seconds": 0, "duration_seconds": 30.0}]

        class FakeTranscriber:
            def transcribe_chunks(self, chunks, **kwargs):
                return "[00:00] تفريغ مكتمل بعد الاستئناف."

        with mock.patch("lecture_transcriber.network.is_online", return_value=True), \
             mock.patch("lecture_transcriber.engine.is_online", return_value=True):
            mon = NetworkStatusMonitor(check_interval=10.0)
            engine = TranscriptionEngine(
                self.cfg_mgr,
                self.db,
                transcriber_factory=lambda cfg: FakeTranscriber(),
                network_monitor=mon,
                download_fn=fake_download,
                prepare_audio_fn=fake_prepare,
                expand_urls_fn=fake_expand,
            )
            engine.start()
            try:
                res = engine.enqueue_urls("PauseCourse", ["https://drive.google.com/file/d/job_pause_1"])
                jid = res["job_ids"][0]
                self.assertTrue(dl_started.wait(timeout=3.0))

                engine.pause_job(jid)
                self._wait_for_status(jid, "paused")
                # .part file preserved on pause!
                part_file = self.tmp / "temp" / "job_pause_1.gdrive.part"
                self.assertTrue(part_file.exists())

                engine.resume_job(jid)
                self._wait_for_status(jid, "done")
                self.assertFalse(part_file.exists())

                # Test cancel of a paused job cleans up temp .part file
                res2 = engine.enqueue_urls("PauseCourse", ["https://drive.google.com/file/d/job_cancel_2"])
                jid2 = res2["job_ids"][0]
                engine.pause_job(jid2)
                self._wait_for_status(jid2, "paused")
                leftover = self.tmp / "temp" / "job_cancel_2.gdrive.part"
                leftover.write_bytes(b"temp-to-clean")
                engine.cancel_job(jid2)
                self._wait_for_status(jid2, "cancelled")
                self.assertFalse(leftover.exists())
            finally:
                engine.stop()

    def test_offline_to_online_recovery_and_dl_status_bool_callback(self) -> None:
        online_flag = {"value": False}
        events: list[dict] = []

        def fake_expand(urls, config=None, fetch_metadata=True, include_playlist=False):
            return [
                ParsedMediaItem(source_type="youtube", url=urls[0], source_id="vid_offline_1", title="Offline Test", seq_num=1)
            ]

        def fake_download(source_type, url, source_id, temp_dir, config=None, progress_cb=None, cancel_event=None, pause_event=None, on_status=None):
            # Simulate wait_for_internet calling on_status(False) then on_status(True) with bools!
            if on_status is not None:
                on_status(False)
                on_status(True)
            raw = Path(temp_dir) / f"{source_id}.mp3"
            raw.write_bytes(b"audio")
            return str(raw)

        def fake_prepare(raw_path, temp_dir, source_id, config=None, cancel_event=None):
            os.remove(raw_path)
            c0 = Path(temp_dir) / f"{source_id}_chunk_000.mp3"
            c0.write_bytes(b"ID3" + b"\x00" * 256)
            return [{"path": str(c0), "offset_seconds": 0, "duration_seconds": 10.0}]

        class FakeTranscriber:
            def transcribe_chunks(self, chunks, **kwargs):
                return "[00:00] تم التفريغ بعد عودة الإنترنت."

        with mock.patch("lecture_transcriber.network.is_online", side_effect=lambda timeout=3.0: online_flag["value"]), \
             mock.patch("lecture_transcriber.engine.is_online", side_effect=lambda timeout=3.0: online_flag["value"]):
            mon = NetworkStatusMonitor(check_interval=0.05, timeout=0.05)
            engine = TranscriptionEngine(
                self.cfg_mgr,
                self.db,
                transcriber_factory=lambda cfg: FakeTranscriber(),
                network_monitor=mon,
                download_fn=fake_download,
                prepare_audio_fn=fake_prepare,
                expand_urls_fn=fake_expand,
            )
            engine.subscribe(lambda ev: events.append(ev))
            engine.start()
            try:
                # Wait for monitor to notice offline
                deadline = time.monotonic() + 2.0
                while engine.is_online and time.monotonic() < deadline:
                    time.sleep(0.03)
                self.assertFalse(engine.is_online)

                res = engine.enqueue_urls("NetCourse", ["https://youtu.be/vid_offline_1"])
                jid = res["job_ids"][0]
                time.sleep(0.25)
                # Job must stay queued (NOT failed!) while offline
                self.assertEqual(self.db.get_job(jid)["status"], "queued")

                # Restore internet -> engine must emit online and finish the job!
                online_flag["value"] = True
                self._wait_for_status(jid, "done", timeout=6.0)
                event_types = [e["type"] for e in events]
                self.assertIn("offline", event_types)
                self.assertIn("online", event_types)
            finally:
                engine.stop()

    def test_keep_audio_saves_concatenated_mp3_beside_transcript(self) -> None:
        import shutil
        import subprocess

        ffmpeg_bin = shutil.which("ffmpeg")
        self.assertIsNotNone(ffmpeg_bin)
        self.cfg_mgr.update(keep_audio=True, download_mode="eco")
        created_chunks: list[Path] = []

        def fake_expand(urls, config=None, fetch_metadata=True, include_playlist=False):
            return [
                ParsedMediaItem(
                    source_type="youtube",
                    url="https://www.youtube.com/watch?v=keep_vid_01",
                    source_id="keep_vid_01",
                    title="درس حفظ الصوت",
                    seq_num=1,
                )
            ]

        def fake_download(source_type, url, source_id, temp_dir, config=None, progress_cb=None, cancel_event=None, pause_event=None, on_status=None):
            raw = Path(temp_dir) / f"{source_id}.mp4"
            raw.write_bytes(b"raw")
            return str(raw)

        def fake_prepare(raw_path, temp_dir, source_id, config=None, cancel_event=None):
            if os.path.isfile(raw_path):
                os.remove(raw_path)
            c0 = Path(temp_dir) / f"{source_id}_chunk_000.mp3"
            c1 = Path(temp_dir) / f"{source_id}_chunk_001.mp3"
            for idx, cp in enumerate((c0, c1)):
                subprocess.run(
                    [
                        ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", f"sine=frequency={440 + idx * 110}:duration=1.5",
                        "-ac", "1", "-ar", "16000", "-b:a", "32k", str(cp),
                    ],
                    check=True,
                )
                created_chunks.append(cp)
            return [
                {"path": str(c0), "offset_seconds": 0, "duration_seconds": 1.5},
                {"path": str(c1), "offset_seconds": 2, "duration_seconds": 1.5},
            ]

        class FakeTranscriber:
            def transcribe_chunks(self, chunks, **kwargs):
                return "[00:00] تفريغ مع حفظ ملف الصوت."

        with mock.patch("lecture_transcriber.network.is_online", return_value=True), \
             mock.patch("lecture_transcriber.engine.is_online", return_value=True):
            mon = NetworkStatusMonitor(check_interval=10.0)
            engine = TranscriptionEngine(
                self.cfg_mgr,
                self.db,
                transcriber_factory=lambda cfg: FakeTranscriber(),
                network_monitor=mon,
                download_fn=fake_download,
                prepare_audio_fn=fake_prepare,
                expand_urls_fn=fake_expand,
            )
            engine.start()
            try:
                res = engine.enqueue_urls("حفظ الصوت", ["https://youtu.be/keep_vid_01"])
                jid = res["job_ids"][0]
                done_job = self._wait_for_status(jid, "done", timeout=10.0)

                md_path = Path(done_job["output_path"])
                self.assertTrue(md_path.exists())
                saved_audio = md_path.parent / "audio" / f"{md_path.stem}.mp3"
                self.assertTrue(saved_audio.exists(), f"Expected saved MP3 at {saved_audio}")

                audio_info = probe_audio_info(saved_audio)
                self.assertTrue(audio_info["has_audio"])
                self.assertAlmostEqual(audio_info["duration_seconds"], 3.0, delta=0.8)

                # Temp chunks must still be cleaned up
                for cp in created_chunks:
                    self.assertFalse(cp.exists())
            finally:
                engine.stop()

    def test_media_extension_stripped_from_md_filename_and_h1_heading(self) -> None:
        from lecture_transcriber.engine import _build_output_path, _find_existing_completed_md

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. _build_output_path strips media extensions
            out_path = _build_output_path(tmpdir, 3, "Chapter1C.mp4")
            self.assertEqual(os.path.basename(out_path), "003_Chapter1C.md")
            self.assertEqual(
                os.path.basename(_build_output_path(tmpdir, 4, "Lecture_04.MKV")),
                "004_Lecture_04.md",
            )

            # 2. _find_existing_completed_md matches both clean and legacy (.mp4.md) files
            legacy_file = Path(tmpdir) / "003_Chapter1C.mp4.md"
            legacy_file.write_text("# Chapter1C\n" + ("x" * 1200), encoding="utf-8")
            self.assertEqual(_find_existing_completed_md(tmpdir, 3, "Chapter1C.mp4"), str(legacy_file))
            legacy_file.unlink()

            clean_file = Path(tmpdir) / "003_Chapter1C.md"
            clean_file.write_text("# Chapter1C\n" + ("x" * 1200), encoding="utf-8")
            self.assertEqual(_find_existing_completed_md(tmpdir, 3, "Chapter1C.mp4"), str(clean_file))

        # 3. Full engine pipeline with Chapter1C.mp4 produces 003_Chapter1C.md and '# Chapter1C'
        def fake_expand(urls, config=None, fetch_metadata=True, include_playlist=False):
            return [
                ParsedMediaItem(
                    source_type="gdrive",
                    url="https://drive.google.com/file/d/chap1c/view",
                    source_id="chap1c",
                    title="Chapter1C.mp4",
                    seq_num=1,
                )
            ]

        def fake_download(source_type, url, source_id, temp_dir, config=None, **kwargs):
            raw = Path(temp_dir) / f"{source_id}.mp4"
            raw.write_bytes(b"raw")
            return str(raw)

        def fake_prepare(raw_path, temp_dir, source_id, config=None, cancel_event=None):
            if os.path.isfile(raw_path):
                os.remove(raw_path)
            c0 = Path(temp_dir) / f"{source_id}_chunk_000.mp3"
            c0.write_bytes(b"ID3" + b"\x00" * 256)
            return [{"path": str(c0), "offset_seconds": 0, "duration_seconds": 15.0}]

        class FakeTranscriber:
            def transcribe_chunks(self, chunks, **kwargs):
                return "[00:00] Welcome to Embedded Systems."

        with mock.patch("lecture_transcriber.network.is_online", return_value=True), \
             mock.patch("lecture_transcriber.engine.is_online", return_value=True):
            mon = NetworkStatusMonitor(check_interval=10.0)
            engine = TranscriptionEngine(
                self.cfg_mgr,
                self.db,
                transcriber_factory=lambda cfg: FakeTranscriber(),
                network_monitor=mon,
                download_fn=fake_download,
                prepare_audio_fn=fake_prepare,
                expand_urls_fn=fake_expand,
            )
            engine.start()
            try:
                res = engine.enqueue_urls("Embedded Systems", ["https://drive.google.com/file/d/chap1c/view"])
                done_job = self._wait_for_status(res["job_ids"][0], "done")
                md_path = Path(done_job["output_path"])
                self.assertEqual(md_path.name, "001_Chapter1C.md")
                md_text = md_path.read_text(encoding="utf-8")
                self.assertTrue(md_text.startswith("# Chapter1C\n"))
                self.assertNotIn("# Chapter1C.mp4", md_text)
            finally:
                engine.stop()


if __name__ == "__main__":
    unittest.main()
