# Copyright 2026 Innodisk Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline tests for tool/test/regression/run_pipelines.py (no docker, no device)."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "tool" / "test" / "regression" / "run_pipelines.py"
PIPELINES_PATH = REPO_ROOT / "regression" / "pipelines.yaml"

_spec = importlib.util.spec_from_file_location("run_pipelines", RUNNER_PATH)
rp = importlib.util.module_from_spec(_spec)
sys.modules["run_pipelines"] = rp
_spec.loader.exec_module(rp)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


class PipelineDefinitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config, self.pipelines = rp.load_pipelines(PIPELINES_PATH)
        self.mapper = rp.load_mapper(REPO_ROOT)

    def test_all_supported_pipelines_defined_and_consistent_with_product(self) -> None:
        self.assertEqual(len(self.pipelines), 21)
        errors, warnings = rp.cross_check_pipelines(self.pipelines, self.mapper)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_cross_check_flags_wrong_calibration_and_unsupported_combo(self) -> None:
        bad = [
            rp.Pipeline(1, "a", "yolov26", "litert", "fp32", True),
            rp.Pipeline(2, "b", "yolov26", "qairt", "fp32", False),
        ]
        errors, warnings = rp.cross_check_pipelines(bad, self.mapper)
        self.assertTrue(any("a: needs_calibration" in e for e in errors))
        self.assertTrue(any("b: qairt/fp32" in e for e in errors))
        self.assertTrue(warnings)

    def test_select_pipelines_by_all_ids_and_globs_keeps_definition_order(self) -> None:
        self.assertEqual(len(rp.select_pipelines(["all"], self.pipelines)), 21)
        picked = rp.select_pipelines(
            ["yolov26-qairt-int8", "yolov10-*-fp32"], self.pipelines
        )
        self.assertEqual(
            [p.id for p in picked],
            ["yolov10-litert-fp32", "yolov10-onnx-fp32", "yolov26-qairt-int8"],
        )
        with self.assertRaises(rp.RegressionError):
            rp.select_pipelines(["yolov8-litert-int8"], self.pipelines)

    def test_pipeline_numbers_are_unique_and_selectable(self) -> None:
        self.assertEqual([p.num for p in self.pipelines], list(range(1, 22)))
        by_number = rp.select_pipelines([3, "21", "15-16"], self.pipelines)
        self.assertEqual(
            [p.id for p in by_number],
            [
                "yolov10-onnx-fp32",
                "yolov26-litert-int8",
                "yolov26-litert-fp32",
                "yolov26-qairt-fp16",
            ],
        )
        self.assertEqual(len(rp.select_pipelines(["7-1"], self.pipelines)), 7)
        with self.assertRaises(rp.RegressionError):
            rp.select_pipelines([99], self.pipelines)


class InputsAndPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config, self.pipelines = rp.load_pipelines(PIPELINES_PATH)
        self.defaults = self.config["defaults"]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _inputs(self, extra: str = "") -> Path:
        return write(
            self.root / "run" / "inputs.yaml",
            f"""
            repo_root: {REPO_ROOT}
            docker_image: iq-foundry:test
            pipelines: [yolov26-litert-fp32, yolov26-litert-int8, yolov11-qairt-fp16]
            modes: [test, qc, mAP]
            models_pt:
              yolov11: /m/yolo11n.pt
              yolov26: /m/yolo26n.pt
            calibration: {{dir: /c, max_calib: 50}}
            map: {{annotations: /a.json, images: /i, max_images: 10}}
            test: {{image: null, images: /t, yaml: /y.yaml}}
            device: {{adb_serial: SER1}}
            extra_flags:
              mAP: "--conf 0.3"
            {extra}
            """,
        )

    def _plans(self, inputs, pid: str):
        pipeline = next(p for p in self.pipelines if p.id == pid)
        plans = rp.build_stage_plans(inputs, self.config, pipeline, self.root / "run")
        return {plan.mode: plan for plan in plans}

    def test_inputs_rejects_token_keys(self) -> None:
        path = self._inputs("qai_hub: {use_existing_login: false, token: abc}")
        with self.assertRaises(rp.RegressionError):
            rp.load_inputs(path, self.defaults)

    def test_modes_are_canonical_order_and_output_defaults_to_run_dir(self) -> None:
        inputs = rp.load_inputs(self._inputs(), self.defaults)
        self.assertEqual(inputs.modes, ["qc", "mAP", "test"])
        self.assertEqual(inputs.output_dir, (self.root / "run").resolve())
        self.assertEqual(inputs.extra_flags["mAP"], ["--conf", "0.3"])
        self.assertEqual(inputs.timeouts_s["qc"], self.defaults["timeouts_s"]["qc"])

    def test_calibration_only_passed_when_pipeline_needs_it(self) -> None:
        inputs = rp.load_inputs(self._inputs(), self.defaults)
        fp32 = self._plans(inputs, "yolov26-litert-fp32")["qc"].argv
        int8 = self._plans(inputs, "yolov26-litert-int8")["qc"].argv
        fp16 = self._plans(inputs, "yolov11-qairt-fp16")["qc"].argv
        self.assertNotIn("--calib_dir", fp32)
        self.assertNotIn("--calib_dir", fp16)
        self.assertIn("--calib_dir", int8)
        self.assertEqual(int8[int8.index("--max_calib") + 1], "50")
        self.assertEqual(
            int8[:5], ["./docker/iqf", "--image", "iq-foundry:test", "run", "qc"]
        )
        self.assertNotIn("--save", int8)

    def test_map_and_test_chain_from_qc_artifact_with_device_flags(self) -> None:
        inputs = rp.load_inputs(self._inputs(), self.defaults)
        plans = self._plans(inputs, "yolov26-litert-int8")
        qc_out = plans["qc"].argv[plans["qc"].argv.index("--output") + 1]
        self.assertTrue(
            qc_out.endswith(
                "artifacts/yolov26-litert-int8/qc/yolov26-litert-int8.tflite"
            )
        )
        map_argv = plans["mAP"].argv
        self.assertEqual(map_argv[map_argv.index("--converted-model") + 1], qc_out)
        self.assertEqual(map_argv[map_argv.index("--max-images") + 1], "10")
        self.assertEqual(map_argv[map_argv.index("--adb-serial") + 1], "SER1")
        self.assertEqual(map_argv[-2:], ["--conf", "0.3"])
        test_argv = plans["test"].argv
        self.assertIn("--adb", test_argv)
        self.assertIn("--images", test_argv)
        # The wrapper's own --image <tag> precedes `run`; the backend test flags follow the mode.
        self.assertNotIn("--image", test_argv[test_argv.index("test") :])
        self.assertTrue(plans["mAP"].model_from_qc)

    def test_prebuilt_model_skips_qc_and_feeds_map_and_test(self) -> None:
        path = self._inputs("prebuilt_models: {yolov26-litert-int8: /p/model.tflite}")
        inputs = rp.load_inputs(path, self.defaults)
        plans = self._plans(inputs, "yolov26-litert-int8")
        self.assertFalse(plans["qc"].selected)
        self.assertEqual(plans["qc"].skip_reason, "prebuilt converted model supplied")
        self.assertFalse(plans["mAP"].model_from_qc)
        self.assertIn("/p/model.tflite", plans["test"].argv)

    def test_controlled_flags_are_rejected(self) -> None:
        path = self._inputs(
            "pipeline_extra_flags: {yolov26-litert-int8: {qc: ['--output', '/x']}}"
        )
        inputs = rp.load_inputs(path, self.defaults)
        inputs.extra_flags["test"] = ["--adb-serial=abc"]
        errors = rp._controlled_flag_errors(inputs)
        self.assertEqual(len(errors), 2)


class ParsingAndEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config, _ = rp.load_pipelines(PIPELINES_PATH)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_parse_adb_devices(self) -> None:
        output = (
            "List of devices attached\n"
            "6dd95b13               device usb:1-10 transport_id:1\n"
            "abc unauthorized usb:1-2\n"
        )
        devices = rp.parse_adb_devices(output)
        self.assertEqual(devices[0]["serial"], "6dd95b13")
        self.assertEqual(devices[0]["usb"], "1-10")
        self.assertEqual(devices[1]["state"], "unauthorized")

    def test_dry_run_inner_and_injected_flags(self) -> None:
        output = (
            "Inner command:\n"
            "  python3 cli.py --type yolov26 --mode qc --runtime litert --precision fp32 "
            "--model /inputs/qc/model/m.pt --calib_dir /inputs/qc/calib_dir\n"
            "Docker command:\n  docker run ...\n"
        )
        inner = rp.parse_dry_run_inner(output)
        self.assertEqual(inner[:2], ["python3", "cli.py"])
        argv = [
            "./docker/iqf",
            "run",
            "qc",
            "--type",
            "t",
            "--runtime",
            "r",
            "--precision",
            "p",
            "--model",
            "m",
        ]
        self.assertEqual(rp.injected_flags(argv, inner), ["--calib_dir"])

    def test_parse_map_result_current_and_legacy_keys(self) -> None:
        current = write(
            self.root / "a.txt",
            """\
            num_images: 300
            reference_map50: 0.523173
            converted_map50: 0.511969
            pct_delta_vs_reference: -2.14%
            trend: decrease
            """,
        )
        metrics = rp.parse_map_result(current)
        self.assertAlmostEqual(metrics["converted_map50"], 0.511969)
        self.assertAlmostEqual(metrics["pct_delta_value"], -2.14)
        legacy = write(self.root / "b.txt", "fp_map50: 0.5\nint_map50: 0.4\n")
        self.assertAlmostEqual(rp.parse_map_result(legacy)["reference_map50"], 0.5)

    def test_evaluate_qc_requires_artifact(self) -> None:
        artifact = self.root / "qc" / "m.tflite"
        artifact.parent.mkdir(parents=True)
        result = rp.evaluate_stage("qc", 0, "exited", artifact, "", self.config)
        self.assertEqual(result["status"], rp.FAIL)
        artifact.write_bytes(b"x" * 10)
        result = rp.evaluate_stage(
            "qc", 0, "exited", artifact, "[ok] wrote", self.config
        )
        self.assertEqual(result["status"], rp.PASS)
        self.assertEqual(result["metrics"]["size_bytes"], 10)

    def test_evaluate_nonzero_exit_extracts_error(self) -> None:
        log = "# command: x\nstuff\n[error] Path does not exist: /nope\n"
        result = rp.evaluate_stage(
            "qc", 1, "exited", self.root / "none", log, self.config
        )
        self.assertEqual(result["status"], rp.FAIL)
        self.assertEqual(result["error_excerpt"], "[error] Path does not exist: /nope")
        traceback_log = (
            "Traceback (most recent call last):\n  File x\nRuntimeError: boom\n"
        )
        result = rp.evaluate_stage(
            "qc", 1, "exited", self.root / "none", traceback_log, self.config
        )
        self.assertEqual(result["error_excerpt"], "RuntimeError: boom")

    def test_evaluate_map_threshold_warning(self) -> None:
        result_file = write(
            self.root / "m.txt",
            "reference_map50: 0.5\nconverted_map50: 0.3\npct_delta_vs_reference: -40.00%\n",
        )
        result = rp.evaluate_stage("mAP", 0, "exited", result_file, "", self.config)
        self.assertEqual(result["status"], rp.PASS_WITH_WARNINGS)

    def test_evaluate_test_outputs_and_warning_patterns(self) -> None:
        out = self.root / "test" / "output"
        out.mkdir(parents=True)
        (out / "classes.txt").write_text("person\n")
        result = rp.evaluate_stage("test", 0, "exited", out, "", self.config)
        self.assertEqual(result["status"], rp.FAIL)
        (out / "a.jpg").write_bytes(b"x")
        (out / "a.txt").write_text("0 0.5 0.5 0.1 0.1 0.9\n")
        log = "avg_total_inference_ms=12.500\navg_model_invoke_ms=8.250\nFalling back to CPUExecutionProvider\n"
        result = rp.evaluate_stage("test", 0, "exited", out, log, self.config)
        self.assertEqual(result["status"], rp.PASS_WITH_WARNINGS)
        self.assertEqual(result["metrics"]["annotated_images"], 1)
        self.assertEqual(result["metrics"]["avg_total_inference_ms"], 12.5)

    def test_evaluate_timeout(self) -> None:
        result = rp.evaluate_stage(
            "mAP", None, "timeout", self.root / "x", "", self.config
        )
        self.assertEqual(result["status"], rp.TIMEOUT)


class ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_execute_stage_tees_output_and_reports_exit(self) -> None:
        log = self.root / "logs" / "p" / "qc.log"
        with (
            mock.patch.object(rp, "docker_container_ids", return_value=set()),
            mock.patch("sys.stdout"),
        ):
            code, outcome, _ = rp.execute_stage(
                [
                    sys.executable,
                    "-c",
                    "print('\\x1b[32m[ok] wrote: x\\x1b[0m'); raise SystemExit(4)",
                ],
                cwd=self.root,
                log_path=log,
                timeout_s=30,
                prefix="p:qc",
                image="img",
                header=["command: fake"],
            )
        self.assertEqual((code, outcome), (4, "exited"))
        text = log.read_text()
        self.assertIn("# command: fake", text)
        self.assertIn("[ok] wrote: x", text)
        self.assertNotIn("\x1b[", text)

    def test_execute_stage_timeout_kills_process(self) -> None:
        log = self.root / "t.log"
        with (
            mock.patch.object(rp, "docker_container_ids", return_value=set()),
            mock.patch("sys.stdout"),
        ):
            _, outcome, duration = rp.execute_stage(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                cwd=self.root,
                log_path=log,
                timeout_s=1,
                prefix="p:qc",
                image="img",
                header=[],
            )
        self.assertEqual(outcome, "timeout")
        self.assertLess(duration, 30)
        self.assertIn("timed out after 1s", log.read_text())


class TokenSecurityTests(unittest.TestCase):
    SECRET = "abcdefghijklmnopqrstuvwxyz0123456789ABCD"

    def test_masker_masks_token_split_across_chunks(self) -> None:
        masker = rp.TokenMasker(self.SECRET)
        out = masker.feed("api_token = " + self.SECRET[:10])
        out += masker.feed(self.SECRET[10:] + "\nnext line\n")
        out += masker.flush()
        self.assertNotIn(self.SECRET, out)
        self.assertNotIn(self.SECRET[:10], out)
        self.assertEqual(out, "api_token = ***\nnext line\n")

    def test_files_containing_finds_leaks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "clean.txt", "nothing here")
            leak = write(root / "logs" / "x.log", f"key {self.SECRET}")
            self.assertEqual(rp.files_containing(self.SECRET, root), [leak])

    def _hub_login_args(self, run_dir: Path):
        inputs = write(
            run_dir / "inputs.yaml",
            f"repo_root: {REPO_ROOT}\ndocker_image: img\npipelines: [1]\n",
        )
        return rp.build_parser().parse_args(
            ["--pipelines", str(PIPELINES_PATH), "hub-login", "--inputs", str(inputs)]
        )

    def test_hub_login_refuses_short_token_without_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._hub_login_args(Path(tmp))
            with (
                mock.patch.dict("os.environ", {rp.TOKEN_ENV: "short"}),
                mock.patch.object(rp.pty, "fork") as fork,
            ):
                with self.assertRaises(rp.RegressionError):
                    rp.cmd_hub_login(args)
                fork.assert_not_called()

    def test_hub_login_refuses_when_token_is_in_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = self._hub_login_args(Path(tmp))
            write(Path(tmp) / "notes.txt", f"token {self.SECRET}")
            with (
                mock.patch.dict("os.environ", {rp.TOKEN_ENV: self.SECRET}),
                mock.patch.object(rp.pty, "fork") as fork,
            ):
                with self.assertRaises(rp.RegressionError) as ctx:
                    rp.cmd_hub_login(args)
                fork.assert_not_called()
                self.assertNotIn(self.SECRET, str(ctx.exception))
                self.assertNotIn(rp.TOKEN_ENV, rp.os.environ)

    def test_inputs_reject_token_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write(
                Path(tmp) / "inputs.yaml",
                f"repo_root: {REPO_ROOT}\ndocker_image: img\npipelines: [1]\n"
                "qai_hub_api_key: x\n",
            )
            with self.assertRaises(rp.RegressionError):
                rp.load_inputs(path, {})


class RunStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _status(self) -> int:
        args = rp.build_parser().parse_args(["status", "--run-dir", str(self.run_dir)])
        with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            return rp.cmd_status(args)

    def test_status_without_any_run_is_an_error(self) -> None:
        self.assertEqual(self._status(), rp.EXIT_ERROR)

    def test_status_while_validating_is_ok(self) -> None:
        rp.write_state(self.run_dir, "validating", pid=rp.os.getpid())
        with mock.patch.object(rp, "runner_alive", return_value=True):
            self.assertEqual(self._status(), rp.EXIT_OK)

    def test_status_reports_dead_runner(self) -> None:
        rp.write_state(self.run_dir, "validating", pid=999999)
        (self.run_dir / rp.RUNNER_OUT).write_text("")
        with mock.patch.object(rp, "runner_alive", return_value=False):
            self.assertEqual(self._status(), rp.EXIT_ERROR)

    def test_runner_alive_rejects_non_positive_pids(self) -> None:
        self.assertFalse(rp.runner_alive(0))
        self.assertFalse(rp.runner_alive(-1))
        self.assertFalse(rp.runner_alive(None))

    def _write_results(self, status: str) -> None:
        rp.write_results(
            self.run_dir,
            {
                "pipelines": [
                    {"id": "x", "stages": {m: {"status": status} for m in rp.MODES}}
                ]
            },
        )

    def test_reuse_allowed_only_if_no_stage_started(self) -> None:
        self._write_results(rp.PENDING)
        with mock.patch("sys.stdout"):
            rp._refuse_reuse(self.run_dir)
        self._write_results(rp.PASS)
        with self.assertRaises(rp.RegressionError):
            rp._refuse_reuse(self.run_dir)

    def test_reuse_refused_while_another_runner_is_active(self) -> None:
        rp.write_state(self.run_dir, "running", pid=12345)
        with (
            mock.patch.object(rp, "runner_alive", return_value=True),
            self.assertRaises(rp.RegressionError),
        ):
            rp._refuse_reuse(self.run_dir)


class ReportTests(unittest.TestCase):
    def test_overall_status_and_report_tables(self) -> None:
        results = {
            "run_id": "r1",
            "run_dir": "/runs/r1",
            "started_at": "t0",
            "finished_at": "t1",
            "duration_s": 75,
            "environment": {
                "git_branch": "b",
                "git_commit": "c",
                "docker_image": "img",
            },
            "validation_checks": {"adb_serial": "SER", "device_model": "Board"},
            "inputs": {"repo_root": "/repo", "modes": ["qc", "mAP", "test"]},
            "pipelines": [
                {
                    "num": 16,
                    "id": "yolov26-litert-fp32",
                    "type": "yolov26",
                    "runtime": "litert",
                    "precision": "fp32",
                    "stages": {
                        "qc": {
                            "status": rp.PASS,
                            "exit_code": 0,
                            "duration_s": 60,
                            "metrics": {"artifact": "/x/m.tflite", "size_bytes": 2048},
                            "log": "logs/yolov26-litert-fp32/qc.log",
                        },
                        "mAP": {
                            "status": rp.FAIL,
                            "exit_code": 1,
                            "duration_s": 5,
                            "error_excerpt": "[error] boom",
                            "command": "./docker/iqf run mAP",
                            "log": "logs/yolov26-litert-fp32/mAP.log",
                            "log_tail": ["[error] boom"],
                        },
                        "test": {"status": rp.SKIPPED, "reason": "test not selected"},
                    },
                }
            ],
        }
        self.assertEqual(rp.overall_status(results["pipelines"][0]["stages"]), rp.FAIL)
        report = rp.render_report(results)
        self.assertIn("| No | Pipeline | qc | mAP | test | Overall |", report)
        self.assertIn(
            "| 16 | yolov26-litert-fp32 | PASS | FAIL | SKIPPED | FAIL |", report
        )
        self.assertIn("## 16. yolov26-litert-fp32", report)
        self.assertIn("| Mode | Status | Exit | Duration | Key result | Log |", report)
        self.assertIn("m.tflite (2.0 KB)", report)
        self.assertIn("### yolov26-litert-fp32 : mAP — FAIL (exit 1)", report)
        self.assertEqual(
            rp.overall_status({m: {"status": rp.PASS} for m in rp.MODES}), rp.PASS
        )


if __name__ == "__main__":
    unittest.main()
