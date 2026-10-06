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
"""Offline tests for the iqf-assistant helper scripts (no docker, no device, no network).

Run from the repository root:
    python3 -m pytest .agents/skills/iqf-assistant/tests -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

SKILL_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_DIR / "scripts"
REPO_ROOT = SKILL_DIR.parents[2]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cs = load("check_setup")
hl = load("hub_login")
job = load("iqf_job")
pm = load("prepare_map_data")

MODEL_NAMES = ["person", "bicycle", "car"]


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
    return path


def image(path: Path, size=(100, 50), fmt=None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(path, format=fmt)
    return path


class TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()


# ----------------------------
# check_setup.py
# ----------------------------


class CheckSetupParserTests(unittest.TestCase):
    def test_os_release(self) -> None:
        info = cs.parse_os_release(
            'NAME="Ubuntu"\nVERSION_ID="22.04"\nID=ubuntu\n# c\n'
        )
        self.assertEqual(info["ID"], "ubuntu")
        self.assertEqual(info["VERSION_ID"], "22.04")

    def test_meminfo(self) -> None:
        self.assertAlmostEqual(
            cs.parse_meminfo_gib("MemTotal:       16777216 kB\n"), 16.0
        )
        self.assertIsNone(cs.parse_meminfo_gib("nothing"))

    def test_adb_devices(self) -> None:
        out = (
            "* daemon started successfully\nList of devices attached\n"
            "abc123  device usb:1-1 product:iq9 model:EXMP_Q911\n"
            "def456  offline\n"
        )
        devices = cs.parse_adb_devices(out)
        self.assertEqual([d["serial"] for d in devices], ["abc123", "def456"])
        self.assertEqual(devices[0]["model"], "EXMP_Q911")

    def test_wsl_conf_systemd(self) -> None:
        self.assertTrue(cs.wsl_conf_has_systemd("[boot]\nsystemd = true\n"))
        self.assertFalse(cs.wsl_conf_has_systemd("[boot]\n# systemd=true\n"))
        self.assertFalse(cs.wsl_conf_has_systemd("[user]\nsystemd=true\n"))

    def test_is_wsl(self) -> None:
        self.assertTrue(cs.is_wsl("Linux 5.15 microsoft-standard-WSL2", {}))
        self.assertTrue(cs.is_wsl("", {"WSL_DISTRO_NAME": "Ubuntu-22.04"}))
        self.assertFalse(cs.is_wsl("Linux 6.8.0-generic", {}))

    def test_check_device_states(self) -> None:
        one = [{"serial": "a", "state": "device", "model": "Q911"}]
        self.assertEqual(cs.check_device(one, "host adb").status, cs.OK)
        two = one + [{"serial": "b", "state": "device"}]
        self.assertEqual(cs.check_device(two, "host adb").status, cs.WARN)
        offline = [{"serial": "a", "state": "offline"}]
        result = cs.check_device(offline, "host adb")
        self.assertEqual(result.status, cs.MISSING)
        self.assertIn("offline", result.detail)
        self.assertEqual(cs.check_device([], "none").status, cs.MISSING)

    def test_readiness_per_mode(self) -> None:
        checks = [
            cs.Check(i, cs.OK, "")
            for i in ("os", "repo", "docker_cli", "docker_group", "docker_daemon")
        ]
        checks += [
            cs.Check("image", cs.OK, ""),
            cs.Check("qai_hub", cs.OK, ""),
            cs.Check("usb", cs.OK, ""),
            cs.Check("device", cs.MISSING, ""),
        ]
        self.assertEqual(
            cs.readiness(checks), {"qc": True, "mAP": False, "test": False}
        )
        checks[-3] = cs.Check("qai_hub", cs.MISSING, "")
        self.assertFalse(cs.readiness(checks)["qc"])


class CheckSetupQaiHubFileTests(TempDirTest):
    def test_login_file_is_never_opened(self) -> None:
        hub = self.tmp / ".qai_hub"
        hub.mkdir(mode=0o700)
        ini = hub / "client.ini"
        ini.write_text("[api]\napi_token = SECRET\n")
        ini.chmod(0o600)
        with (
            mock.patch.object(cs, "QAI_HUB_DIR", hub),
            mock.patch("builtins.open", side_effect=AssertionError("opened")),
            mock.patch.object(Path, "read_text", side_effect=AssertionError("read")),
        ):
            result = cs.check_qai_hub_file()
        self.assertEqual(result.status, cs.OK)
        self.assertNotIn("SECRET", result.detail)

    def test_loose_permissions_warn_and_missing(self) -> None:
        hub = self.tmp / ".qai_hub"
        with mock.patch.object(cs, "QAI_HUB_DIR", hub):
            self.assertEqual(cs.check_qai_hub_file().status, cs.MISSING)
            hub.mkdir()
            (hub / "client.ini").write_text("x")
            (hub / "client.ini").chmod(0o644)
            self.assertEqual(cs.check_qai_hub_file().status, cs.WARN)


# ----------------------------
# hub_login.py
# ----------------------------

TOKEN = "abcdefghijklmnopqrstuvwxyz0123456789"


class HubLoginTests(TempDirTest):
    def test_token_problem(self) -> None:
        self.assertIsNotNone(hl.token_problem(""))
        self.assertIsNotNone(hl.token_problem("short"))
        self.assertIsNotNone(hl.token_problem("abcdefgh ijklmnopqrstu"))
        self.assertIsNone(hl.token_problem(TOKEN))

    def test_masker_handles_split_chunks(self) -> None:
        masker = hl.TokenMasker(TOKEN)
        text = f"key={TOKEN} done {TOKEN}"
        chunks = [text[i : i + 5] for i in range(0, len(text), 5)]
        out = "".join(masker.feed(c) for c in chunks) + masker.flush()
        self.assertEqual(out, "key=*** done ***")

    def _fake_repo(self) -> Path:
        repo = self.tmp / "repo"
        script = write(
            repo / "qaihub_login.sh",
            """\
            #!/usr/bin/env bash
            # fake: echo the key back (must be masked) and report whether the env leaked
            echo "configuring with $2"
            if [ -n "${IQF_QAI_HUB_TOKEN:-}" ]; then echo ENV_HAS_TOKEN=yes; else echo ENV_HAS_TOKEN=no; fi
            mkdir -p "$HOME/.qai_hub" && echo "[api]" > "$HOME/.qai_hub/client.ini"
            chmod 644 "$HOME/.qai_hub/client.ini"
            """,
        )
        script.chmod(0o755)
        return repo

    def _run(self, repo: Path, token: str | None):
        env = {k: v for k, v in os.environ.items() if k != hl.TOKEN_ENV}
        env["HOME"] = str(self.tmp / "home")
        if token is not None:
            env[hl.TOKEN_ENV] = token
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "hub_login.py"), "--repo", str(repo)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def test_login_masks_token_and_tightens_permissions(self) -> None:
        repo = self._fake_repo()
        result = self._run(repo, TOKEN)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertNotIn(TOKEN, output)
        self.assertIn("configuring with ***", output)
        self.assertIn("ENV_HAS_TOKEN=no", output)
        self.assertIn("[ok] QAI Hub login saved", output)
        hub = self.tmp / "home" / ".qai_hub"
        self.assertEqual(stat.S_IMODE(hub.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((hub / "client.ini").stat().st_mode), 0o600)

    def test_missing_or_bad_token_runs_nothing(self) -> None:
        repo = self._fake_repo()
        for token in (None, "too short"):
            result = self._run(repo, token)
            self.assertEqual(result.returncode, 1)
            self.assertIn("[error]", result.stdout)
        self.assertFalse((self.tmp / "home" / ".qai_hub").exists())


# ----------------------------
# iqf_job.py
# ----------------------------


class JobTests(TempDirTest):
    def _job(self, *argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(SCRIPTS / "iqf_job.py"), *argv],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    @staticmethod
    def _value(output: str, key: str) -> str:
        for line in output.splitlines():
            if line.startswith(f"{key}: "):
                return line.split(": ", 1)[1]
        raise AssertionError(f"{key} not in output:\n{output}")

    def _wait_finished(self, job_dir: str) -> str:
        for _ in range(100):
            out = self._job("status", job_dir).stdout
            if "STATE: finished" in out:
                return out
            time.sleep(0.1)
        raise AssertionError("job did not finish")

    def test_start_status_and_outputs(self) -> None:
        cmd = "echo '[ok] wrote: /tmp/x/model.tflite'; echo progress; exit 3"
        started = self._job(
            "start",
            "--repo",
            str(self.tmp),
            "--name",
            "qc",
            "--wait-s",
            "0",
            "--",
            "bash",
            "-c",
            cmd,
        )
        self.assertEqual(started.returncode, 0, started.stdout)
        job_dir = self._value(started.stdout, "JOB_DIR")
        self.assertTrue(
            job_dir.startswith(str(self.tmp / "out" / "iqf-assistant" / "jobs"))
        )
        out = self._wait_finished(job_dir)
        self.assertIn("EXIT_CODE: 3", out)
        self.assertIn("OUTPUT: /tmp/x/model.tflite", out)
        self.assertIn("progress", out)

    def test_one_job_at_a_time_and_stop(self) -> None:
        first = self._job(
            "start",
            "--repo",
            str(self.tmp),
            "--name",
            "long",
            "--wait-s",
            "0.5",
            "--",
            "sleep",
            "30",
        )
        job_dir = self._value(first.stdout, "JOB_DIR")
        self.assertEqual(self._value(first.stdout, "STATE"), "running")
        second = self._job(
            "start", "--repo", str(self.tmp), "--name", "x", "--", "true"
        )
        self.assertEqual(second.returncode, 1)
        self.assertIn("another job is still running", second.stdout)
        stopped = self._job("stop", job_dir)
        self.assertIn("STATE: finished", stopped.stdout)

    def test_status_rejects_non_job_folder(self) -> None:
        self.assertEqual(self._job("status", str(self.tmp)).returncode, 1)


# ----------------------------
# prepare_map_data.py
# ----------------------------


class ClassMappingTests(unittest.TestCase):
    def test_exact_loose_user_and_unknown(self) -> None:
        mapping, notes, unknown = pm.resolve_class_mapping(
            ["person", "Bi-cycle", "auto", "dog"], MODEL_NAMES, {"auto": "car"}, False
        )
        self.assertEqual(mapping, {"person": 0, "Bi-cycle": 1, "auto": 2})
        self.assertEqual(unknown, ["dog"])
        self.assertTrue(any("Bi-cycle" in n for n in notes))

    def test_drop(self) -> None:
        mapping, _, unknown = pm.resolve_class_mapping(
            ["dog", "cat"], MODEL_NAMES, {"cat": "drop"}, True
        )
        self.assertEqual(mapping, {"dog": None, "cat": None})
        self.assertEqual(unknown, [])

    def test_bad_user_target(self) -> None:
        with self.assertRaises(ValueError):
            pm.resolve_class_mapping(["a"], MODEL_NAMES, {"a": "unicorn"}, False)

    def test_parse_class_map(self) -> None:
        self.assertEqual(
            pm.parse_class_map("Person = person, x=drop"),
            {"Person": "person", "x": "drop"},
        )


class NamesFileTests(TempDirTest):
    def test_yaml_dict_list_and_txt(self) -> None:
        a = write(self.tmp / "a.yaml", "names:\n  0: car\n  1: person\n")
        b = write(self.tmp / "b.yaml", "names: [car, person]\n")
        c = write(self.tmp / "classes.txt", "car\nperson\n\n")
        for path in (a, b, c):
            self.assertEqual(pm.load_names_file(path), ["car", "person"])

    def test_fallback_parser_without_pyyaml(self) -> None:
        text = "nc: 2\nnames:\n  - car\n  - person\n"
        self.assertEqual(pm._parse_simple_yaml_names(text), ["car", "person"])
        self.assertEqual(
            pm._parse_simple_yaml_names("names:\n  0: car\n  1: 'person'\n"),
            {0: "car", 1: "person"},
        )


class ScanAndBuildTests(TempDirTest):
    def test_ultralytics_layout_reorder_polygon_and_confidence(self) -> None:
        root = self.tmp / "ds"
        image(root / "images" / "val" / "a.jpg", (100, 50))
        image(root / "images" / "val" / "b.jpg", (100, 50))
        image(root / "images" / "train" / "c.jpg")
        # dataset order: car=0, person=1 (model order: person=0, car=2)
        write(root / "data.yaml", "names:\n  0: car\n  1: person\n")
        write(
            root / "labels" / "val" / "a.txt",
            "1 0.5 0.5 0.2 0.4\n0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n",
        )
        write(root / "labels" / "val" / "b.txt", "0 0.5 0.5 0.2 0.2 0.91\n")
        write(root / "labels" / "train" / "c.txt", "0 0.5 0.5 0.1 0.1\n")

        with self.assertRaises(pm.UserInputNeeded):
            pm.scan(root / "labels", root / "images")
        ds = pm.scan(root / "labels", root / "images", split="val")
        self.assertEqual(ds.format_name, "yolo")
        self.assertEqual(ds.dataset_names, ["car", "person"])
        self.assertTrue(pm.silent_problems(ds, MODEL_NAMES))

        result = pm.build_coco(ds, MODEL_NAMES)
        coco = result.coco
        self.assertEqual(
            [im["file_name"] for im in coco["images"]], ["val/a.jpg", "val/b.jpg"]
        )
        self.assertEqual([c["name"] for c in coco["categories"]], MODEL_NAMES)
        anns = coco["annotations"]
        self.assertEqual(len(anns), 3)
        # person box (dataset id 1 -> model person, category 1)
        self.assertEqual(anns[0]["category_id"], 1)
        self.assertEqual(anns[0]["bbox"], [40.0, 15.0, 20.0, 20.0])
        # polygon -> box, car (dataset id 0 -> model index 2 -> category 3)
        self.assertEqual(anns[1]["category_id"], 3)
        self.assertEqual(anns[1]["bbox"], [10.0, 5.0, 20.0, 10.0])
        # confidence column dropped
        self.assertEqual(anns[2]["bbox"], [40.0, 20.0, 20.0, 10.0])
        self.assertTrue(any("polygon" in n for n in result.notes))
        self.assertTrue(any("re-numbered" in n for n in result.notes))

    def test_yolo_without_names_uses_model_order_and_flags_out_of_range(self) -> None:
        image(self.tmp / "img" / "a.png")
        write(self.tmp / "lbl" / "a.txt", "2 0.5 0.5 0.5 0.5\n")
        ds = pm.scan(self.tmp / "lbl", self.tmp / "img")
        self.assertIsNone(ds.dataset_names)
        self.assertEqual(
            pm.build_coco(ds, MODEL_NAMES).coco["annotations"][0]["category_id"], 3
        )
        write(self.tmp / "lbl" / "a.txt", "7 0.5 0.5 0.5 0.5\n")
        with self.assertRaises(pm.UserInputNeeded):
            pm.build_coco(pm.scan(self.tmp / "lbl", self.tmp / "img"), MODEL_NAMES)

    def test_voc_nested_with_case_mismatch(self) -> None:
        root = self.tmp / "voc"
        image(root / "JPEGImages" / "sub" / "x.jpg", (200, 100))
        write(
            root / "Annotations" / "sub" / "x.xml",
            """\
            <annotation><filename>x.jpg</filename>
              <object><name>Person</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>80</ymax></bndbox></object>
              <object><name>car</name><bndbox><xmin>-5</xmin><ymin>0</ymin><xmax>250</xmax><ymax>50</ymax></bndbox></object>
            </annotation>
            """,
        )
        ds = pm.scan(root / "Annotations", root / "JPEGImages")
        self.assertEqual(ds.format_name, "voc")
        self.assertTrue(ds.nested_images)
        result = pm.build_coco(ds, MODEL_NAMES)
        anns = result.coco["annotations"]
        self.assertEqual(anns[0]["bbox"], [10.0, 20.0, 40.0, 60.0])
        self.assertEqual(
            anns[1]["bbox"], [0.0, 0.0, 200.0, 50.0]
        )  # clipped to the image
        self.assertTrue(any("'Person'->'person'" in n for n in result.notes))

    def test_labelme_unknown_class_needs_input_then_drop(self) -> None:
        root = self.tmp / "lm"
        image(root / "p.webp", (64, 64), fmt="WEBP")
        image(root / "q.jpg", (64, 64))
        shapes = [
            {
                "label": "person",
                "points": [[1, 2], [30, 40]],
                "shape_type": "rectangle",
            },
            {
                "label": "dog",
                "points": [[0, 0], [10, 10], [5, 20]],
                "shape_type": "polygon",
            },
            {"label": "person", "points": [[3, 3]], "shape_type": "point"},
        ]
        write(root / "p.json", json.dumps({"shapes": shapes, "imagePath": "p.webp"}))
        write(root / "q.json", json.dumps({"shapes": [], "imagePath": "q.jpg"}))
        ds = pm.scan(root, root)
        self.assertEqual(ds.format_name, "labelme")
        self.assertTrue(ds.convert_images)
        self.assertEqual(ds.skipped_shapes, 1)
        with self.assertRaises(pm.UserInputNeeded) as ctx:
            pm.build_coco(ds, MODEL_NAMES)
        self.assertIn("dog", str(ctx.exception))

        out = self.tmp / "out"
        result = pm.build_coco(
            ds, MODEL_NAMES, drop_unknown=True, out_images_dir=out / "images"
        )
        self.assertEqual(result.images_root, out / "images")
        self.assertEqual(
            sorted(im["file_name"] for im in result.coco["images"]), ["p.png", "q.jpg"]
        )
        self.assertEqual(len(result.coco["annotations"]), 1)
        pm.write_outputs(ds, result, out, MODEL_NAMES)
        self.assertTrue((out / "images" / "p.png").is_file())
        self.assertTrue((out / "images" / "q.jpg").is_file())
        self.assertTrue((out / "annotations_coco.json").is_file())
        self.assertIn("not** changed", (out / "conversion_report.md").read_text())

    def test_cvat(self) -> None:
        image(self.tmp / "img" / "a.jpg", (100, 100))
        write(
            self.tmp / "cvat.xml",
            """\
            <annotations>
              <image id="0" name="a.jpg" width="100" height="100">
                <box label="car" xtl="10" ytl="10" xbr="30" ybr="40"/>
                <polygon label="person" points="1,1;20,1;20,20"/>
              </image>
            </annotations>
            """,
        )
        ds = pm.scan(self.tmp / "cvat.xml", self.tmp / "img")
        self.assertEqual(ds.format_name, "cvat")
        anns = pm.build_coco(ds, MODEL_NAMES).coco["annotations"]
        self.assertEqual([a["category_id"] for a in anns], [3, 1])
        self.assertEqual(anns[1]["bbox"], [1.0, 1.0, 19.0, 19.0])

    def test_coco_fixups(self) -> None:
        image(self.tmp / "img" / "nested" / "a.jpg", (80, 60))
        image(self.tmp / "img" / "b.jpg", (80, 60))
        coco = {
            "images": [
                {"id": 7, "file_name": "a.jpg"},  # wrong path, no size
                {"id": 8, "file_name": "b.jpg", "width": 80, "height": 60},
                {"id": 9, "file_name": "missing.jpg", "width": 1, "height": 1},
            ],
            "annotations": [
                {"id": 1, "image_id": 7, "category_id": 3, "bbox": [1, 2, 3, 4]},
                {"id": 2, "image_id": 8, "category_id": 1, "bbox": [0, 0, 0, 4]},
                {"id": 3, "image_id": 9, "category_id": 1, "bbox": [0, 0, 1, 1]},
            ],
            "categories": [{"id": 1, "name": "Person"}, {"id": 3, "name": "car"}],
        }
        write(self.tmp / "ann.json", json.dumps(coco))
        ds = pm.scan(self.tmp / "ann.json", self.tmp / "img")
        result = pm.build_coco(ds, MODEL_NAMES)
        images = result.coco["images"]
        self.assertEqual(
            images[0], {"id": 1, "file_name": "nested/a.jpg", "width": 80, "height": 60}
        )
        self.assertEqual(len(images), 2)
        anns = result.coco["annotations"]
        self.assertEqual(
            len(anns), 1
        )  # the zero-width box and the missing image are left out
        self.assertEqual(anns[0]["category_id"], 3)
        self.assertTrue(any("not found" in w for w in result.warnings))
        self.assertTrue(any("width/height" in n for n in result.notes))

    def test_unsupported_and_empty(self) -> None:
        write(self.tmp / "labels.csv", "a,b\n")
        image(self.tmp / "img" / "a.jpg")
        with self.assertRaises(pm.Unsupported):
            pm.scan(self.tmp / "labels.csv", self.tmp / "img")
        (self.tmp / "emptylabels").mkdir()
        with self.assertRaises(pm.Unsupported):
            pm.scan(self.tmp / "emptylabels", self.tmp / "img")


class HostSafetyTests(TempDirTest):
    def _convert(self, *extra: str) -> int:
        image(self.tmp / "img" / "a.jpg")
        write(self.tmp / "lbl" / "a.txt", "0 0.5 0.5 0.1 0.1\n")
        (self.tmp / "m.pt").write_bytes(b"x")
        argv = [
            "convert",
            "--annotations",
            str(self.tmp / "lbl"),
            "--images",
            str(self.tmp / "img"),
            "--model",
            str(self.tmp / "m.pt"),
            "--repo",
            str(REPO_ROOT),
            *extra,
        ]
        with mock.patch.object(
            pm.subprocess, "run", side_effect=AssertionError("docker")
        ):
            return pm.main(argv)

    def test_refuses_output_inside_inputs(self) -> None:
        self.assertEqual(
            self._convert("--out", str(self.tmp / "img" / "conv")), pm.EXIT_ERROR
        )
        self.assertFalse((self.tmp / "img" / "conv").exists())

    def test_refuses_existing_output(self) -> None:
        (self.tmp / "exists").mkdir()
        self.assertEqual(
            self._convert("--out", str(self.tmp / "exists")), pm.EXIT_ERROR
        )

    def test_missing_input_path(self) -> None:
        argv = [
            "inspect",
            "--annotations",
            str(self.tmp / "nope"),
            "--images",
            str(self.tmp),
            "--model",
            str(self.tmp / "m.pt"),
            "--repo",
            str(REPO_ROOT),
        ]
        with mock.patch.object(
            pm.subprocess, "run", side_effect=AssertionError("docker")
        ):
            self.assertEqual(pm.main(argv), pm.EXIT_ERROR)


if __name__ == "__main__":
    unittest.main()
