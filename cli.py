#!/usr/bin/env python3
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
import argparse
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

from tool.test_map import run_fp_int_pair_map_eval
from yolo_models.yolov10 import YOLOV10_TEST_DEFAULTS, YoloV10Pipeline
from yolo_models.yolov11 import YOLOV11_TEST_DEFAULTS, YoloV11Pipeline
from yolo_models.yolov26 import YOLOV26_TEST_DEFAULTS, YoloV26Pipeline

DEFAULT_REMOTE_RUNNER_LOCAL = str(
    Path(__file__).resolve().parent / "tool" / "remote_tflite_raw_runner.py"
)
DEFAULT_MAP_RESULTS_DIR = Path(__file__).resolve().parent / "out" / "mAP_results"
DEFAULT_QC_RESULTS_DIR = Path(__file__).resolve().parent / "out" / "model"
DEFAULT_TEST_RESULTS_DIR = Path(__file__).resolve().parent / "out" / "test"
ANSI_YELLOW = "\033[33m"
ANSI_RESET = "\033[0m"
WARNING_HOLD_SECONDS = 1.2


PIPELINES = {
    "yolov10": YoloV10Pipeline,
    "yolov11": YoloV11Pipeline,
    "yolov26": YoloV26Pipeline,
}

TEST_DEFAULTS = {
    "yolov10": YOLOV10_TEST_DEFAULTS,
    "yolov11": YOLOV11_TEST_DEFAULTS,
    "yolov26": YOLOV26_TEST_DEFAULTS,
}


def get_pipeline(model_type: str):
    if model_type not in PIPELINES:
        raise SystemExit(
            f"[error] Unsupported --type {model_type}. "
            f"Use one of: {list(PIPELINES.keys())}"
        )
    return PIPELINES[model_type]()


def show_warning(message: str, hold_seconds: float = WARNING_HOLD_SECONDS) -> None:
    print(f"{ANSI_YELLOW}{message}{ANSI_RESET}", file=sys.stderr, flush=True)
    if hold_seconds > 0:
        time.sleep(hold_seconds)


def parse_args():
    mode_help = (
        "Mode-specific requirements:\n"
        "\n"
        "QC Mode\n"
        "  Required:\n"
        "    --mode qc --type --model --calib_dir\n"
        "  Optional:\n"
        "    --output --max_calib --qc-head --qc-quant-scheme\n"
        "\n"
        "mAP Mode\n"
        "  Required:\n"
        "    --mode mAP --type --annotations --images --fp-model --int-model\n"
        "  Optional:\n"
        "    --output_text --conf --fp-head --nms --max-det --max-images\n"
        "    --adb-serial --remote-workdir --remote-runner-local\n"
        "    --remote-runner-remote\n"
        "    --qnn-lib --backend --no-qnn\n"
        "\n"
        "Test Mode\n"
        "  Required:\n"
        "    --mode test --type --model --yaml\n"
        "    and exactly one of: --image OR --images\n"
        "  Optional:\n"
        "    --out/--output --conf --nms --topk --max-det --postprocess-flow\n"
        "    --o2o-nms --disable-int8-prefilter\n"
        "  Optional (only when --adb is used):\n"
        "    --adb-serial --remote-workdir --qnn-lib --backend --no-qnn\n"
    )

    p = argparse.ArgumentParser(
        "iQ-Foundry",
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Mode overview:\n"
            "  qc   : quantize/convert a model to TFLite\n"
            "  mAP  : evaluate FP vs INT mAP@0.5 pair\n"
            "  test : run int8 inference (IQ9-native or via adb)"
        ),
        epilog=mode_help,
    )

    common = p.add_argument_group("Global Required")
    common.add_argument(
        "--mode", required=True, choices=["qc", "test", "mAP"], help="Execution mode"
    )
    common.add_argument(
        "--type",
        required=True,
        choices=list(PIPELINES.keys()),
        help="Model type: yolov10 | yolov11 | yolov26",
    )
    common.add_argument(
        "--model", help="Model path\nRequired in: qc (.pt), test (.tflite)"
    )

    shared = p.add_argument_group("Shared Optional")
    shared.add_argument(
        "--images",
        help="Image directory\nRequired in: mAP; test requires one of --images/--image",
    )

    qc = p.add_argument_group("QC Args (--mode qc)")
    qc.add_argument(
        "--calib_dir", default=None, help="Required: calibration image folder"
    )
    qc.add_argument(
        "--output",
        help=(
            "Optional: output .tflite path\n"
            "Default: out/model/<type>/<type>_int8_<timestamp>.tflite"
        ),
    )
    qc.add_argument(
        "--max_calib",
        type=int,
        default=200,
        help="Optional: max calibration images (default: 200)",
    )
    qc.add_argument(
        "--qc-head",
        choices=["one2many", "one2one"],
        default=None,
        help="Optional: head override for yolov10/yolov26 (ignored for yolov11)",
    )
    qc.add_argument(
        "--qc-quant-scheme",
        choices=["mse", "minmax"],
        default=None,
        help=(
            "Optional: quant scheme override\n"
            "Defaults: yolov10=mse, yolov11=minmax, yolov26=mse"
        ),
    )

    map_eval = p.add_argument_group("mAP Args (--mode mAP)")
    map_eval.add_argument(
        "--annotations",
        help=(
            "Required: COCO annotations JSON or custom annotation directory "
            "with lables in 'txt' or 'xml' format"
        ),
    )
    map_eval.add_argument(
        "--fp-model", dest="fp_model", help="Required: FP .pt model path"
    )
    map_eval.add_argument(
        "--int-model", dest="int_model", help="Required: INT .tflite model path"
    )
    map_eval.add_argument(
        "--output_text",
        help=(
            "Optional: output text report path\n"
            "Default: out/mAP_results/<type>/<type>_mAP_result_<timestamp>.txt"
        ),
    )
    map_eval.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Optional: pre-NMS confidence\nDefaults: mAP=0.25, test=model default",
    )
    map_eval.add_argument(
        "--fp-head",
        choices=["one2many", "one2one"],
        default=None,
        help="Optional: FP head override for yolov10/yolov26",
    )
    map_eval.add_argument(
        "--nms",
        type=float,
        default=None,
        help="Optional: NMS IoU threshold\nDefaults: mAP=0.7, test=model default",
    )
    map_eval.add_argument(
        "--max-det",
        dest="max_det",
        type=int,
        default=None,
        help=(
            "Optional: max detections per image\nDefaults: mAP=300, test=model default"
        ),
    )
    map_eval.add_argument(
        "--max-images",
        dest="max_images",
        type=int,
        default=300,
        help="Optional: number of images to process (default: 300)",
    )
    adb_opts = p.add_argument_group("ADB Optional (mAP + test --adb)")
    adb_opts.add_argument(
        "--adb-serial",
        default=None,
        help="Optional: ADB device serial for target device",
    )
    adb_opts.add_argument(
        "--remote-workdir",
        dest="remote_workdir",
        default="/data/local/tmp/yolo_map_eval",
        help="Optional: remote working directory on device",
    )
    adb_opts.add_argument(
        "--remote-runner-local",
        dest="remote_runner_local",
        default=DEFAULT_REMOTE_RUNNER_LOCAL,
        help="Optional: local remote runner path (used by mAP mode)",
    )
    adb_opts.add_argument(
        "--remote-runner-remote",
        dest="remote_runner_remote",
        default="/data/local/tmp/yolo_map_eval/remote_tflite_raw_runner.py",
        help="Optional: remote runner path on device (used by mAP mode)",
    )
    adb_opts.add_argument(
        "--qnn-lib",
        dest="qnn_lib",
        default="/usr/lib/libQnnTFLiteDelegate.so",
        help="Optional: QNN delegate library path on device",
    )
    adb_opts.add_argument(
        "--backend", default="htp", help="Optional: QNN backend type (default: htp)"
    )
    adb_opts.add_argument(
        "--no-qnn",
        dest="no_qnn",
        action="store_true",
        help="Optional: disable QNN delegate",
    )

    test = p.add_argument_group("Test Args (--mode test)")
    test.add_argument(
        "--image", help="Single image path\nRequired if --images is not provided"
    )
    test.add_argument("--yaml", help="Required: class names yaml path")
    test.add_argument(
        "--adb",
        action="store_true",
        help="Optional: run inference via adb push/shell/pull",
    )
    test.add_argument(
        "--out",
        dest="output",
        help="Optional: alias of --output for test output directory",
    )
    test.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Optional: top-k before NMS (default: model setting)",
    )
    test.add_argument(
        "--postprocess-flow",
        dest="postprocess_flow",
        choices=["auto", "default", "o2o", "o2m"],
        default=None,
        help="Optional: postprocess flow override (default: model setting)",
    )
    test.add_argument(
        "--o2o-nms",
        dest="o2o_nms",
        action="store_true",
        help="Optional: when flow=o2o, enable class-wise NMS",
    )
    test.add_argument(
        "--disable-int8-prefilter",
        dest="disable_int8_prefilter",
        action="store_true",
        help="Optional: disable int8 class prefilter in postprocess",
    )

    if len(sys.argv) == 1:
        p.print_help()
        raise SystemExit(0)

    return p.parse_args()


def resolve_default_fp_head(model_type: str, fp_head_override: str | None) -> str:
    if model_type == "yolov11":
        if fp_head_override is not None:
            show_warning(
                "[warn] --fp-head is not applicable for yolov11 "
                "and will be ignored (using default head)."
            )
        return "default"

    if fp_head_override is not None:
        return fp_head_override
    if model_type == "yolov10":
        return "one2many"
    if model_type == "yolov26":
        return "one2many"
    return "one2many"


def resolve_default_qc_head(model_type: str, qc_head_override: str | None) -> str:
    if model_type == "yolov11":
        if qc_head_override is not None:
            show_warning(
                "[warn] --qc-head ignored for yolov11 "
                "(yolov11 supports only the default head)"
            )
        return "default"

    if qc_head_override is not None:
        return qc_head_override

    if model_type == "yolov10":
        return "one2many"
    if model_type == "yolov26":
        return "one2many"
    return "one2many"


def resolve_default_qc_quant_scheme(
    model_type: str, qc_quant_override: str | None
) -> str:
    if qc_quant_override is not None:
        return qc_quant_override
    if model_type == "yolov11":
        return "minmax"
    return "mse"


def resolve_default_qc_output_path(
    model_type: str, output_override: str | None, quant_label: str = "int8"
) -> str:
    if output_override:
        return output_override
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(
        DEFAULT_QC_RESULTS_DIR / model_type / f"{model_type}_{quant_label}_{ts}.tflite"
    )


def resolve_default_output_text(
    model_type: str, output_text_override: str | None
) -> str:
    if output_text_override:
        return output_text_override
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(
        DEFAULT_MAP_RESULTS_DIR / model_type / f"{model_type}_mAP_result_{ts}.txt"
    )


def resolve_default_test_output_path(
    model_type: str, output_override: str | None
) -> str:
    if output_override:
        return output_override
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(DEFAULT_TEST_RESULTS_DIR / model_type / f"{model_type}_inference_{ts}")


def resolve_default_map_conf(conf_override: float | None) -> float:
    if conf_override is not None:
        return conf_override
    return 0.25


def resolve_default_map_nms(nms_override: float | None) -> float:
    if nms_override is not None:
        return nms_override
    return 0.7


def resolve_default_map_max_det(max_det_override: int | None) -> int:
    if max_det_override is not None:
        return max_det_override
    return 300


def is_iq9_runtime() -> bool:
    return platform.machine().lower() in {"aarch64", "arm64"}


def _run_qc_mode(args: argparse.Namespace) -> None:
    pipe = get_pipeline(args.type)
    if not args.model:
        raise SystemExit("[error] qc requires --model")
    if not args.calib_dir:
        raise SystemExit("[error] qc requires --calib_dir (calibration image folder)")

    effective_qc_head = resolve_default_qc_head(args.type, args.qc_head)
    effective_qc_quant_scheme = resolve_default_qc_quant_scheme(
        args.type, args.qc_quant_scheme
    )
    effective_output = resolve_default_qc_output_path(
        args.type, args.output, quant_label="int8"
    )

    Path(effective_output).parent.mkdir(parents=True, exist_ok=True)
    pipe.quantize_convert(
        model_path=args.model,
        out_tflite=effective_output,
        calib_dir=args.calib_dir,
        max_calib=args.max_calib,
        qc_head=effective_qc_head,
        qc_quant_scheme=effective_qc_quant_scheme,
    )
    print(f"[ok] wrote: {effective_output}")


def _run_map_mode(args: argparse.Namespace) -> None:
    if not args.annotations or not args.images:
        raise SystemExit("[error] mAP requires --annotations and --images")
    if not args.fp_model or not args.int_model:
        raise SystemExit("[error] mAP requires --fp-model and --int-model")

    effective_fp_head = resolve_default_fp_head(args.type, args.fp_head)
    effective_output_text = resolve_default_output_text(args.type, args.output_text)
    effective_conf = resolve_default_map_conf(args.conf)
    effective_nms = resolve_default_map_nms(args.nms)
    effective_max_det = resolve_default_map_max_det(args.max_det)
    try:
        run_fp_int_pair_map_eval(
            model_type=args.type,
            fp_model=args.fp_model,
            int_model=args.int_model,
            annotations=args.annotations,
            images=args.images,
            output_text=effective_output_text,
            conf=effective_conf,
            nms=effective_nms,
            max_det=effective_max_det,
            max_images=args.max_images,
            fp_head=effective_fp_head,
            adb_serial=args.adb_serial,
            remote_workdir=args.remote_workdir,
            remote_runner_local=args.remote_runner_local,
            remote_runner_remote=args.remote_runner_remote,
            qnn_lib=args.qnn_lib,
            backend=args.backend,
            no_qnn=args.no_qnn,
        )
    except (
        FileNotFoundError,
        PermissionError,
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        raise SystemExit(f"[error] {exc}") from exc
    print(f"[ok] wrote: {effective_output_text}")


def _resolve_test_mode_options(
    args: argparse.Namespace,
) -> tuple[dict, str, float, float, int, int, str, bool]:
    defaults = TEST_DEFAULTS[args.type]
    effective_output = resolve_default_test_output_path(args.type, args.output)
    effective_conf = args.conf if args.conf is not None else defaults["conf_thres"]
    effective_nms = args.nms if args.nms is not None else defaults["iou_thres"]
    effective_topk = args.topk if args.topk is not None else defaults["topk"]
    effective_max_det = (
        args.max_det if args.max_det is not None else defaults["max_det"]
    )
    effective_postprocess_flow = args.postprocess_flow or "auto"
    effective_o2o_nms = args.o2o_nms

    if args.type == "yolov11":
        if effective_postprocess_flow in ("o2o", "o2m"):
            show_warning(
                "[warn] yolov11 supports only the default postprocess flow. "
                f"Ignoring --postprocess-flow {effective_postprocess_flow} "
                "and using default."
            )
            effective_postprocess_flow = "default"
        if effective_o2o_nms:
            show_warning(
                "[warn] --o2o-nms is not applicable for yolov11 and will be ignored."
            )
            effective_o2o_nms = False

    return (
        defaults,
        effective_output,
        effective_conf,
        effective_nms,
        effective_topk,
        effective_max_det,
        effective_postprocess_flow,
        effective_o2o_nms,
    )


def _run_test_mode(args: argparse.Namespace) -> None:
    from tool.inference import run_test_inference_adb, run_test_inference_local

    if not args.model:
        raise SystemExit("[error] test requires --model (INT .tflite)")
    if not args.yaml:
        raise SystemExit("[error] test requires --yaml")
    if bool(args.images) == bool(args.image):
        raise SystemExit("[error] test requires exactly one of --images or --image")

    show_warning(
        "[warn] Please verify that --type and --model match before running inference."
    )
    (
        defaults,
        effective_output,
        effective_conf,
        effective_nms,
        effective_topk,
        effective_max_det,
        effective_postprocess_flow,
        effective_o2o_nms,
    ) = _resolve_test_mode_options(args)

    runner = run_test_inference_adb if args.adb else run_test_inference_local
    runner_kwargs = {
        "model_path": args.model,
        "yaml_path": args.yaml,
        "output_dir": effective_output,
        "model_type": args.type,
        "default_flow": defaults["default_flow"],
        "conf_thres": effective_conf,
        "iou_thres": effective_nms,
        "topk": effective_topk,
        "max_det": effective_max_det,
        "image_dir": args.images,
        "image_path": args.image,
        "postprocess_flow": effective_postprocess_flow,
        "o2o_nms": effective_o2o_nms,
        "disable_int8_prefilter": args.disable_int8_prefilter,
        "no_qnn": args.no_qnn,
        "qnn_lib": args.qnn_lib,
        "backend": args.backend,
    }
    if args.adb:
        runner_kwargs.update(
            adb_serial=args.adb_serial,
            remote_workdir=args.remote_workdir,
        )
    else:
        runner_kwargs["enforce_iq9_native"] = True

    try:
        runner(**runner_kwargs)
    except (
        FileNotFoundError,
        PermissionError,
        ValueError,
        KeyError,
        RuntimeError,
        OSError,
    ) as exc:
        raise SystemExit(f"[error] {exc}") from exc
    print(f"[ok] wrote: {effective_output}")


def main():
    args = parse_args()
    if is_iq9_runtime() and (args.mode != "test" or args.adb):
        show_warning(
            "[warn] IQ9 runtime supports only '--mode test' without '--adb'. "
            "Use an x86 host for qc/mAP or adb-orchestrated test runs."
        )
        raise SystemExit(1)

    if args.mode == "qc":
        _run_qc_mode(args)
        return
    if args.mode == "mAP":
        _run_map_mode(args)
        return
    if args.mode == "test":
        _run_test_mode(args)
        return


if __name__ == "__main__":
    main()
