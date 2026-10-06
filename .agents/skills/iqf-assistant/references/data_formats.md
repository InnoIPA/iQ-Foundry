# What each mode accepts

These rules come from the iQ-Foundry code: `cli.py`, `tool/test_map.py`,
`tool/inference_tflite.py` and `docker/iqf_path_mapper.py`. Use them to check the user's inputs
and to explain problems in plain words.

## Models and pairs

**Model families (`--type`):** `yolov10`, `yolov11`, `yolov26`.

**Runtime / precision pairs.** Only these combinations are valid:

| Runtime | Precisions | qc output |
|---|---|---|
| litert | int8, fp32 | `.tflite` |
| onnx | fp32, w8a16 | `.onnx` (mAP and test also take `.onnx.zip`) |
| qairt | int8, w8a16, fp16 | `.bin` |

**How conversion happens:**
- litert and onnx convert through Qualcomm AI Hub, which needs internet and the login.
- qairt converts on this computer, but qc still requires the QAI Hub login file to exist.

## qc (convert)

**Inputs:**
- `--model`: a YOLO `.pt` file of the chosen family.
- `--calib_dir`: required for litert/int8, onnx/w8a16, qairt/int8 and qairt/w8a16; ignored for
  the other pairs.
  - Pictures must be directly in the folder; sub-folders are not read.
  - Any picture format PIL can open is accepted.
  - The pictures are sorted, and the first `--max_calib` are used (default 200).
  - Best results come from 50–200 pictures that look like the real use case.

**Advanced options**, only if the user asks:
- `--qc-head one2many|one2one` (yolov10 and yolov26). Remember it for mAP's `--fp-head`.
- `--qc-quant-scheme mse|minmax`.

## mAP (accuracy check)

**Required:** `--annotations`, `--images`, `--reference-model` (`.pt`), `--converted-model`.

**Defaults:** `--max-images` is 300 (the first N pictures), `--conf` is 0.25. The converted model
runs on the board over adb.

The result file contains `reference_map50`, `converted_map50`,
`abs_delta_converted_minus_reference` and `pct_delta_vs_reference`.

### Label formats mAP mode reads by itself

The class names always come from the `.pt` model. mAP mode does not use a `.yaml` file.

**1. COCO `.json` file.**
- `images` entries need `id`, `file_name`, `width` and `height`. The picture is found at
  `<images>/<file_name>`; sub-folders in `file_name` are fine.
- **Every listed picture must exist**, or the run stops.
- `categories` are matched to the model **by exact name** (capitals matter). The ids do not
  matter, so the standard COCO 91-style ids work.
- A category name the model does not have stops the run.
- `bbox` is `[x, y, width, height]` in pixels.

**2. A folder of YOLO `.txt` files.**
- Each line is `class cx cy w h` (exactly 5 values), normalised to 0–1.
- The class number is the model's class index, starting at 0.
- Label files and pictures must sit **directly** inside their folders (no sub-folders), and
  are paired by file name, e.g. `a.txt` with `a.jpg`.

**3. A folder of Pascal VOC `.xml` files.**
- Each `<object>` has a `<name>` and a `<bndbox>` in pixels.
- `<name>` must exactly match a model class name.
- Files are paired with pictures by file name, with no sub-folders.

**For YOLO and VOC folders:**
- Only `.jpg .jpeg .png .bmp` pictures directly in `--images` are read; others are silently
  skipped.
- A picture without a label file counts as a picture with no objects.

### What `prepare_map_data.py` reformats, and why

The tool always writes one COCO file (`annotations_coco.json`) into a **new** folder. That is
the most robust format for mAP mode. The user's files are never changed.

| Found | Why mAP mode cannot use it as it is | What the reformat does |
|---|---|---|
| YOLO in the Ultralytics layout (`images/val`, `labels/val`, `data.yaml`) or other sub-folders | mAP mode only reads files directly in the folder | Finds every picture and label in sub-folders and pairs them by path, then by name |
| YOLO with a `data.yaml` / `classes.txt` whose class order differs from the model | mAP mode would read the numbers in the model's order and give a **wrong score without any error** | Re-numbers the classes by name |
| YOLO segmentation polygons, or a 6th "confidence" column | mAP mode needs exactly 5 values per line | Turns polygons into boxes and drops the extra column |
| LabelMe `.json` (one per picture) or CVAT `.xml` | Not a format mAP mode reads | Rectangles and polygons become boxes; points and lines are skipped and counted |
| Class names that differ only in capitals, spaces, `_` or `-` (e.g. `Person` and `person`) | Names must match exactly | Matches them to the model's names and lists every match in the report |
| Class names the model does not have | The run would stop | Asks the user to map them (`--class-map`) or leave them out (`--drop-unknown-classes`) |
| COCO pictures missing from the folder, or listed with a different path | The run would stop | Matches pictures by file name, and leaves out (and reports) pictures that do not exist |
| COCO without `width` / `height` | The run would stop | Reads the sizes from the pictures |
| `.webp` / `.tif` / `.tiff` / `.jfif` pictures with YOLO, VOC, LabelMe or CVAT labels | Skipped by mAP mode | Saves `.png` copies. Then **all** pictures are copied into `<out>/images` so they share one folder |
| Several splits (`train`, `val`, `test`) | It is not clear which to use | Asks; `--split val` is typical |

After converting, the tool checks the result with mAP mode's own loaders
(`resolve_annotations_for_map`, `load_coco_images`, `build_model_class_to_coco_category_id_map`
in `tool/test_map.py`) before printing `VERDICT: CONVERTED`.

## test (try it on pictures)

**Required:**
- `--model`: the converted model.
- `--yaml`: a class-names file.
- `--images <folder>` or `--image <file>`: one of the two, not both.
- `--adb`: needed when running from this computer.

**Pictures:** `.jpg .jpeg .png .bmp`, directly in the folder.

**The YAML** needs a top-level `names:` key. It can be a list (`names: [person, car]`, or one
`- name` per line) or a numbered map (`0: person`) starting at 0 with no gaps. The number of
names must equal the model's number of classes.
- `prepare_map_data.py class-yaml --model <model.pt>` writes a correct file from the `.pt`
  model.
- The standard 80-class COCO yaml only fits models trained on COCO.

**Output folder:**
- one annotated copy of each picture;
- one `<name>.txt` per picture, with lines `class cx cy w h score`, normalised;
- `classes.txt`.

## Paths (all modes)

- **Windows paths:** `C:\...` and `\\wsl$\<this distro>\...` work; the wrapper converts them.
  Other network (UNC) paths do not.
- **Colons:** a `:` elsewhere in a path breaks the Docker mount.
- **Inputs:** they must exist before the run, and are mounted read-only.
- **Output paths:** the skill leaves them at the defaults under `out/`. If `--output` is given, its parent folder must exist, and a test output folder must not exist yet.
- **File ownership:** results written by `./docker/iqf` may be owned by root. If the user cannot
  delete them, they can run `sudo chown -R $USER out/` themselves.
