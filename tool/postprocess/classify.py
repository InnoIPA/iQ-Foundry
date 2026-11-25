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
"""Classification (ResNet50) pre- and post-processing shared by `test` and `eval`.

The exported graph takes NHWC float32 [1, S, S, 3] in [0, 1] (the ImageNet mean / std are
inside the model) and returns the raw class scores `logits` [1, num_classes].

Everything around the model is here: the resize + center-crop that turns an image into the
input (the same one torchvision's ImageNet recipe uses), softmax, top-k, drawing the result
on the image and the label text. The runtimes only have to produce the logits.
"""

from __future__ import annotations

import os

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

# Short side is scaled to size * RESIZE_RATIO (256 for the usual 224) before the center crop.
RESIZE_RATIO = 256 / 224
TOP_K = 5


def preprocess_classify(image_bgr, size: int = 224):
    """BGR image -> NHWC float32 [1, size, size, 3], RGB in [0, 1]: short side to 256, center crop."""
    h, w = image_bgr.shape[:2]
    short = int(round(size * RESIZE_RATIO))
    scale = short / min(h, w)
    new_w, new_h = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top, left = (new_h - size) // 2, (new_w - size) // 2
    crop = resized[top : top + size, left : left + size]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray((rgb.astype(np.float32) / 255.0)[None, ...])


def softmax(logits):
    x = np.asarray(logits, dtype=np.float32).reshape(-1)
    e = np.exp(x - x.max())
    return e / e.sum()


def top_k(logits, k: int = TOP_K):
    """The k best classes as (class ids, probabilities), best first."""
    probs = softmax(logits)
    ids = np.argsort(-probs, kind="stable")[:k]
    return ids, probs[ids]


def save_classify_outputs(output_dir, image_name: str, image_bgr, logits, class_names, k: int = TOP_K) -> int:
    """Write <image> with the best class on it and <stem>.txt (top-k, "rank class_id name prob")."""
    ids, probs = top_k(logits, k)
    names = [class_names[c] if c < len(class_names) else str(c) for c in ids]
    lines = [f"{r} {int(c)} {n} {float(p):.6f}" for r, (c, n, p) in enumerate(zip(ids, names, probs), start=1)]

    out_img = image_bgr.copy()
    height, width = out_img.shape[:2]
    label = f"{names[0]} {float(probs[0]):.2f}"
    # Text scales with the image and shrinks until the whole label fits the width.
    scale = max(0.3, min(width, height) / 400)
    while scale > 0.2:
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        if text_w <= width - 8:
            break
        scale -= 0.05
    origin = (4, text_h + 4)
    thick = max(1, int(round(scale * 2)))
    cv2.putText(out_img, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(out_img, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, image_name), out_img)
    stem = os.path.splitext(image_name)[0]
    with open(os.path.join(output_dir, stem + ".txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return len(lines)
