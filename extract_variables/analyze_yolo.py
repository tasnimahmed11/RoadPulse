"""
analyze_yolo.py
────────────────────────────────────────────────────────────────────────────
Runs YOLO object detection on every street-view image per road and writes
one aggregated row per road to results_yolo.csv.

Zero Ultralytics dependency — uses ONNX Runtime + OpenCV only.
No background threads, no telemetry, no network calls at runtime.

FIRST-TIME SETUP (one-time, do this before running)
────────────────────────────────────────────────────
Step 1 — Export yolo11n.pt → yolo11n.onnx  (needs ultralytics, run ONCE)

    python convert_to_onnx.py

  This produces yolo11n.onnx in the current directory.
  After that, ultralytics is never needed again.

Step 2 — Verify the packages are installed:

    pip install onnxruntime opencv-python-headless --break-system-packages

  Both are likely already installed on the HPC.

USAGE
─────
  # Test run — first 5 roads only
  python analyze_yolo.py --test

  # Full run (resumes automatically if interrupted)
  python analyze_yolo.py

  # Custom options
  python analyze_yolo.py --model ./yolo11n.onnx \
                          --image-dir /scratch/ms13757/saferoads/streetview_images \
                          --conf 0.3 \
                          --limit 200

OUTPUTS  →  results_yolo.csv
─────────────────────────────
  road_id                     unique road identifier
  images_found                how many of the 4 headings existed on disk
  total_vehicles              cars + trucks + buses + motorcycles + bicycles
  total_pedestrians           person class count
  total_cars / trucks / buses / motorcycles / bicycles
  max_vehicles_single_image   peak count across the 4 headings
  avg_vehicles_per_image      mean across headings
  per_heading_json            full per-heading breakdown (JSON string)
  processed_at                timestamp

RESUME BEHAVIOUR
────────────────
  Each road id is appended to results_yolo.csv immediately after processing.
  On re-run the script reads the CSV first and skips already-completed ids.
  Safe to Ctrl+C and restart at any time.
"""

import os
import json
import time
import argparse
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort

from road_analysis_config import (
    IMAGE_DIR, YOLO_RESULTS, DOWNLOAD_TRACKER,
    get_image_paths, get_all_road_ids,
    load_downloaded_ids, load_processed_ids, append_csv_row,
)

# ── COCO class list (80 classes, indices 0-79) ─────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

VEHICLE_CLASSES    = {"car", "truck", "bus", "motorcycle", "bicycle"}
PEDESTRIAN_CLASSES = {"person"}

FIELDNAMES = [
    "road_id",
    "images_found",
    "total_vehicles",
    "total_pedestrians",
    "total_cars",
    "total_trucks",
    "total_buses",
    "total_motorcycles",
    "total_bicycles",
    "max_vehicles_single_image",
    "avg_vehicles_per_image",
    "per_heading_json",
    "processed_at",
]

# ── YOLO input size (all yolo11 variants use 640×640) ─────────────────────
INPUT_SIZE = 640


# ── Model loader ───────────────────────────────────────────────────────────

def load_model(onnx_path: str) -> ort.InferenceSession:
    """Load ONNX model with ONNX Runtime. Uses GPU if available, else CPU."""
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    print(f"  ONNX Runtime providers: {providers}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(onnx_path, sess_options=sess_options, providers=providers)


# ── Pre/post processing ────────────────────────────────────────────────────

def preprocess(img_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple]:
    """
    Letterbox-resize to INPUT_SIZE × INPUT_SIZE, normalise to [0,1], NCHW.
    Returns (blob, scale, (pad_w, pad_h)) for coordinate de-scaling.
    """
    h, w = img_bgr.shape[:2]
    scale = min(INPUT_SIZE / w, INPUT_SIZE / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = (INPUT_SIZE - nw) // 2
    pad_h = (INPUT_SIZE - nh) // 2

    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized

    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, /255
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis]        # HWC → NCHW
    return blob, scale, (pad_w, pad_h)


def postprocess(
    output: np.ndarray,
    scale: float,
    pad: tuple,
    conf_thresh: float,
    orig_shape: tuple,
) -> list[dict]:
    """
    Decode YOLO11 output tensor → list of {label, conf, box}.

    YOLO11 ONNX output shape: [1, 84, num_anchors]
      rows 0-3   : cx, cy, w, h  (in INPUT_SIZE space)
      rows 4-83  : class scores
    """
    pad_w, pad_h = pad
    orig_h, orig_w = orig_shape[:2]

    preds = output[0]                       # [84, num_anchors]
    preds = preds.T                         # [num_anchors, 84]

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]             # [num_anchors, 80]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(class_ids)), class_ids]

    mask = confidences >= conf_thresh
    boxes_xywh  = boxes_xywh[mask]
    class_ids   = class_ids[mask]
    confidences = confidences[mask]

    detections = []
    for (cx, cy, bw, bh), cls_id, conf in zip(boxes_xywh, class_ids, confidences):
        # De-letterbox: remove padding, undo scale
        cx = (cx - pad_w) / scale
        cy = (cy - pad_h) / scale
        bw = bw / scale
        bh = bh / scale

        x1 = max(0, int(cx - bw / 2))
        y1 = max(0, int(cy - bh / 2))
        x2 = min(orig_w, int(cx + bw / 2))
        y2 = min(orig_h, int(cy + bh / 2))

        label = COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else "unknown"
        detections.append({"label": label, "conf": float(conf), "box": [x1, y1, x2, y2]})

    # NMS per class
    return _nms(detections)


def _nms(detections: list, iou_thresh: float = 0.45) -> list:
    """Simple per-class NMS."""
    if not detections:
        return []
    by_class = defaultdict(list)
    for d in detections:
        by_class[d["label"]].append(d)

    kept = []
    for label, dets in by_class.items():
        dets = sorted(dets, key=lambda x: -x["conf"])
        while dets:
            best = dets.pop(0)
            kept.append(best)
            dets = [d for d in dets if _iou(best["box"], d["box"]) < iou_thresh]
    return kept


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ── Per-road detection ─────────────────────────────────────────────────────

def analyze_road(road_id: str, image_paths: list, session: ort.InferenceSession, conf: float) -> dict:
    """Run ONNX YOLO on all images for one road and return aggregated metrics."""
    input_name  = session.get_inputs()[0].name
    per_heading = {}
    total_counts = defaultdict(int)

    for img_path in image_paths:
        try:
            heading = int(
                os.path.basename(img_path)
                  .split("_heading_")[1]
                  .replace(".jpg", "")
            )
        except (IndexError, ValueError):
            heading = -1

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"\n    [WARNING] Could not read image: {img_path}")
            continue

        blob, scale, pad = preprocess(img_bgr)
        output = session.run(None, {input_name: blob})[0]
        dets   = postprocess(output, scale, pad, conf, img_bgr.shape)

        img_vehicles    = 0
        img_pedestrians = 0
        img_by_class    = defaultdict(int)

        for d in dets:
            label = d["label"]
            img_by_class[label] += 1
            if label in VEHICLE_CLASSES:
                img_vehicles += 1
            if label in PEDESTRIAN_CLASSES:
                img_pedestrians += 1

        per_heading[heading] = {
            "vehicles":    img_vehicles,
            "pedestrians": img_pedestrians,
            "by_class":    dict(img_by_class),
        }
        for cls, cnt in img_by_class.items():
            total_counts[cls] += cnt

    n              = len(image_paths)
    vcounts        = [v["vehicles"] for v in per_heading.values()]
    total_vehicles = sum(total_counts.get(c, 0) for c in VEHICLE_CLASSES)

    return {
        "road_id":                   road_id,
        "images_found":              n,
        "total_vehicles":            total_vehicles,
        "total_pedestrians":         sum(total_counts.get(c, 0) for c in PEDESTRIAN_CLASSES),
        "total_cars":                total_counts.get("car",        0),
        "total_trucks":              total_counts.get("truck",      0),
        "total_buses":               total_counts.get("bus",        0),
        "total_motorcycles":         total_counts.get("motorcycle", 0),
        "total_bicycles":            total_counts.get("bicycle",    0),
        "max_vehicles_single_image": max(vcounts) if vcounts else 0,
        "avg_vehicles_per_image":    round(total_vehicles / n, 2) if n else 0,
        "per_heading_json":          json.dumps(per_heading),
        "processed_at":              time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YOLO detection (ONNX Runtime, no Ultralytics) → results_yolo.csv"
    )
    parser.add_argument("--image-dir", default=IMAGE_DIR,
                        help=f"Image folder (default: {IMAGE_DIR})")
    parser.add_argument("--output",    default=YOLO_RESULTS,
                        help=f"Output CSV (default: {YOLO_RESULTS})")
    parser.add_argument("--model",     default="yolo11n.onnx",
                        help="Path to ONNX weights file (default: yolo11n.onnx)")
    parser.add_argument("--conf",      type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--limit",     type=int,   default=None,
                        help="Max roads to process this session")
    parser.add_argument("--test",      action="store_true",
                        help="Test mode: process only the first 5 roads")
    args = parser.parse_args()

    if args.test:
        args.limit = 5
        print("=== TEST MODE: processing first 5 roads only ===")

    # ── Check ONNX file exists ─────────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"\nERROR: ONNX model not found: {args.model}")
        print(f"\nRun the one-time converter first:")
        print(f"  python convert_to_onnx.py")
        print(f"\nOr if you want to specify a different path:")
        print(f"  python analyze_yolo.py --model /path/to/yolo11n.onnx")
        return

    # ── Load resume state ──────────────────────────────────────────────────
    already_done = load_processed_ids(args.output)
    downloaded   = load_downloaded_ids(DOWNLOAD_TRACKER)
    all_ids      = get_all_road_ids(args.image_dir)

    pending = [rid for rid in all_ids if rid not in already_done]
    if downloaded:
        pending = [rid for rid in pending if rid in downloaded]
    if args.limit:
        pending = pending[:args.limit]

    print(f"Roads on disk:     {len(all_ids)}")
    print(f"Already analyzed:  {len(already_done)}")
    print(f"To process now:    {len(pending)}")

    if not pending:
        print("Nothing to do — all roads already analyzed.")
        return

    # ── Load ONNX model ────────────────────────────────────────────────────
    print(f"\nLoading ONNX model: {args.model} …")
    t0      = time.time()
    session = load_model(args.model)
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # ── Main loop ──────────────────────────────────────────────────────────
    processed = 0
    t_start   = time.time()

    for i, road_id in enumerate(pending, 1):
        image_paths = get_image_paths(road_id, args.image_dir)
        if not image_paths:
            print(f"  [{i}/{len(pending)}] Road {road_id}: no images found, skipping")
            continue

        print(f"  [{i}/{len(pending)}] Road {road_id} — {len(image_paths)} images … ", end="", flush=True)
        t_road  = time.time()
        result  = analyze_road(road_id, image_paths, session, args.conf)
        elapsed = time.time() - t_road

        append_csv_row(args.output, FIELDNAMES, result)
        processed += 1

        avg_so_far = (time.time() - t_start) / processed
        remaining  = len(pending) - i
        eta_s      = avg_so_far * remaining
        eta_str    = f"{eta_s/60:.1f}min" if eta_s > 90 else f"{eta_s:.0f}s"

        print(
            f"✓  vehicles={result['total_vehicles']:>3}  "
            f"pedestrians={result['total_pedestrians']:>3}  "
            f"({elapsed:.1f}s)  ETA {eta_str}"
        )

    total = time.time() - t_start
    print(f"\nDone. {processed} roads analyzed → {args.output}")
    print(f"Total time: {total:.0f}s  |  Avg per road: {total/processed:.1f}s" if processed else "")


if __name__ == "__main__":
    main()