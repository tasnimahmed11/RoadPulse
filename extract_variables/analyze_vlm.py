"""
analyze_vlm.py
────────────────────────────────────────────────────────────────────────────
Uses a locally running VLM via Ollama to score street-view images per road.
Default model: qwen2.5vl:7b  (good balance of speed vs accuracy)

Other recommended Ollama vision models:
  qwen2.5vl:3b       faster, lighter
  qwen2.5vl:72b      best quality, needs a large GPU
  llava:13b          alternative if Qwen not available
  minicpm-v:8b       another solid option

SETUP (one-time)
────────────────
  1. Install Ollama:  https://ollama.com/download
  2. Pull the model:  ollama pull qwen2.5vl:7b
  3. Ollama runs automatically as a local server on http://localhost:11434

USAGE
─────
  # Test run — first 5 roads only
  python analyze_vlm.py --test

  # Full run (resumes automatically if interrupted)
  python analyze_vlm.py

  # Custom options
  python analyze_vlm.py --model qwen2.5vl:3b \
                         --image-dir /data/streetview_images \
                         --limit 200

OUTPUTS  →  results_vlm.csv
────────────────────────────
  road_id                 unique road identifier
  images_sent             number of images included in the prompt
  greenery_score          0–10  presence and density of trees / vegetation
  greenery_intensity      none | sparse | moderate | dense
  footpath_score          0–10  quality and continuity of pedestrian paths
  lighting_score          0–10  visible street lighting infrastructure
  shade_score             0–10  degree of canopy / shade cover
  road_condition_score    0–10  visible surface quality / maintenance state
  overall_walkability     0–10  holistic walkability assessment
  vlm_notes               free-text observations from the model
  processed_at            timestamp

CUSTOMISING SCORING CRITERIA
─────────────────────────────
  Edit SCORING_CRITERIA below. Each entry needs a "field" and "description".
  The prompt is rebuilt automatically — no other code changes needed.

RESUME BEHAVIOUR
────────────────
  Results are written to CSV immediately after each road is processed.
  On re-run the script reads the CSV and skips already-completed road IDs.
"""

import os
import re
import json
import time
import base64
import argparse
import urllib.request
import urllib.error

from road_analysis_config import (
    IMAGE_DIR, VLM_RESULTS, DOWNLOAD_TRACKER,
    get_image_paths, get_all_road_ids,
    load_downloaded_ids, load_processed_ids, append_csv_row,
)

# ── Ollama server ──────────────────────────────────────────────────────────
OLLAMA_URL    = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5vl:7b"

# ── Scoring criteria (edit here to change what the VLM scores) ─────────────
SCORING_CRITERIA = [
    {
        "field":       "greenery_score",
        "description": "Presence and density of trees, grass, and vegetation. 0 = none, 10 = abundant",
    },
    {
        "field":       "greenery_intensity",
        "description": "Categorical level — respond with exactly one of: none | sparse | moderate | dense",
    },
    {
        "field":       "footpath_score",
        "description": "Quality and continuity of pedestrian footpaths or sidewalks. 0 = none or broken, 10 = wide and fully continuous",
    },
    {
        "field":       "lighting_score",
        "description": "Visible street lighting infrastructure. 0 = no lights visible, 10 = well-lit with many lights",
    },
    {
        "field":       "shade_score",
        "description": "Degree of shade or tree canopy over the road or path. 0 = fully exposed, 10 = well shaded",
    },
    {
        "field":       "road_condition_score",
        "description": "Visible road or path surface quality. 0 = very poor / broken / unpaved, 10 = excellent",
    },
    {
        "field":       "building_height_score",
        "description": "Estimated typical height of buildings visible in the surroundings. 0 = no buildings or single-storey only, 5 = mid-rise (4–8 floors), 10 = high-rise towers (9+ floors). Base this on all visible structures across all images.",
    },
    {
        "field":       "building_density_score",
        "description": "How closely packed and numerous the buildings are around the street. 0 = open land / isolated buildings, 5 = suburban with gaps between buildings, 10 = continuous urban block with no gaps",
    },
    {
        "field":       "population_density_score",
        "description": "Estimated residential/commercial population density of the immediate neighbourhood, inferred from building height, building density, land use (residential towers, shops, markets, villas, warehouses, farmland, etc.), and any visible human activity. 0 = very sparse (desert, farmland, industrial only), 10 = very dense urban area (high-rise residential blocks, busy streets, mixed use). Use building evidence as the primary signal — not greenery or road quality.",
    },
    {
        "field":       "land_use_type",
        "description": "Dominant land use visible — respond with exactly one of: residential_low | residential_high | commercial | mixed | industrial | open_land | unknown",
    },
    {
        "field":       "overall_walkability",
        "description": "Holistic walkability and pedestrian comfort score. 0 = hostile / unsafe, 10 = very walkable",
    },
    {
        "field":       "vlm_notes",
        "description": "One or two sentences of free-text observations about the street environment",
    },
]

FIELDNAMES = (
    ["road_id", "images_sent"]
    + [c["field"] for c in SCORING_CRITERIA]
    + ["processed_at"]
)


# ── Prompt builder ─────────────────────────────────────────────────────────

def build_prompt() -> str:
    criteria_lines = "\n".join(
        f'  "{c["field"]}": {c["description"]}'
        for c in SCORING_CRITERIA
    )
    field_keys = ", ".join(f'"{c["field"]}"' for c in SCORING_CRITERIA)

    return f"""You are analysing street-view images of a single road segment captured from up to 4 compass headings.

Score the road segment on the following criteria based on everything visible across ALL provided images:

{criteria_lines}

Rules:
- Numeric scores must be integers 0–10.
- greenery_intensity must be exactly one of: none, sparse, moderate, dense.
- vlm_notes must be a plain string (one or two sentences).
- Respond ONLY with a valid JSON object containing exactly these keys: {field_keys}
- Do not include any text, explanation, or markdown outside the JSON object."""


# ── Image encoding ─────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    """Return base64-encoded JPEG string for Ollama's images field."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def heading_label(path: str) -> str:
    try:
        h = int(os.path.basename(path).split("_heading_")[1].replace(".jpg", ""))
        return {0: "North (0°)", 90: "East (90°)",
                180: "South (180°)", 270: "West (270°)"}.get(h, f"{h}°")
    except (IndexError, ValueError):
        return os.path.basename(path)


# ── Ollama call ────────────────────────────────────────────────────────────

def call_ollama(model: str, image_paths: list, prompt: str) -> dict:
    """
    Send images + prompt to Ollama /api/chat and parse the JSON response.

    Ollama's chat endpoint accepts a 'images' list (base64 strings) alongside
    the text content in the user message.
    """
    # Build a single user message with labelled image references in text
    image_labels = "\n".join(
        f"Image {i+1} — {heading_label(p)}"
        for i, p in enumerate(image_paths)
    )
    full_text = f"{image_labels}\n\n{prompt}"

    payload = {
        "model":  model,
        "stream": False,
        "messages": [
            {
                "role":    "user",
                "content": full_text,
                "images":  [encode_image(p) for p in image_paths],
            }
        ],
        "options": {
            "temperature": 0.1,   # low temp → more consistent structured output
        },
    }

    body = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        raw_response = json.loads(resp.read().decode("utf-8"))

    raw_text = raw_response["message"]["content"].strip()

    # Strip optional markdown fences: ```json ... ```
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$",          "", raw_text.strip())

    return json.loads(raw_text)


def safe_call_ollama(model, image_paths, prompt, road_id, retries=3) -> dict:
    """Retry wrapper — returns null scores on total failure so road is still recorded."""
    for attempt in range(1, retries + 1):
        try:
            return call_ollama(model, image_paths, prompt)

        except json.JSONDecodeError as e:
            print(f"\n    [JSON parse error — attempt {attempt}/{retries}] {e}")

        except urllib.error.URLError as e:
            print(f"\n    [Ollama connection error — attempt {attempt}/{retries}] {e}")
            print("    Is Ollama running?  Start it with:  ollama serve")
            time.sleep(3)

        except Exception as e:
            print(f"\n    [Error — attempt {attempt}/{retries}] {e}")
            time.sleep(2)

    print(f"\n    [FAILED] Road {road_id} — recording nulls after {retries} attempts")
    return {c["field"]: None for c in SCORING_CRITERIA}


# ── Ollama availability check ──────────────────────────────────────────────

def check_ollama_running():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def check_model_available(model: str) -> bool:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            data = json.loads(r.read())
            names = [m["name"] for m in data.get("models", [])]
            # Match loosely: "qwen2.5vl:7b" matches "qwen2.5vl:7b"
            return any(model in n or n in model for n in names)
    except Exception:
        return False


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VLM scoring of street-view images via Ollama → results_vlm.csv"
    )
    parser.add_argument("--model",     default=DEFAULT_MODEL,
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--image-dir", default=IMAGE_DIR,
                        help=f"Image folder (default: {IMAGE_DIR})")
    parser.add_argument("--output",    default=VLM_RESULTS,
                        help=f"Output CSV (default: {VLM_RESULTS})")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Max roads to process this session")
    parser.add_argument("--delay",     type=float, default=0.0,
                        help="Optional sleep between calls in seconds (default: 0)")
    parser.add_argument("--test",      action="store_true",
                        help="Test mode: process only the first 5 roads")
    args = parser.parse_args()

    if args.test:
        args.limit = 5
        print("=== TEST MODE: processing first 5 roads only ===")

    # ── Pre-flight checks ──────────────────────────────────────────────────
    print(f"Checking Ollama is running … ", end="", flush=True)
    if not check_ollama_running():
        print("NOT FOUND")
        print("\nOllama does not appear to be running.")
        print("Start it with:   ollama serve")
        print("Then re-run this script.")
        return
    print("OK")

    print(f"Checking model '{args.model}' is available … ", end="", flush=True)
    if not check_model_available(args.model):
        print("NOT FOUND")
        print(f"\nModel '{args.model}' is not pulled yet.")
        print(f"Pull it with:   ollama pull {args.model}")
        print("Then re-run this script.")
        return
    print("OK")

    # ── Load state ─────────────────────────────────────────────────────────
    already_done = load_processed_ids(args.output)
    downloaded   = load_downloaded_ids(DOWNLOAD_TRACKER)
    all_ids      = get_all_road_ids(args.image_dir)

    pending = [rid for rid in all_ids if rid not in already_done]
    if downloaded:
        pending = [rid for rid in pending if rid in downloaded]

    if args.limit:
        pending = pending[:args.limit]

    print(f"\nRoads on disk:     {len(all_ids)}")
    print(f"Already scored:    {len(already_done)}")
    print(f"To process now:    {len(pending)}")

    if not pending:
        print("\nNothing to do — all roads already scored.")
        return

    prompt = build_prompt()

    # ── Main loop ──────────────────────────────────────────────────────────
    processed = 0
    t_start   = time.time()

    for i, road_id in enumerate(pending, 1):
        image_paths = get_image_paths(road_id, args.image_dir)
        if not image_paths:
            print(f"  [{i}/{len(pending)}] Road {road_id}: no images found, skipping")
            continue

        print(f"  [{i}/{len(pending)}] Road {road_id} — {len(image_paths)} images … ", end="", flush=True)
        t0 = time.time()

        scores = safe_call_ollama(args.model, image_paths, prompt, road_id)

        elapsed = time.time() - t0
        row = {
            "road_id":      road_id,
            "images_sent":  len(image_paths),
            "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **scores,
        }
        append_csv_row(args.output, FIELDNAMES, row)
        processed += 1

        walkability  = scores.get("overall_walkability",    "?")
        greenery     = scores.get("greenery_score",         "?")
        pop_density  = scores.get("population_density_score","?")
        land_use     = scores.get("land_use_type",          "?")
        print(f"✓  walkability={walkability}  greenery={greenery}  pop_density={pop_density}  land_use={land_use}  ({elapsed:.1f}s)")

        if args.delay:
            time.sleep(args.delay)

    total_time = time.time() - t_start
    avg = total_time / processed if processed else 0
    print(f"\nDone. {processed} roads scored → {args.output}")
    print(f"Total time: {total_time:.0f}s  |  Avg per road: {avg:.1f}s")


if __name__ == "__main__":
    main()