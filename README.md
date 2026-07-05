# RoadPulse

**🗺️ Live map:** [https://roadpulse-map.vercel.app](https://roadpulse-map.vercel.app)

**ADB: AI for Safer Roads Innovation Challenge**
### How might we use AI and mobility data to determine where speed limits are misaligned with real-world road conditions, supporting evidence-based speed management across Asia and the Pacific? 

Develop an analytical model that: 

- Assesses whether posted speed limits align with Safe System principles 
- Identifies road segments where limits are inconsistent with road function or vulnerable road user exposure
- Produces a spatial output, a map-based visualization, highlighting priority segments for review or intervention 
- Is scalable and replicable across countries in Asia and the Pacific 

---

This repository contains our team's data exploration and analysis work for the ADB AI for Safer Roads 2026 Innovation Challenge.

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set API key (for image download)

```bash
export GOOGLE_MAPS_API_KEY="your-key-here"
```

---

## Pipeline

Run these folders in order:

### 1. `download_road_images/`
Downloads 4-directional Street View images for valid road segments from a GeoJSON file.

```bash
cd download_road_images
python download_images.py path/to/ADB_Innovation_Thailand.geojson
```

- Images save to `streetview_images/`
- Progress is tracked in `download_tracker.csv` (safe to re-run)

### 2. `extract_variables/`
Extracts road features from the downloaded images using YOLO and a local VLM (Ollama).

```bash
cd extract_variables

# YOLO — vehicle/pedestrian counts
python analyze_yolo.py --test    # test on 5 roads
python analyze_yolo.py           # full run

# VLM — road condition, walkability, etc. (requires Ollama + qwen2.5vl:7b)
python analyze_vlm.py --test
python analyze_vlm.py
```

Outputs: `results_yolo.csv` and `results_vlm.csv` (merge into `variables_*.csv` for risk scoring).

### 3. `Risk_Score/`
Calculates image-based risk scores and merges with accident data.

Open and run the notebooks:
- `risk_score_cal_thailand.ipynb` — Thailand
- `risk_score_cal_maharashtra.ipynb` — Maharashtra

Outputs: `final_risk_scores_thailand.csv`, `final_risk_scores_maharashtra.csv`

### 4. Root-level notebooks
Run in order once the data above is in place:

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Initial exploration of the cleaned road-segment GeoJSONs for India and Thailand. |
| `02_lens1_speed_safety_assessment.ipynb` | **Lens 1** — scores each segment on how far observed speeds exceed Safe System thresholds, weighted by traffic volume → `SpeedSafetyScore`. |
| `03_lens2_risk_identification.ipynb` | **Lens 2** — scores each segment on infrastructure/exposure risk derived from street-level imagery (YOLO + VLM) → `RiskScore`. |
| `04_combined_score.ipynb` | Merges Lens 1 + Lens 2 into a single `CompositeScore`, merges in rule-based policy recommendations from `policy_rec/`, and builds the interactive map. |

Each of `02`–`04` saves its scored output to `data/*_safety_score.csv`, `data/*_risk_score.csv`, and `data/*_composite_score.csv` respectively (geometry stored as WKT).

### 5. `policy_rec/`
Rule-based intervention recommendation per road segment (e.g. *Provide footpath*, *Add speed camera/speed limit enforcement*, *No priority intervention needed*), keyed by `OBJECTID`. Merged into the composite score as a descriptive field — it's not blended into `CompositeScore`, but it's browsable as its own layer on the interactive map.

---

## Interactive Map

`04_combined_score.ipynb` builds a single map with 8 toggleable layers (2 countries × 4 score types: Speed Safety, Risk Identification, Composite, Policy Recommendation). Use the dropdowns in the top-right to switch country and score type; click any road segment for its full score breakdown and recommendation.

See the live map link at the top of this README. *(The map file is too large to host directly in this repo — it's deployed separately via Vercel.)*

---

## Data

The datasets used in this project are not included in this repository due to their large size. Download them from the link below and place them in a `.adb_data/` folder at the root of the repo:

**Download link:** `<PLACEHOLDER>`

Expected structure:
```
.adb_data/
├── ADB_Innovation_Maharashtra.geojson
├── ADB_Innovation_Thailand.geojson
└── AI for Safer Roads 2026 - Data User Guide v1.0.pdf
```
