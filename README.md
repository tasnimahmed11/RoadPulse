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

#### Lens 1: how the Speed Safety Score is calculated

`SpeedSafetyScore` measures how far observed speeds exceed *biomechanically safe* thresholds, weighted by how much traffic actually uses the road.

1. **Gap features** — three raw metrics are derived per segment and clipped at zero (only exceedances count):
   - `f85LimitGap` = F85th-percentile speed − Speed Limit (fast-tail over-speeding)
   - `MedianLimitGap` = Median speed − Speed Limit (systemic, typical-driver over-speeding)
   - `SpeedSpread` = F85th-percentile speed − Median speed (speed inconsistency, a known independent crash risk factor)
2. **Safe System thresholds** — rather than trusting the posted limit, each segment is assigned a safe speed threshold looked up by `(RoadClass, LandUse)` from the World Bank R4L Framework (Turner et al. 2024, *Guide for Safe Speeds*). This flags exceedance against the biomechanically safe speed even when a driver is technically obeying an outdated or too-permissive posted limit.
3. **SpeedComponent (0–1)** — five signals are blended with R4L-derived weights: `F85ExceedsSafeThreshold` (0.25), `MedianExceedsSafeThreshold` (0.25), normalised median gap (0.20), normalised F85 gap (0.15), normalised speed spread (0.15). The two binary exceedance flags carry half the weight since they're the most directly evidence-linked signals.
4. **Volume weighting & rescaling** — `RawScore = SpeedComponent × log1p(traffic volume)`, then min-max rescaled to **0–100** within each country. This ensures a dangerous speed profile only reaches a high score if the road also carries meaningful traffic exposure.
5. **Priority tiers** — Low (0–25), Medium (25–50), High (50–75), Critical (75–100). Scores are country-relative, not comparable across India and Thailand in absolute terms.

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
