# RoadPulse
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

### 3. Run the notebook

```bash
jupyter notebook data_exploration.ipynb
```

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
