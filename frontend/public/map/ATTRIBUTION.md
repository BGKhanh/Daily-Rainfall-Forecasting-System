# Map Boundary Sources, Licences, and Attribution

## Administrative Version & Naming Convention
- **Administrative layout:** Pre-merger Vietnam **63 provinces & municipalities** (in force until **30 June 2025**) plus **2 offshore island archipelagos** (`Hoang Sa Island` and `Truong Sa Island`), totalling **65 interactive vector features**.
- **Rationale:** Matches the repository's pre-2025 geography (`src/config/constants.py::Config.province_coordinates`) and ensures Ho Chi Minh City covers only the former municipality boundaries around the single validated NASA POWER coordinate (`10.7769°N, 106.7009°E`), rather than the post-July 2025 merged boundary encompassing unvalidated Binh Duong and Ba Ria - Vung Tau.
- **Display names:** All 65 display names use clean English ASCII spellings. By project owner decision, the two island archipelagos are displayed as **`Hoang Sa Island`** (local: *Quan dao Hoang Sa*, Da Nang administration) and **`Truong Sa Island`** (local: *Quan dao Truong Sa*, Khanh Hoa administration).
- **Con Dao sub-feature normalization:** `geoBoundaries` VNM ADM1 exports Con Dao as a 64th polygon; it is merged into `Ba Ria - Vung Tau` (`VN_BARIA_VUNGTAU`) to match the canonical 63-province boundary layout, while offshore Paracel/Spratly sub-polygons embedded inside `Da Nang` and `Khanh Hoa` (`lon >= 110.5°E`) are separated into the dedicated `Hoang Sa Island` (`VN_HOANG_SA`) and `Truong Sa Island` (`VN_TRUONG_SA`) features.

---

## Verified Source Repositories & Licences (Hybrid Sources)

### 1. 63 Pre-Merger Mainland Provinces
- **Source:** `wmgeolab/geoBoundaries` — `VNM-ADM1` (`boundaryID: VNM-ADM1-63759600`, `boundaryYearRepresented: 2008`, `sourceDataUpdateDate: 2023-01-19`, build `Dec 12, 2023`)
- **Repository & File URLs:**
  - Repository: `https://github.com/wmgeolab/geoBoundaries`
  - Source GeoJSON: `https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/VNM/ADM1/geoBoundaries-VNM-ADM1_simplified.geojson`
  - Metadata API: `https://www.geoboundaries.org/api/current/gbOpen/VNM/ADM1/`
- **Retrieval Date:** `2026-11-18` (commit `9469f09`)
- **Licence (verified from source repository `LICENSE` and metadata):**
  - Boundary dataset (`boundaryLicense`): **Public Domain** (Wikimedia Commons boundary source)
  - Project code & derivatives (`LICENSE` file in `wmgeolab/geoBoundaries`): **Creative Commons Attribution 4.0 International (CC BY 4.0)**
- **Required Attribution:**
  - *geoBoundaries (Runfola, D. et al., 2020), VNM-ADM1-63759600 — CC BY 4.0 / Public Domain boundary data.*

### 2. Offshore Archipelago Features (`Hoang Sa Island` and `Truong Sa Island`)
- **Source:** `nguyenduy1133/Free-GIS-Data` (Pre-2025 Administrative Divisions)
- **Repository & File URLs:**
  - Repository: `https://github.com/nguyenduy1133/Free-GIS-Data`
  - Source GeoJSON: `Vietnam Administrative Divisions (Pre-2025) - Đơn vị hành chính Việt Nam (Trước 2025)/Provinces_included_Paracel_SpratlyIslands_combine.geojson`
- **Retrieval Date:** `2026-11-18` (branch `main`)
- **Terms Read from Source Repository (`README.md`):**
  - Note: This repository does **not** include a separate SPDX `LICENSE` file; its `README.md` (*"Copyright and Attribution"* section) states verbatim: *"This data is provided free of charge for public use. We kindly request that you cite this repository as the source when using or distributing this data in your projects, publications, or applications."*
- **Required Attribution:**
  - *Nguyen Duy Liem. (2025, July 10). Vietnam GIS Data. GitHub. https://github.com/nguyenduy1133/Free-GIS-Data*

---

## Coordinate Simplification & Runtime Policy
- Coordinates were simplified with Ramer-Douglas-Peucker (`eps = 0.002°` mainland, `0.003°` islands) and rounded to 4 decimal places (`~11 m`) for compact browser rendering without any external map tile or CDN requests at runtime.
- Map attribution is rendered visibly on the map UI and references this attribution specification.
