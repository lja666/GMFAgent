# GMFAgent

Web app for earthquake scenario runs and ground motion field (PGA, SA) computation. Results are shown on maps. Two UIs: chat (`app_chat.py`) and tabbed (`app_streamlit.py`).

## Prerequisites

- Python 3.10 or 3.11
- Dependencies in `requirements.txt` (Streamlit, OpenQuake, GeoPandas, Rasterio, Folium, Pydantic-AI)
- Base data: rasters, admin shapefile, GMPE catalog; optionally ML model (see Base data)
- `DEEPSEEK_API_KEY` for GMPE selection and chat (set in env or `config.py`)

## Installation

From the project root (folder with `app_chat.py`, `config.py`):

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
pip install pydantic-ai-provider-deepseek
```

Put base data under `base_data/` or run `python scripts/copy_base_data.py` (edit the script for your paths). Set `DEEPSEEK_API_KEY`:

- Windows PowerShell: `$env:DEEPSEEK_API_KEY="your-key"`
- Linux/macOS: `export DEEPSEEK_API_KEY="your-key"`

## Base data

Paths default to `base_data/`. Expected layout:

| Path under `base_data/`        | Description |
|--------------------------------|-------------|
| `rasters/population.tif`       | Population raster |
| `rasters/vs30.tif`             | Vs30 raster |
| `rasters/dem.tif`              | DEM raster |
| `shapefiles/admin/`            | Admin boundaries (.shp + sidecars) |
| `fault/china_faults.csv`       | Fault CSV |
| `gmpe_root/`                   | GMPE catalog (CSV/model files) |
| `ml_models/ml_xgb.pickle.dat`  | Optional ML model |

See `base_data/README.md`. Override paths via env (see Configuration).

## Configuration

`config.py`; overrides via environment:

| Purpose           | Env variable           | Default |
|-------------------|------------------------|--------|
| Output dir        | `GMFAGENT_OUTPUT`      | `output/` |
| Population raster | `GMFAGENT_RASTER_POP`  | `base_data/rasters/population.tif` |
| Vs30 raster       | `GMFAGENT_RASTER_VS30` | `base_data/rasters/vs30.tif` |
| DEM raster        | `GMFAGENT_RASTER_DEM`  | `base_data/rasters/dem.tif` |
| Admin shapefile   | `GMFAGENT_SHAPEFILE`   | under `base_data/shapefiles/admin/` |
| GMPE catalog      | `GMFAGENT_GMPE_ROOT`   | `base_data/gmpe_root` |
| ML model          | `GMFAGENT_ML_XGB`      | `base_data/ml_models/ml_xgb.pickle.dat` |
| API key           | `DEEPSEEK_API_KEY`     | — |
| Catalog URL       | `GMFAGENT_USGS_URL`    | built-in |

## Running

From project root:

```bash
streamlit run app_chat.py
```

or

```bash
streamlit run app_streamlit.py
```

Browser opens at `http://localhost:8501`. To bind to all interfaces:

```bash
streamlit run app_chat.py --server.address 0.0.0.0 --server.port 8501
```

## Chat usage (app_chat.py)

- **Scenario**: e.g. *Run scenario Japan M7.3, lon 141.18, lat 37.71, depth 20 km* — needs mag, depth, lon, lat; region name optional. Tectonic class inferred if not given.
- **Map**: *Show map* or *Show PGA map* shows the last run. Layers: PGA, SA(0.3s/1.0s/3.0s), Population, Vs30, DEM.
- **Events**: *List completed events* lists event IDs (sorted 1, 2, … 10). *Show event5 vs30 map* or *Show map for &lt;event_id&gt;* shows that event’s map.
- **Monitoring**: *Start event detection for Japan, M6* — polls catalog and runs pipeline for new events. Stop via sidebar/message button.
- **Catalog**: Ask for recent earthquakes by region and magnitude (no pipeline run).

UI and replies are in English. Run summary includes GMPE weights per model.

## Output

Each run writes to a subfolder under the output dir (default `output/`). Folder name = event ID (e.g. `custom_event9_mag7.0_...`). Contents: `grid_pga.csv`, `event_info.json`, and PNGs (`pga.png`, `sa_0_3.png`, etc.) when produced. Run summary and reply show GMPE weights (e.g. `AbrahamsonGulerce2020SInter: 0.25`).

## Deployment

Local: run as above; ensure base data paths exist and output dir is writable. On a server: use a dedicated env, set env vars, run Streamlit under systemd/Docker; put Nginx (or similar) in front for HTTPS; do not expose the Streamlit port directly; keep API key and paths in env.

Example systemd unit:

```ini
[Unit]
Description=GMFAgent
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/GMFAgent
Environment="PATH=/path/to/GMFAgent/.venv/bin"
ExecStart=/path/to/GMFAgent/.venv/bin/streamlit run app_chat.py --server.address 0.0.0.0 --server.port 8501
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## Project layout

```
GMFAgent/
├── app_chat.py          # Chat UI
├── app_streamlit.py     # Tabbed UI
├── agent_controller.py  # Agent and tools
├── agent.py             # Pipeline (grid → GMPE → compute)
├── config.py
├── requirements.txt
├── base_data/
├── gmfagent_tools/      # LOC_OS, GM_CACU, MP_DISP, EQ_PARA, ...
├── scripts/             # copy_base_data.py, etc.
└── output/
```

## License

See repo. When using external data (catalog, LandScan, OpenQuake GMPEs), follow their terms.
