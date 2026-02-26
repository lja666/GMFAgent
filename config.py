# -*- coding: utf-8 -*-
"""GMFAgent config. Override via env vars."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# All base data (rasters, shapefiles, fault CSV, GMPE catalog, ML model) under one folder
BASE_DATA = PROJECT_ROOT / "base_data"

OUTPUT_BASE = Path(os.environ.get("GMFAGENT_OUTPUT", str(PROJECT_ROOT / "output")))

USGS_URL = os.environ.get("GMFAGENT_USGS_URL",
    "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson")

RASTER_POPULATION = Path(os.environ.get("GMFAGENT_RASTER_POP",
    str(BASE_DATA / "rasters" / "population.tif")))
RASTER_VS30 = Path(os.environ.get("GMFAGENT_RASTER_VS30",
    str(BASE_DATA / "rasters" / "vs30.tif")))
RASTER_DEM = Path(os.environ.get("GMFAGENT_RASTER_DEM",
    str(BASE_DATA / "rasters" / "dem.tif")))
SHAPEFILE_ADMIN = Path(os.environ.get("GMFAGENT_SHAPEFILE",
    str(BASE_DATA / "shapefiles" / "admin" / "ne_10m_admin_1_states_provinces.shp")))
FAULT_CSV_CHINA = Path(os.environ.get("GMFAGENT_FAULT_CSV",
    str(BASE_DATA / "fault" / "china_faults.csv")))

GMPE_ROOT = Path(os.environ.get("GMFAGENT_GMPE_ROOT",
    str(BASE_DATA / "gmpe_root")))

ML_XGB = Path(os.environ.get("GMFAGENT_ML_XGB",
    str(BASE_DATA / "ml_models" / "ml_xgb.pickle.dat")))
# ML_XGB 使用的特征列索引，对应 Pre[:, 0], Pre[:, 18], Pre[:, 27], Pre[:, 31]
# 映射：0=depth, 18=mag, 27=hyp_dist, 31=vs30
ML_XGB_FEATURE_INDICES = [0, 18, 27, 31]

# IMT periods: None=PGA, else SA(period); e.g. [None, 0.3, 1.0, 3.0]
IMT_SA_PERIODS = [0.3, 1.0, 3.0]

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-xxx")

# RAG / LanceDB (override via env)
LANCEDB_PATH = Path(os.environ.get("GMFAGENT_LANCEDB_PATH", str(PROJECT_ROOT)))
LANCEDB_TABLE = os.environ.get("GMFAGENT_LANCEDB_TABLE", "rag_table")
RAG_FALLBACK_TOKEN_LIMIT = int(os.environ.get("GMFAGENT_RAG_FALLBACK_TOKENS", "2000"))
RAG_CONTENT_COLUMN = os.environ.get("GMFAGENT_RAG_CONTENT_COLUMN", "content")
RAG_RERANK_WEIGHT = float(os.environ.get("GMFAGENT_RAG_RERANK_WEIGHT", "0.7"))
# Max chars of RAG context in LLM prompt (DeepSeek limit 131072 tokens; reduce if context overflow)
RAG_PROMPT_MAX_CHARS = int(os.environ.get("GMFAGENT_RAG_PROMPT_MAX_CHARS", "4000"))
