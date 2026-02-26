# Base Data

All paths used by GMFAgent point here by default. Put the following files/dirs in place, or run from project root:

```bash
python scripts/copy_base_data.py
```

to copy from the legacy paths (edit `scripts/copy_base_data.py` if your files are elsewhere).

| Path under base_data | Description |
|----------------------|-------------|
| `rasters/population.tif` | Population raster (e.g. LandScan global) |
| `rasters/vs30.tif` | Vs30 global raster |
| `rasters/dem.tif` | DEM raster |
| `shapefiles/admin/` | Admin boundaries shapefile (.shp + .dbf, .shx, .prj, etc.) |
| `fault/china_faults.csv` | China fault CSV (lon, lat, jiaodu, ...) |
| `gmpe_root/` | GMPE catalog directory (CSV/model files) |
| `base_data/ml_models/ml_xgb.pickle.7z` | ML_XGB pickle model (Compressed Archive)|

Override any path via environment variables (see config.py).
