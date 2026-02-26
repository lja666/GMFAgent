# -*- coding: utf-8 -*-
"""Grid extraction from rasters by epicenter and magnitude. Outputs grid.csv."""
import os
import warnings
from pathlib import Path

import numpy as np
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from rasterio.transform import xy
from shapely.geometry import box, Polygon

try:
    from config import OUTPUT_BASE, RASTER_POPULATION, RASTER_VS30, RASTER_DEM, SHAPEFILE_ADMIN
except ImportError:
    OUTPUT_BASE = Path("./output")
    RASTER_POPULATION = Path(r"D:\DeskTop\应急地震人员伤亡数据和计算\landscan-global-2022-assets\landscan-global-2022.tif")
    RASTER_VS30 = Path(r"D:\DeskTop\台湾地震\global_vs30_tif\global_vs30.tif")
    RASTER_DEM = Path(r"D:\DeskTop\栅格文件自动化处理\hyd_glo_dem_30s\hyd_glo_dem_30s.tif")
    SHAPEFILE_ADMIN = Path(r"D:\Q Gis文件\ne_10m_admin_1_states_provinces\ne_10m_admin_1_states_provinces.shp")


def get_grid(
    usgs_id: str,
    earthquake_lon: float,
    earthquake_lat: float,
    mag: float,
    store_base: Path = None,
    plot: bool = False,
    raster_pop: Path = None,
    raster_vs30: Path = None,
    raster_dem: Path = None,
    shapefile_admin: Path = None,
) -> str:
    """
    Build impact-area grid from epicenter and magnitude. Writes grid.csv.
    store_path = store_base / usgs_id
    Returns: full path to grid.csv
    """
    store_base = Path(store_base or OUTPUT_BASE)
    raster_pop = Path(raster_pop or RASTER_POPULATION)
    raster_vs30 = Path(raster_vs30 or RASTER_VS30)
    raster_dem = Path(raster_dem or RASTER_DEM)
    shapefile_admin = Path(shapefile_admin or SHAPEFILE_ADMIN)

    store_path = store_base / usgs_id
    store_path.mkdir(parents=True, exist_ok=True)

    impact_range_km = 80 if mag < 6 else 150 if 6 <= mag < 7 else 200 if 7 <= mag < 8 else 300
    lat_range = impact_range_km / 111.0
    lon_range = impact_range_km / (111.0 * np.cos(np.radians(earthquake_lat)))
    lon_min = earthquake_lon - lon_range
    lon_max = earthquake_lon + lon_range * 0.5
    lat_min = earthquake_lat - lat_range
    lat_max = earthquake_lat + lat_range

    if not raster_pop.exists():
        raise FileNotFoundError(f"Population raster not found: {raster_pop}")

    with rasterio.open(raster_pop) as ds:
        geom = box(lon_min, lat_min, lon_max, lat_max)
        geo = [geom.__geo_interface__]
        out_image, out_transform = mask(ds, geo, crop=True)
        out_image = out_image[0]

        rows, cols = out_image.shape
        geoms = []
        values = []

        for row in range(rows):
            for col in range(cols):
                value = out_image[row, col]
                if ds.nodata is not None and value == ds.nodata:
                    continue
                x_min, y_min = xy(out_transform, row, col)
                x_max, y_max = xy(out_transform, row + 1, col + 1)
                geom_cell = box(x_min, y_min, x_max, y_max)
                geoms.append(geom_cell)
                values.append(value)

    gdf = gpd.GeoDataFrame({'population': values}, geometry=geoms, crs=ds.crs)
    gdf['Centroid'] = gdf['geometry'].apply(lambda x: x.centroid)
    gdf['longitude'] = gdf['Centroid'].apply(lambda x: x.x)
    gdf['latitude'] = gdf['Centroid'].apply(lambda x: x.y)
    gdf["XX"] = round(gdf["longitude"], 4) * 10000
    gdf["XX_Round"] = np.round(gdf["XX"], 0).astype(int)
    gdf["YY"] = round(gdf["latitude"], 4) * 10000
    gdf["YY_Round"] = np.round(gdf["YY"], 0).astype(int)
    gdf['locid'] = (gdf['XX_Round']).astype(str) + (gdf['YY_Round']).astype(str)
    gdf["locid"] = gdf["locid"].astype('int64')
    gdf = gdf.drop(['XX', 'XX_Round', 'YY', 'YY_Round', 'Centroid'], axis=1)

    if shapefile_admin.exists():
        boundry_shp = gpd.read_file(shapefile_admin)
        xmin, ymin, xmax, ymax = lon_min, lat_min, lon_max, lat_max
        bbox = gpd.GeoSeries([Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])], crs=gdf.crs)
        bbox_gdf = gpd.GeoDataFrame(geometry=bbox, crs=boundry_shp.crs)
        gdf_clipped = gpd.overlay(boundry_shp, bbox_gdf, how='intersection')
        gdf = gpd.sjoin(gdf, gdf_clipped[['geometry', 'adm1_code', 'name', 'name_zh', 'woe_label']],
                        how="left", predicate='intersects')
    else:
        gdf['adm1_code'] = ''
        gdf['name'] = ''
        gdf['name_zh'] = ''
        gdf['woe_label'] = ''
        gdf_clipped = None

    center_lons = gdf['longitude']
    center_lats = gdf['latitude']

    if raster_vs30.exists():
        with rasterio.open(raster_vs30) as vs30_src:
            vs30_values = [list(vs30_src.sample([(lon, lat)]))[0][0] for lon, lat in zip(center_lons, center_lats)]
            vs30_arr = np.array(vs30_values)
            vs30_arr[vs30_arr == 600] = np.nan
        gdf['vs30'] = vs30_arr
    else:
        gdf['vs30'] = np.nan

    if raster_dem.exists():
        with rasterio.open(raster_dem) as dem:
            dem_values = [list(dem.sample([(lon, lat)]))[0][0] for lon, lat in zip(center_lons, center_lats)]
            dem_arr = np.array(dem_values, dtype=float)
            dem_arr[(dem_arr > 10000) | (dem_arr < 0)] = np.nan
        gdf['dem'] = dem_arr
    else:
        gdf['dem'] = np.nan

    gdf_export = gdf.copy()
    for col in ('population', 'vs30', 'dem'):
        if col not in gdf_export.columns:
            gdf_export[col] = np.nan
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Geometry column does not contain geometry", category=UserWarning)
        gdf_export['geometry'] = gdf_export['geometry'].apply(lambda x: x.wkt if x is not None else '')
    out_cols = ['population', 'geometry', 'longitude', 'latitude', 'locid', 'adm1_code', 'name', 'name_zh', 'woe_label', 'vs30', 'dem']
    out_cols = [c for c in out_cols if c in gdf_export.columns]
    grid_path = store_path / "grid.csv"
    gdf_export[out_cols].to_csv(grid_path, index=0)

    if plot and gdf_clipped is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize
        from matplotlib.lines import Line2D
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        from matplotlib.patches import FancyArrowPatch

        def _plot_col(columns):
            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
            vals = gdf[columns].replace([np.inf, -np.inf], np.nan).dropna()
            vmin = float(np.nanmin(vals)) if len(vals) else 0.0
            vmax = float(np.nanmax(vals)) if len(vals) else 1.0
            if columns == 'population':
                vmin = max(1.0, vmin) if vmin > 0 else 1.0
                vmax = max(vmax, vmin + 1.0)
                norm = LogNorm(vmin=vmin, vmax=vmax)
            else:
                if vmax <= vmin:
                    vmax = vmin + 1.0
                norm = Normalize(vmin=vmin, vmax=vmax)
            gdf.plot(column=columns, ax=ax, legend=False, edgecolor=None, cmap="viridis_r", norm=norm)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
            cbar.set_label(columns, fontsize=10)
            gdf_clipped.plot(facecolor='none', edgecolor='black', linewidth=1, ax=ax)
            for _, row in gdf_clipped.iterrows():
                centroid = row['geometry'].centroid
                ax.text(centroid.x, centroid.y, row['name'], fontsize=8, ha='center', color='blue', weight='bold')
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            scalebar = AnchoredSizeBar(ax.transData, 50/111, '50 km', loc='lower right', pad=0.1, borderpad=0.5, sep=5, size_vertical=0.02, color='red', frameon=False)
            ax.add_artist(scalebar)
            north_arrow = FancyArrowPatch((0.95, 0.88), (0.95, 0.945), transform=ax.transAxes, arrowstyle='simple', mutation_scale=28, lw=1.8, color='black', zorder=10, clip_on=False)
            ax.add_artist(north_arrow)
            ax.text(0.95, 0.87, 'N', transform=ax.transAxes, ha='center', va='top', fontsize=11, color='black', zorder=10, clip_on=False)
            ax.scatter(earthquake_lon, earthquake_lat, c='r', marker='*', s=90, zorder=10)
            epicenter_handle = Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=12, label='Epicenter')
            ax.legend(handles=[epicenter_handle], loc='best')
            plt.savefig(store_path / f"{columns}.png", bbox_inches="tight", pad_inches=0.05)
            plt.close()

        for col in ['population', 'vs30', 'dem']:
            if col in gdf.columns:
                _plot_col(col)

    return str(grid_path)
