# -*- coding: utf-8 -*-
"""Interactive map display (Folium)."""
from pathlib import Path
from typing import Optional

import pandas as pd
import geopandas as gpd
from shapely import wkt

try:
    import folium
    from branca.element import Element, Figure
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


def _add_colorbar_legend(m, legend_label: str, vmin_fmt: str, vmax_fmt: str):
    """Add colorbar legend at bottom-left."""
    legend_html = f'''
    <div style="position:absolute;bottom:30px;left:10px;z-index:1000;background:white;padding:10px 14px;border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.3);font-size:12px;font-family:Arial,sans-serif;">
        <b>{legend_label}</b><br>
        <div style="height:16px;width:160px;margin:6px 0;border-radius:4px;background:linear-gradient(to right,#0066ff,#00ff88,#ffff00,#ff6600,#ff3300);"></div>
        <div style="display:flex;justify-content:space-between;font-size:11px;width:160px;">
            <span>low {vmin_fmt}</span><span>high {vmax_fmt}</span>
        </div>
    </div>
    '''
    m.get_root().html.add_child(Element(legend_html))

# 超过此点数自动采样，保证地图流畅；同时会在返回信息中说明「显示 x/y 点」
MAX_POINTS_DEFAULT = 2000


def load_grid_from_csv(csv_path: str) -> Optional[gpd.GeoDataFrame]:
    """Load grid from CSV, geometry as WKT."""
    try:
        df = pd.read_csv(csv_path)
        if 'geometry' not in df.columns:
            return None
        df['geometry'] = df['geometry'].apply(lambda x: wkt.loads(x) if pd.notna(x) and x else None)
        df = df.dropna(subset=['geometry'])
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        return gdf
    except Exception:
        return None


def _add_choropleth(folium_map, gdf: gpd.GeoDataFrame, col: str, name: str, legend: str) -> None:
    if col not in gdf.columns or gdf[col].isna().all():
        return
    gdf_clean = gdf.dropna(subset=[col])
    if len(gdf_clean) == 0:
        return
    folium.GeoJson(
        gdf_clean.__geo_interface__,
        name=name,
        style_function=lambda x: {
            'fillColor': _color_by_value(x['properties'].get(col), gdf_clean[col].min(), gdf_clean[col].max()),
            'color': 'gray',
            'weight': 0.5,
            'fillOpacity': 0.6,
        },
        tooltip=folium.GeoJsonTooltip(fields=[col], aliases=[legend], localize=True),
    ).add_to(folium_map)


def _color_by_value(v, vmin, vmax, log_scale=False):
    """Simple colormap, linear or log."""
    try:
        v = float(v)
        vmin, vmax = float(vmin), float(vmax)
        if vmax <= vmin or v <= 0:
            return '#3388ff'
        if log_scale and vmin > 0:
            import math
            v, vmin, vmax = math.log10(v), math.log10(vmin), math.log10(vmax)
        t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
        t = max(0, min(1, t))
        if t < 0.25:
            r, g, b = 0, int(128 + t * 4 * 127), 255
        elif t < 0.5:
            r, g, b = 0, 255, int(255 - (t - 0.25) * 4 * 255)
        elif t < 0.75:
            r, g, b = int((t - 0.5) * 4 * 255), 255, 0
        else:
            r, g, b = 255, int(255 - (t - 0.75) * 4 * 255), 0
        return f'#{r:02x}{g:02x}{b:02x}'
    except (TypeError, ValueError):
        return '#3388ff'


# 地震动参数列名：用于筛选和显示
IMT_COLUMNS = ["PGA", "SA_0_3", "SA_1_0", "SA_3_0"]


def make_interactive_map(
    grid_csv_path: str,
    pga_csv_path: Optional[str] = None,
    epicenter_lon: Optional[float] = None,
    epicenter_lat: Optional[float] = None,
    layer: str = "population",
    max_points: Optional[int] = MAX_POINTS_DEFAULT,
    event_info: Optional[dict] = None,
    pga_min_display: float = 30,
    imt_filter_col: Optional[str] = None,
    imt_min: Optional[float] = None,
) -> tuple[Optional["folium.Map"], Optional[dict]]:
    """
    Build interactive Folium map.
    layer: population | vs30 | dem | pga | sa_0_3 | sa_1_0 | sa_3_0
    pga_min_display: deprecated, use imt_filter_col + imt_min
    imt_filter_col: column to filter by (PGA, SA_0_3, SA_1_0, SA_3_0)
    imt_min: show only grid with imt_filter_col > this (gal)
    max_points: downsample if exceeded; None to skip
    Returns: (folium_map, legend_info)
    """
    if not FOLIUM_AVAILABLE:
        return None, None

    filter_col = imt_filter_col
    filter_min = imt_min if imt_min is not None else pga_min_display

    pga_path = Path(pga_csv_path) if pga_csv_path else Path(grid_csv_path).parent / "grid_pga.csv"
    use_pga_filter = pga_path.exists()

    if use_pga_filter:
        gdf = load_grid_from_csv(str(pga_path))
        if gdf is not None and len(gdf) > 0:
            if filter_col and filter_col in gdf.columns:
                gdf = gdf[gdf[filter_col] > filter_min]
            else:
                pga_cols = [c for c in gdf.columns if c in IMT_COLUMNS]
                if not pga_cols:
                    std = {"population", "vs30", "dem", "geometry", "longitude", "latitude", "locid", "hyp_dist", "mag"}
                    pga_cols = [c for c in gdf.columns if c not in std and pd.api.types.is_numeric_dtype(gdf[c])]
                if pga_cols:
                    fc = filter_col if filter_col in pga_cols else pga_cols[0]
                    gdf = gdf[gdf[fc] > filter_min]
    else:
        gdf = load_grid_from_csv(grid_csv_path)

    if gdf is None or len(gdf) == 0:
        return None, None

    n_total = len(gdf)
    if max_points is not None and n_total > max_points:
        step = max(1, int((n_total / max_points) ** 0.5))
        gdf = gdf.iloc[::step]
    n_display = len(gdf)

    col_map = {
        "population": ("population", "Population"),
        "vs30": ("vs30", "Vs30 (m/s)"),
        "dem": ("dem", "Elevation (m)"),
        "pga": ("PGA", "PGA (gal)"),
        "sa_0_3": ("SA_0_3", "SA(0.3) (gal)"),
        "sa_1_0": ("SA_1_0", "SA(1.0) (gal)"),
        "sa_3_0": ("SA_3_0", "SA(3.0) (gal)"),
    }
    if layer in col_map:
        col, legend_label = col_map[layer]
    else:
        col, legend_label = ("population", "Value")
    if col and col not in gdf.columns:
        if layer.startswith("sa_"):
            avail = [c for c in IMT_COLUMNS if c in gdf.columns]
            col = avail[0] if avail else ("population" if "population" in gdf.columns else list(gdf.columns)[0])
            legend_label = f"{col} (gal)" if col in IMT_COLUMNS else col
        else:
            col = "population" if "population" in gdf.columns else list(gdf.columns)[0]
            legend_label = col

    center_lat = epicenter_lat if epicenter_lat is not None else gdf.geometry.centroid.y.mean()
    center_lon = epicenter_lon if epicenter_lon is not None else gdf.geometry.centroid.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles="cartodbpositron")

    if col:
        gdf_clean = gdf.dropna(subset=[col])
        if len(gdf_clean) > 0:
            vmin, vmax = gdf_clean[col].min(), gdf_clean[col].max()
            use_log = col in ["population", "pga"] or "PGA" in str(col) or "SA" in str(col)
            if use_log:
                import numpy as np
                vmin = max(vmin, 1e-6) if vmin <= 0 else vmin
                vmax = max(vmax, vmin + 1)
            folium.GeoJson(
                gdf_clean.__geo_interface__,
                name=legend_label,
                style_function=lambda x, c=col, vm=vmin, vx=vmax, lg=use_log: {
                    "fillColor": _color_by_value(x["properties"].get(c), vm, vx, lg),
                    "color": "gray",
                    "weight": 0.5,
                    "fillOpacity": 0.65,
                },
                tooltip=folium.GeoJsonTooltip(fields=[col], aliases=[legend_label], localize=True),
            ).add_to(m)

            legend_info = {
                "label": legend_label,
                "vmin_fmt": f"{vmin:.1f}" if vmin >= 1 else f"{vmin:.2e}",
                "vmax_fmt": f"{vmax:.1f}" if vmax >= 1 else f"{vmax:.2e}",
                "points_total": n_total,
                "points_displayed": n_display,
            }
        else:
            legend_info = None
    else:
        legend_info = None

    if epicenter_lon is not None and epicenter_lat is not None:
        popup_html = "Hypocenter"
        if event_info:
            popup_html = f"""<b>Hypocenter</b><br>
            Place: {event_info.get('place','')}<br>
            Mag: {event_info.get('mag','')}<br>
            Depth: {event_info.get('depth','')} km<br>
            Lon: {epicenter_lon:.4f}° Lat: {epicenter_lat:.4f}°
            """
        folium.Marker(
            [epicenter_lat, epicenter_lon],
            popup=folium.Popup(popup_html, max_width=280),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    if event_info:
        info_html = f'''
        <div style="position:fixed;top:10px;right:10px;z-index:9999;background:white;padding:10px 14px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,0.3);font-size:12px;">
            <b>Event</b><br>
            Place: {event_info.get('place','')}<br>
            Mag: {event_info.get('mag','')} Depth: {event_info.get('depth','')} km<br>
            Type: {event_info.get('telc_class','')} Mech: {event_info.get('Mech','')}<br>
            Time: {event_info.get('event_time','')}
        </div>
        '''
        m.get_root().html.add_child(folium.Element(info_html))

    folium.LayerControl().add_to(m)
    return m, legend_info if col else None


def _plot_one_imt_png(
    gdf_clean: gpd.GeoDataFrame,
    col: str,
    out_path: Path,
    epicenter_lon: float,
    epicenter_lat: float,
    gdf_clipped: Optional[gpd.GeoDataFrame],
) -> bool:
    """Draw one IMT map and save to out_path. Legend for epicenter is always 'Epicenter'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    vmin = max(gdf_clean[col].min(), 1e-6)
    vmax = max(gdf_clean[col].max(), vmin + 1)
    gdf_clean.plot(column=col, ax=ax, legend=False, edgecolor=None, cmap="viridis_r", norm=LogNorm(vmin=vmin, vmax=vmax))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r, norm=LogNorm(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label(f"{col} (gal)", fontsize=10)

    if gdf_clipped is not None:
        gdf_clipped.plot(facecolor="none", edgecolor="black", linewidth=1, ax=ax)
        for _, row in gdf_clipped.iterrows():
            centroid = row["geometry"].centroid
            ax.text(centroid.x, centroid.y, row.get("name", ""), fontsize=8, ha="center", color="blue", weight="bold")

    scalebar = AnchoredSizeBar(
        ax.transData, 50 / 111, "50 km", loc="lower right", pad=0.1, borderpad=0.5,
        sep=5, size_vertical=0.02, color="red", frameon=False,
    )
    ax.add_artist(scalebar)
    north_arrow = FancyArrowPatch(
        (0.95, 0.88), (0.95, 0.945), transform=ax.transAxes, arrowstyle="simple",
        mutation_scale=28, lw=1.8, color="black", zorder=10, clip_on=False,
    )
    ax.add_artist(north_arrow)
    ax.text(0.95, 0.87, "N", transform=ax.transAxes, ha="center", va="top", fontsize=11, color="black", zorder=10, clip_on=False)
    ax.scatter(epicenter_lon, epicenter_lat, c="red", marker="*", s=150, zorder=10)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    epicenter_handle = Line2D([0], [0], marker="*", color="w", markerfacecolor="red", markersize=12, label="Epicenter")
    ax.legend(handles=[epicenter_handle], loc="best")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    return True


def save_pga_png(
    grid_pga_csv_path: str,
    out_dir: str,
    epicenter_lon: float,
    epicenter_lat: float,
    pga_min_display: float = 30,
    shapefile_admin: Optional[Path] = None,
) -> Optional[str]:
    """
    Plot PGA and SA maps from grid_pga.csv; save to out_dir/pga.png, sa_0_3.png, sa_1_0.png, sa_3_0.png.
    Only cells with chosen IMT > pga_min_display. Includes admin borders, scalebar, north arrow.
    Returns path to pga.png (first saved), or None.
    """
    gdf = load_grid_from_csv(grid_pga_csv_path)
    if gdf is None or len(gdf) == 0:
        return None
    imt_cols = [c for c in IMT_COLUMNS if c in gdf.columns and pd.api.types.is_numeric_dtype(gdf[c])]
    if not imt_cols:
        std_cols = {"population", "vs30", "dem", "geometry", "longitude", "latitude", "locid", "hyp_dist", "mag"}
        imt_cols = [c for c in gdf.columns if c not in std_cols and pd.api.types.is_numeric_dtype(gdf[c])]
    if not imt_cols:
        return None

    gdf_clipped = None
    if shapefile_admin is None:
        try:
            from config import SHAPEFILE_ADMIN
            shapefile_admin = SHAPEFILE_ADMIN
        except ImportError:
            shapefile_admin = None
    shp_path = Path(shapefile_admin) if shapefile_admin else None
    if shp_path and shp_path.exists():
        try:
            boundry_shp = gpd.read_file(shp_path)
            xmin, ymin, xmax, ymax = gdf.total_bounds
            from shapely.geometry import Polygon
            bbox = gpd.GeoSeries(
                [Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])],
                crs=gdf.crs,
            )
            bbox_gdf = gpd.GeoDataFrame(geometry=bbox, crs=boundry_shp.crs)
            gdf_clipped = gpd.overlay(boundry_shp, bbox_gdf, how="intersection")
        except Exception:
            gdf_clipped = None

    out_path_base = Path(out_dir)
    first_saved = None
    for col in imt_cols:
        gdf_filtered = gdf[gdf[col] > pga_min_display].dropna(subset=[col])
        if len(gdf_filtered) == 0:
            continue
        fname = "pga.png" if col == "PGA" else f"{col.lower()}.png"
        out_path = out_path_base / fname
        try:
            if _plot_one_imt_png(gdf_filtered, col, out_path, epicenter_lon, epicenter_lat, gdf_clipped):
                if first_saved is None:
                    first_saved = str(out_path)
        except Exception:
            continue
    return first_saved
