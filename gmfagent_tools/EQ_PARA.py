# -*- coding: utf-8 -*-
"""USGS earthquake catalog query (event parameters)."""
import json
import datetime
import requests
import pandas as pd
import numpy as np
from math import radians
from pathlib import Path

from global_land_mask import globe

try:
    from config import USGS_URL, FAULT_CSV_CHINA
except ImportError:
    USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
    FAULT_CSV_CHINA = Path(r"D:\DeskTop\系统平台\usgs_query\断层信息\china_eqfault_boundary_and_angles_drop_duplicates_dgree_from_E.csv")


def _is_land(lon: float, lat: float) -> bool:
    return globe.is_land(lat, lon)


def infer_telc_class(lon: float, lat: float, depth_km: float) -> str:
    """Infer tectonic class from location and depth: land -> Crustal; ocean depth<50 -> Interface; ocean depth>=50 -> Slab."""
    if _is_land(lon, lat):
        return "Crustal"
    return "Interface" if depth_km < 50 else "Slab"


def _haversine(lon1: float, lat1: float, lon2_lat2) -> np.ndarray:
    """Spherical distance in km from (lon1,lat1) to a set of points."""
    lon2_lat2 = pd.DataFrame(np.array(lon2_lat2).reshape(-1, 2), columns=['longitude', 'latitude'])
    lon2 = lon2_lat2.iloc[:, 0]
    lat2 = lon2_lat2.iloc[:, 1]
    lon1, lat1 = map(radians, [lon1, lat1])
    lon2 = lon2.apply(lambda x: radians(x))
    lat2 = lat2.apply(lambda x: radians(x))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = ((dlat/2).apply(np.sin))**2 + np.cos(lat1) * np.cos(lat2) * ((dlon/2).apply(np.sin))**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c * 6371


def _rake_from_mechanism(Mech: str) -> int:
    """Map Mech (R/O/U/S/N) to typical rake angle."""
    mapping = {'R': 90, 'O': 0, 'U': -90, 'S': 0, 'N': -90}
    return mapping.get(Mech, 0)


# Monitoring region key -> list of nation strings (as in USGS place last segment). None = global (no filter).
USGS_REGION_NATIONS = {
    "global": None,
    "japan": ["Japan", "Japan region", "Japan Earthquake", "Japan Sea"],
    "china": ["China", "china"],
    "indonesia": ["Indonesia", "Indonesia region"],
    "chile": ["Chile", "Chile region", "off the coast of Chile"],
    "argentina": ["Argentina", "Argentina region"],
    "turkey": ["Turkey", "Turkey region"],
    "iran": ["Iran", "Iran region"],
    "philippines": ["Philippines", "Philippines region"],
    "peru": ["Peru", "Peru region"],
    "new_zealand": ["New Zealand", "New Zealand region"],
    "mexico": ["Mexico", "Mexico region"],
}


def usgs_query(
    url: str = None,
    region: str = None,
    min_mag: float = None,
) -> list[dict]:
    """
    Fetch earthquake events from USGS.
    region: key in USGS_REGION_NATIONS ("global", "japan", "china", etc.). Default "japan".
    min_mag: minimum magnitude (default 5.0). Only used when region/min_mag are set.
    Returns list of dicts with usgs_id, mag, depth, lon, lat, place, nation,
    telc_class, Mech, jiaodu, event_time, etc.
    """
    url = url or USGS_URL
    events = []
    try:
        response = requests.get(url, stream=True, timeout=30)
        result = json.loads(response.text)
        result_df = pd.DataFrame(result['features'])

        if 'properties' not in result_df.columns:
            return events

        result_df['nation'] = [
            result_df['properties'].iloc[i]['place'].split(',')[-1][1:].strip()
            for i in range(len(result_df))
        ]
        result_df['mag'] = [result_df['properties'].iloc[i]['mag'] for i in range(len(result_df))]

        if region is not None and min_mag is not None:
            mag_ok = result_df['mag'] >= min_mag
            nations = USGS_REGION_NATIONS.get(region)
            if nations is None:
                result_df = result_df[mag_ok]
            else:
                result_df = result_df[mag_ok & result_df['nation'].isin(nations)]
        else:
            mask_jp = result_df['nation'].isin(['Japan', 'Japan region', 'Japan Earthquake']) & (result_df['mag'] >= 6.0)
            mask_cn = result_df['nation'].isin(['China', 'china']) & (result_df['mag'] >= 5.0)
            result_df = result_df[mask_jp | mask_cn]

        if len(result_df) == 0:
            return events

        fault_df = None
        if FAULT_CSV_CHINA.exists():
            fault_df = pd.read_csv(FAULT_CSV_CHINA)

        id_list = [f.get('id') for f in result['features']]
        for _, row in result_df.iterrows():
            idx = id_list.index(row['id'])
            feat = result['features'][idx]
            props = feat['properties']
            coords = feat['geometry']['coordinates']

            lon_info = coords[0]
            lat_info = coords[1]
            dep_info = coords[2]
            event_mag = props['mag']
            event_place = props['place'].split(',')[0].strip()
            event_nation = props['place'].split(',')[1][1:].strip() if ',' in props['place'] else ''
            event_time = datetime.datetime.fromtimestamp(
                props['time'] / 1000
            ).strftime('%Y-%m-%dT%H:%M:%S.970Z')

            telc_class = 'Crustal' if _is_land(lon_info, lat_info) else (
                'Interface' if dep_info < 50 else 'Slab'
            )
            Mech = 'R'
            jiaodu = 0

            if event_nation == 'China' and fault_df is not None and 'lon' in fault_df.columns:
                fault_df = fault_df.copy()
                fault_df['dist'] = _haversine(lon_info, lat_info, fault_df[['lon', 'lat']])
                jiaodu = int(fault_df.loc[fault_df['dist'].idxmin(), 'jiaodu'])

            events.append({
                'usgs_id': row['id'],
                'mag': event_mag,
                'depth': dep_info,
                'lon': lon_info,
                'lat': lat_info,
                'place': event_place,
                'nation': event_nation,
                'telc_class': telc_class,
                'Mech': Mech,
                'jiaodu': jiaodu,
                'rake': jiaodu if event_nation == 'China' else _rake_from_mechanism(Mech),
                'event_time': event_time,
            })

    except Exception as e:
        raise RuntimeError(f"USGS query failed: {e}") from e
    return events
