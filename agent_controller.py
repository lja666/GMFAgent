# -*- coding: utf-8 -*-
"""Unified GMFAgent controller: natural language -> tools -> backend."""
import re
from pathlib import Path
from typing import Optional

# App state: set by app_chat before agent run, read by tools
_app_state: dict = {}

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False

try:
    from config import OUTPUT_BASE, USGS_URL, DEEPSEEK_API_KEY
except ImportError:
    OUTPUT_BASE = Path("./output")
    USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
    DEEPSEEK_API_KEY = ""

# Region name -> approximate (lon, lat) for scenario when user only gives region
REGION_COORDS = {
    "japan": (140.0, 36.0), "Japan": (140.0, 36.0),
    "taiwan": (121.0, 24.0), "Taiwan": (121.0, 24.0),
    "china": (105.0, 35.0), "China": (105.0, 35.0),
    "indonesia": (118.0, -2.0), "Indonesia": (118.0, -2.0),
    "chile": (-72.0, -35.0), "Chile": (-72.0, -35.0),
    "argentina": (-64.0, -38.0), "Argentina": (-64.0, -38.0),
    "turkey": (35.0, 39.0), "Turkey": (35.0, 39.0),
    "iran": (53.0, 32.0), "Iran": (53.0, 32.0),
    "philippines": (121.0, 13.0), "Philippines": (121.0, 13.0),
    "peru": (-75.0, -10.0), "Peru": (-75.0, -10.0),
    "new zealand": (174.0, -41.0), "New Zealand": (174.0, -41.0),
    "mexico": (-102.0, 23.0), "Mexico": (-102.0, 23.0),
}

# 地区大致范围 (lon_min, lon_max, lat_min, lat_max)，用于校验经纬度是否明显不在该地区
REGION_BOUNDS = {
    "japan": (128.0, 146.0, 30.0, 46.0),
    "taiwan": (119.5, 122.2, 21.8, 25.3),
    "china": (73.0, 135.0, 18.0, 54.0),
    "indonesia": (95.0, 141.0, -11.0, 6.0),
    "chile": (-76.0, -66.0, -56.0, -17.0),
    "argentina": (-74.0, -53.0, -55.0, -21.0),
    "turkey": (26.0, 45.0, 36.0, 42.0),
    "iran": (44.0, 64.0, 25.0, 40.0),
    "philippines": (117.0, 127.0, 5.0, 20.0),
    "peru": (-81.0, -68.0, -18.0, 0.0),
    "new_zealand": (166.0, 179.0, -47.0, -34.0),
    "mexico": (-118.0, -86.0, 14.0, 33.0),
}


def _region_key_from_name(name: str) -> str:
    """从用户输入的 nation/place 得到 REGION_BOUNDS 的 key（小写、空格变下划线）。"""
    if not name:
        return ""
    key = name.strip().lower().replace(" ", "_").replace(",", "")
    if key == "newzealand":
        key = "new_zealand"
    return key


def _coords_in_region(lon: float, lat: float, region_key: str) -> bool:
    """判断 (lon, lat) 是否在 region 大致范围内。"""
    bounds = REGION_BOUNDS.get(region_key)
    if not bounds:
        return True  # 未知地区不校验
    lon_min, lon_max, lat_min, lat_max = bounds
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def _get_session():
    return _app_state.get("session", {})


def _next_custom_event_number() -> int:
    """Return next custom_event N by scanning OUTPUT_BASE for custom_event{N}_* dirs."""
    out = Path(OUTPUT_BASE)
    if not out.exists():
        return 1
    import re
    pattern = re.compile(r"^custom_event(\d+)_")
    nums = []
    for d in out.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                nums.append(int(m.group(1)))
    return max(nums, default=0) + 1


def _make_scenario_event_id(mag: float, depth: float, lon: float, lat: float, nation: str, place: str) -> str:
    """Generate detailed event id: custom_event{N}_mag{M}_{region}_dep{D}_lat{L}_lon{L}_{date}_{time}."""
    from datetime import datetime
    n = _next_custom_event_number()
    region = (nation or place or "unknown").strip().lower().replace(" ", "_").replace(",", "")[:32]
    if not region.replace("_", "").isalnum():
        region = "unknown"
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H_%M")
    return f"custom_event{n}_mag{mag:.1f}_{region}_dep{int(round(depth))}_lat{lat:.2f}_lon{lon:.2f}_{date_str}_{time_str}"


def run_scenario(
    mag: float,
    depth: float,
    lon: Optional[float] = None,
    lat: Optional[float] = None,
    nation: str = "",
    place: str = "",
    telc_class: str = "",
    use_ml: bool = True,
    save_images: bool = False,
    n_rounds: int = 1,
) -> str:
    """Run a scenario earthquake: grid + GMPE/ML selection + PGA computation.
    depth is required (km). telc_class optional: if not given, inferred from lon/lat/depth (land->Crustal; ocean depth<50->Interface; ocean depth>=50->Slab).
    Use nation/place (e.g. Japan) if lon/lat not given - approximate coords will be used.
    """
    session = _get_session()
    user_gave_lonlat = lon is not None and lat is not None
    region_name = (nation or place or "").strip()
    if lon is None or lat is None:
        key = (region_name or "Japan").strip()
        lon, lat = REGION_COORDS.get(key, REGION_COORDS.get("Japan", (140.0, 36.0)))
    # 同时给了经纬度和地区时，若坐标明显不在该地区内则提醒
    region_mismatch_warning = ""
    if user_gave_lonlat and region_name:
        bounds_key = _region_key_from_name(region_name)
        if bounds_key and not _coords_in_region(lon, lat, bounds_key):
            region_mismatch_warning = (
                f"Note: The coordinates ({lon:.2f}, {lat:.2f}) are outside the typical range for \"{region_name}\". "
                "Using your coordinates. To use the region center instead, provide only the region name.\n\n"
            )
    telc = (telc_class or "").strip().capitalize()
    telc_was_inferred = telc not in ("Crustal", "Interface", "Slab")
    if telc_was_inferred:
        from gmfagent_tools.EQ_PARA import infer_telc_class
        telc = infer_telc_class(lon, lat, depth)
    usgs_id = _make_scenario_event_id(mag, depth, lon, lat, nation or "", place or "")
    event = {
        "mag": mag, "depth": depth, "lon": lon, "lat": lat,
        "nation": nation or "", "place": place or "",
        "usgs_id": usgs_id, "telc_class": telc, "Mech": "R", "rake": 90,
        "event_time": "2025-01-01T00:00:00.000Z",
    }
    log_lines = []
    def _log(msg):
        log_lines.append(msg)
        cb = _app_state.get("log_callback")
        if cb:
            cb(msg)
    _log("Running scenario computation…")
    from agent import run_full_pipeline
    result = run_full_pipeline(
        event=event, use_ai_select=True, n_rounds=n_rounds, use_ml=use_ml,
        plot_grid=save_images, log_callback=_log,
    )
    session["last_result"] = result
    if result.get("error"):
        return f"Failed: {result['error']}"
    session["task_log_lines"] = session.get("task_log_lines") or log_lines
    pga_path = result.get("pga_path")
    _output_abs = Path(pga_path).parent.resolve() if pga_path else None
    _project_root = Path(__file__).resolve().parent
    if _output_abs:
        try:
            output_dir = str(_output_abs.relative_to(_project_root))
        except ValueError:
            output_dir = str(_output_abs)
    else:
        output_dir = ""
    out = region_mismatch_warning + "Scenario run completed.\n\n"
    if telc_was_inferred:
        out += f"**Tectonic class** inferred from location and depth: **{telc}**.\n\n"
    out += f"**Output directory**: {output_dir}\n\n"
    out += f"Place: {result.get('selected_event', {}).get('place', '')}\n"
    out += f"Grid: {result.get('grid_path', '')}\n"
    out += f"PGA: {result.get('pga_path', '')}\n"
    model_used = result.get("model_used") or ""
    if model_used.strip():
        out += "**GMPE weights (used in ensemble):**\n"
        lines = [ln.strip() for ln in model_used.strip().split("\n") if ln.strip()]
        for line in lines:
            if ":" in line:
                out += f"- {line}\n"
            else:
                out += f"- {line}: 1.00\n"
        out += "\n"
    out += "Ask to \"show map\" or \"show PGA for event_id\" to view the interactive map.\n\n"
    out += f"**PNG directory**: {output_dir}\n(pga.png, sa_0_3.png, sa_1_0.png, sa_3_0.png, etc.)"
    return out


def start_event_detection(
    region: str = "japan",
    min_mag: float = 5.0,
    poll_interval: int = 30,
    use_ml: bool = True,
    n_rounds: int = 1,
) -> str:
    """Start automatic event detection: poll earthquake catalog and run pipeline for new events.
    region: japan, global, china, indonesia, chile, argentina, turkey, iran, philippines, peru, new_zealand, mexico
    """
    import time
    session = _get_session()
    session["polling_active"] = True
    session["polling_stop"] = False
    session["poll_last_time"] = time.time()  # 避免首次 rerun 后立刻跑一轮 USGS，先让页面刷出
    session["last_wait_log_time"] = 0  # 再次启动时允许立即打印「等待下次检查」
    session["polling_params"] = {
        "region": region.lower(), "min_mag": min_mag, "poll_interval": poll_interval,
        "use_ml": use_ml, "n_rounds": n_rounds,
    }
    session["polling_seen_ids"] = session.get("polling_seen_ids", set())
    _region_label = {"japan": "Japan", "global": "Global", "china": "China", "argentina": "Argentina",
                     "indonesia": "Indonesia", "chile": "Chile", "turkey": "Turkey", "iran": "Iran",
                     "philippines": "Philippines", "peru": "Peru", "new_zealand": "New Zealand", "mexico": "Mexico", "taiwan": "Taiwan"}
    region_label = _region_label.get(region.lower(), region)
    return (
        f"Event detection started for {region_label} (M ≥ {min_mag}).\n\n"
        f"**Settings**\n"
        f"- Region: {region_label}\n"
        f"- Minimum magnitude: {min_mag}\n"
        f"- Poll interval: {poll_interval} s\n"
        f"- ML model: {'Yes' if use_ml else 'No'}\n\n"
        f"The earthquake catalog will be polled every {poll_interval} s; new events will trigger ground motion computation.\n\n"
        "Click **Stop monitoring** (below or in the sidebar) to end. Log updates below."
    )


def stop_event_detection() -> str:
    """Stop automatic event detection."""
    session = _get_session()
    session["polling_stop"] = True
    return "Event detection stop requested."


def query_usgs(region: str = "global", min_mag: float = 4.5) -> str:
    """Query recent earthquakes from the catalog. region: japan, global, china, argentina, etc."""
    from gmfagent_tools import usgs_query
    events = usgs_query(USGS_URL, region=region.lower(), min_mag=min_mag)
    if not events:
        return "No events found."
    lines = [f"Found {len(events)} event(s):"]
    for ev in events[:10]:
        lines.append(f"- {ev.get('place','')} M{ev.get('mag','')} depth {ev.get('depth','')}km ({ev.get('usgs_id','')})")
    if len(events) > 10:
        lines.append(f"... and {len(events)-10} more")
    return "\n".join(lines)


def _event_id_sort_key(name: str):
    """Sort key for event IDs: use first number in name for natural order (1, 2, ..., 10 not 1, 10, 2)."""
    m = re.search(r"(\d+(?:\.\d+)?)", name)
    return (float(m.group(1)) if m else 0, name)


def list_completed_events() -> str:
    """List all completed events (scenarios or presets) that have PGA results and can be viewed on a map.
    Returns a numbered list with event_id and brief info (place, mag, depth). User can then ask to show map for one by event_id."""
    out_path = Path(OUTPUT_BASE)
    if not out_path.exists():
        return "No output directory. Run a scenario first."
    dirs = [d for d in out_path.iterdir() if d.is_dir() and (d / "grid_pga.csv").exists()]
    dirs.sort(key=lambda d: _event_id_sort_key(d.name))
    events = []
    for d in dirs:
        event_id = d.name
        info_path = d / "event_info.json"
        if info_path.exists():
            try:
                import json
                with open(info_path, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                place = ev.get("place", ev.get("nation", "")) or ""
                mag = ev.get("mag", "")
                depth = ev.get("depth", "")
                events.append(f"- **{event_id}**: {place}, M{mag}, depth {depth} km")
            except Exception:
                events.append(f"- **{event_id}**")
        else:
            events.append(f"- **{event_id}**")
    if not events:
        return "No completed events yet. Run a scenario first (e.g. 'Run scenario Japan M7.3 depth 20')."
    return "Completed events (use event_id to show map, e.g. 'Show map for scenario_7.3_20'):\n\n" + "\n".join(events)


def get_last_result() -> str:
    """Get summary of last scenario/event result."""
    session = _get_session()
    r = session.get("last_result")
    if not r:
        return "No result yet. Run a scenario first (e.g. 'Run scenario Japan M7.3 depth 20')."
    if r.get("error"):
        return f"Last run failed: {r['error']}"
    out = f"Place: {r.get('selected_event', {}).get('place', '')}\n"
    out += f"Grid: {r.get('grid_path', '')}\n"
    out += f"PGA: {r.get('pga_path', '')}\n"
    out += f"Weights: {r.get('model_used', '')}"
    return out


def show_map(
    event_id_or_path: Optional[str] = None,
    layer: str = "pga",
    no_sampling: bool = True,
) -> str:
    """Show interactive map. event_id_or_path: from list_completed_events (e.g. scenario_7.3_20); omit for last run.
    layer: pga, sa_0_3, sa_1_0, sa_3_0, population, vs30, dem.
    no_sampling: always True (full points, no downsampling). Kept for API compatibility only."""
    session = _get_session()
    r = session.get("last_result")
    pga_path = None
    epic_lon = epic_lat = None
    if event_id_or_path:
        base = Path(OUTPUT_BASE).resolve()
        p = base / event_id_or_path / "grid_pga.csv"
        if p.exists():
            pga_path = str(p)
            ep = Path(pga_path).parent / "event_info.json"
            if ep.exists():
                import json
                with open(ep, "r", encoding="utf-8") as f:
                    ev = json.load(f)
                    epic_lon, epic_lat = ev.get("lon"), ev.get("lat")
    if not pga_path and r:
        pga_path = r.get("pga_path")
        ev = r.get("selected_event", {})
        epic_lon, epic_lat = ev.get("lon"), ev.get("lat")
    if not pga_path:
        return "No PGA data to show. Run a scenario first."
    # 写入当前地图对应的事件 ID 与基本参数，供聊天界面展示
    event_id = event_id_or_path or Path(pga_path).parent.name
    ev = {}
    info_path = Path(pga_path).parent / "event_info.json"
    if info_path.exists():
        try:
            import json
            with open(info_path, "r", encoding="utf-8") as f:
                ev = json.load(f)
        except Exception:
            pass
    if not ev and r:
        ev = r.get("selected_event", {}) or {}
    session["last_map_event_id"] = event_id
    session["last_map_event_info"] = {
        "event_id": event_id,
        "place": ev.get("place", ev.get("nation", "")) or "",
        "mag": ev.get("mag", ""),
        "depth": ev.get("depth", ""),
    }
    # 人口、高程、vs30、PGA：按 PGA>30 gal 筛选；各周期 SA：按对应 SA>50 gal 筛选
    if layer in ("sa_0_3", "sa_1_0", "sa_3_0"):
        imt_filter_col = {"sa_0_3": "SA_0_3", "sa_1_0": "SA_1_0", "sa_3_0": "SA_3_0"}[layer]
        imt_min = 50
    else:
        imt_filter_col = "PGA"
        imt_min = 30
    max_pts = None  # 不再采样，始终显示全部格点
    try:
        from gmfagent_tools.MP_DISP import make_interactive_map
        m, legend_info = make_interactive_map(
            pga_path, pga_path, epic_lon, epic_lat, layer=layer,
            max_points=max_pts, imt_filter_col=imt_filter_col, imt_min=imt_min,
        )
        if m is None:
            return "Map display failed (folium may be missing). Run: pip install folium, then refresh. Results are in the output directory as PNGs."
        session["last_map_html"] = m._repr_html_()
        session["last_map_layer"] = layer
        session["last_map_legend"] = legend_info
        _app_state["last_map"] = m
        _app_state["last_map_layer"] = layer
        return "Map ready. Displayed below."
    except Exception as e:
        return f"Map display error: {e}. Try: pip install folium. Results are in the output directory as PNGs."


def create_controller_agent():
    """Create the unified GMFAgent controller."""
    if not PYDANTIC_AI_AVAILABLE or not DEEPSEEK_API_KEY:
        return None
    model = OpenAIChatModel("deepseek-chat", provider=DeepSeekProvider(api_key=DEEPSEEK_API_KEY))
    agent = Agent(
        model,
        output_type=str,
        system_prompt="""You are GMFAgent, a ground motion field estimation assistant. Support multi-turn conversation. Always reply in English.

REQUIRED params for run_scenario: mag, depth (km), and location as longitude and latitude (lon, lat). Optional: nation/place; if provided, calculation can be more accurate. telc_class is NOT required: when user does NOT give it, infer automatically from location and depth (land->Crustal; ocean depth<50->Interface; ocean depth>=50->Slab). Do NOT ask the user for telc_class; just call run_scenario and the tool will infer it.
If user wants to run a scenario but misses ANY required param (mag, depth, or lon/lat), do NOT call run_scenario. Instead ask clearly for: magnitude (mag), depth (km), and longitude/latitude (lon, lat); optionally mention that region/country name helps accuracy.

The map appears in chat ONLY when you actually call show_map(...). If you only reply with text (e.g. \"Map displayed\") without calling the tool, the user will see no map. So whenever the user asks to display/show/view a map, you MUST call show_map(...) first, then you may add a short confirmation in English.

Memory: When the user asks to show the map or PGA without naming a specific event_id, use the last run from context — call show_map(layer=\"pga\") and omit event_id_or_path so the tool uses the last run's result. Always call show_map; do not reply with text only.

After run_scenario completes: do NOT offer a choice between map and PNG. The result message already states the output directory. When you report the completion to the user, you MUST include the exact GMPE weights from the tool result (the \"GMPE weights (used in ensemble)\" section): list each model name and its weight value (e.g. AbrahamsonGulerce2020SInter: 0.25). Do not summarize by only naming the models without the numeric weights. Only when user asks to show/display/view the map: call show_map (omit event_id_or_path for last run, or pass event_id if user named one) with layer=... to show the interactive map. Point to the saved folder only if they ask.

For event detection: \"Start Japan M6.0 earthquake monitoring\" or \"Start event detection for Japan, M6+\" -> start_event_detection(region=\"japan\", min_mag=6.0). User can say \"Stop\" -> stop_event_detection. Describe this as \"event detection\" or \"recent earthquake catalog\"; do not mention data source or API names in replies.

When user asks ONLY to list or show completed/historical events (e.g. list completed events, show historical events, list computed events): call ONLY list_completed_events() and reply with the list in English. Do NOT call show_map in this case. Only call show_map when user explicitly asks to VIEW/SHOW a map for an event (e.g. show map for scenario_7.3_20).

When user asks to view a map: If they specify an event (e.g. \"event5\", \"event 8\", \"show event5 vs30 map\", \"scenario_7.3_20\", \"preset_001\"), you MUST call show_map(event_id_or_path=\"<that event_id>\", layer=...) with the matching event_id from list_completed_events (e.g. \"event5\" -> custom_event5_..., \"event 8\" -> custom_event8_...). Only when they do NOT name any event (e.g. just \"show map\" after a run) call show_map(layer=...) with no event_id to show the last run. Layers: pga, sa_0_3, sa_1_0, sa_3_0, population, vs30, dem.

Tools: list_completed_events, run_scenario, start_event_detection, stop_event_detection, query_usgs, get_last_result, show_map. Respond briefly and clearly in English. Use line breaks (\\n) for readability.""",
        tools=[list_completed_events, run_scenario, start_event_detection, stop_event_detection, query_usgs, get_last_result, show_map],
    )
    return agent
