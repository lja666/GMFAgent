# -*- coding: utf-8 -*-
"""GMFAgent: USGS query -> grid -> GMPE selection -> PGA computation."""
from pathlib import Path
from typing import Optional

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic import BaseModel
    PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False

from gmfagent_tools import usgs_query, get_grid, gmpe_select_and_compute, gmpe_compute_with_model
from gmfagent_tools.DA_FUS import list_gmpe_files, read_gmpe_csv, get_gmpe_root

try:
    from config import OUTPUT_BASE, USGS_URL, DEEPSEEK_API_KEY, GMPE_ROOT
except ImportError:
    OUTPUT_BASE = Path("./output")
    USGS_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_day.geojson"
    DEEPSEEK_API_KEY = ""
    GMPE_ROOT = Path(__file__).resolve().parent.parent / "gmpe_root1"


def _create_gmpe_agent():
    import warnings
    if not PYDANTIC_AI_AVAILABLE:
        warnings.warn("pydantic_ai or DeepSeek provider not installed", stacklevel=2)
        return None
    if not DEEPSEEK_API_KEY:
        warnings.warn("DEEPSEEK_API_KEY not set", stacklevel=2)
        return None

    class ModelWeight(BaseModel):
        key1: str
        key2: str
        weight: float = 0.5

    class GmpeWeights(BaseModel):
        gmpe_models: list[ModelWeight]

    model = OpenAIChatModel('deepseek-chat', provider=DeepSeekProvider(api_key=DEEPSEEK_API_KEY))
    gmpe_root = str(get_gmpe_root())

    agent = Agent(
        model,
        output_type=[GmpeWeights, str],
        system_prompt=f"""You are a seismology expert. Assign weights to suitable GMPEs from the catalog given event parameters.
Catalog path: {gmpe_root}
Call list_gmpe_files and read_gmpe_csv to inspect models. Pick by telc_class (Crustal/Interface/Slab) and region.
Return gmpe_models with key1, key2, weight (0~1, weights should sum to ~1).""",
        tools=[list_gmpe_files, read_gmpe_csv],
    )
    return agent


def run_full_pipeline(
    usgs_id: Optional[str] = None,
    event: Optional[dict] = None,
    url: str = None,
    use_ai_select: bool = True,
    n_rounds: int = 1,
    use_ml: bool = False,
    ml_model_name: Optional[str] = None,
    plot_grid: bool = False,
    progress_callback=None,
    log_callback=None,
) -> dict:
    """Run full flow: query -> grid -> GMPE select -> PGA. Returns dict with events, selected_event, grid_path, pga_path, grid_df, model_used."""
    result = {"events": [], "selected_event": None, "grid_path": None, "pga_path": None, "grid_df": None, "model_used": None}

    if event is None:
        events = usgs_query(url or USGS_URL)
        result["events"] = events
        if not events:
            return result
        event = events[0]
        usgs_id = event["usgs_id"]

    usgs_id = usgs_id or event.get("usgs_id", "unknown")
    store_path = Path(OUTPUT_BASE) / usgs_id
    store_path.mkdir(parents=True, exist_ok=True)

    result["selected_event"] = event

    if log_callback:
        log_callback("Generating grid from rasters (about 1–2 min)…")
    try:
        grid_path = get_grid(
            usgs_id=usgs_id,
            earthquake_lon=event["lon"],
            earthquake_lat=event["lat"],
            mag=event["mag"],
            store_base=OUTPUT_BASE,
            plot=plot_grid,
        )
        result["grid_path"] = grid_path
        if log_callback:
            log_callback("Grid done. Selecting GMPE and computing PGA…")
    except Exception as e:
        result["error"] = f"Grid generation failed: {e}"
        return result

    agent = _create_gmpe_agent() if use_ai_select else None
    try:
        pga_path, grid_df, model_used = gmpe_select_and_compute(
            event=event,
            grid_csv_path=grid_path,
            agent=agent,
            n_rounds=n_rounds,
            use_ml=use_ml,
            ml_model_name=ml_model_name,
            progress_callback=progress_callback,
            log_callback=log_callback,
        )
        result["pga_path"] = pga_path
        result["grid_df"] = grid_df
        result["model_used"] = model_used
        if log_callback:
            log_callback("PGA computation completed.")
        import json
        event_path = store_path / "event_info.json"
        with open(event_path, "w", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        result["error"] = f"Ground motion computation failed: {e}\n\n{tb}"

    return result


def run_step_query(url: str = None) -> list[dict]:
    """Run USGS query only."""
    return usgs_query(url or USGS_URL)


def run_step_grid(usgs_id: str, event: dict, plot: bool = False) -> str:
    """Run grid generation only."""
    return get_grid(usgs_id, event["lon"], event["lat"], event["mag"], store_base=OUTPUT_BASE, plot=plot)


def run_step_gmpe(
    event: dict,
    grid_csv_path: str,
    use_ai: bool = True,
    n_rounds: int = 1,
    use_ml: bool = False,
    ml_model_name: str = None,
    progress_callback=None,
    log_callback=None,
) -> tuple[str, "pd.DataFrame", str]:
    """Run GMPE select and compute only."""
    agent = _create_gmpe_agent() if use_ai else None
    return gmpe_select_and_compute(event, grid_csv_path, agent, n_rounds, use_ml, ml_model_name, progress_callback=progress_callback, log_callback=log_callback)


def run_polling_cycle(
    seen_ids: set,
    use_ai_select: bool = True,
    n_rounds: int = 1,
    use_ml: bool = False,
    ml_model_name: Optional[str] = None,
    url: str = None,
    region: str = "japan",
    min_mag: float = 5.0,
    progress_callback=None,
    log_callback=None,
) -> tuple[list[dict], set]:
    """Poll USGS once, run full pipeline only for new events. Returns (results, updated seen_ids)."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    events = usgs_query(url or USGS_URL, region=region, min_mag=min_mag)
    new_events = [ev for ev in events if ev.get("usgs_id") and ev.get("usgs_id") not in seen_ids]
    if log_callback:
        if not events:
            log_callback(f"[{ts}] No events in catalog.")
        elif not new_events:
            log_callback(f"[{ts}] No new events ({len(events)} total, all already processed).")
        else:
            log_callback(f"[{ts}] Found {len(new_events)} new event(s), processing…")
    results = []
    for ev in events:
        eid = ev.get("usgs_id", "")
        if eid and eid not in seen_ids:
            res = run_full_pipeline(event=ev, use_ai_select=use_ai_select, n_rounds=n_rounds, use_ml=use_ml, ml_model_name=ml_model_name, progress_callback=progress_callback, log_callback=log_callback)
            results.append(res)
            seen_ids.add(eid)
    return results, seen_ids
