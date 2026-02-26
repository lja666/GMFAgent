# -*- coding: utf-8 -*-
"""GMPE selection and PGA computation."""
import importlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from math import radians

try:
    from config import OUTPUT_BASE, GMPE_ROOT, ML_XGB, ML_XGB_FEATURE_INDICES, IMT_SA_PERIODS, DEEPSEEK_API_KEY, RAG_PROMPT_MAX_CHARS
except ImportError:
    OUTPUT_BASE = Path("./output")
    GMPE_ROOT = Path(__file__).resolve().parent.parent.parent / "gmpe_root1"
    ML_XGB = Path("")
    ML_XGB_FEATURE_INDICES = [0, 18, 27, 31]
    IMT_SA_PERIODS = [0.3, 1.0, 3.0]
    DEEPSEEK_API_KEY = ""
    RAG_PROMPT_MAX_CHARS = 4000

ML_MODEL_NAMES = ["ML_XGB"]


def _haversine(lon1: float, lat1: float, lon2_lat2) -> np.ndarray:
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


def gmpe_compute_with_model(
    grid_csv_path: str,
    mag: float,
    depth: float,
    hypocenter_lon: float,
    hypocenter_lat: float,
    model_type: str = "openquake",
    key1: str = None,
    key2: str = None,
    ml_model_name: str = None,
    usgs_id: str = None,
    pga_min_filter: float = 10,
) -> tuple[str, pd.DataFrame]:
    """
    Compute PGA at grid points using given GMPE.
    Args: grid_csv_path, mag, depth, hypocenter_lon/lat, model_type,
    key1/key2 for OpenQuake, ml_model_name for ML, pga_min_filter (default 10 gal).
    Returns: (output_csv_path, grid_with_pga_df)
    """
    mag = float(mag) if not isinstance(mag, (list, np.ndarray)) else float(mag[0])
    depth = float(depth) if not isinstance(depth, (list, np.ndarray)) else float(depth[0])
    hypocenter_lon = float(hypocenter_lon) if not isinstance(hypocenter_lon, (list, np.ndarray)) else float(hypocenter_lon[0])
    hypocenter_lat = float(hypocenter_lat) if not isinstance(hypocenter_lat, (list, np.ndarray)) else float(hypocenter_lat[0])

    grid_df = pd.read_csv(grid_csv_path)
    grid_df['longitude'] = pd.to_numeric(grid_df['longitude'], errors='coerce').astype(float)
    grid_df['latitude'] = pd.to_numeric(grid_df['latitude'], errors='coerce').astype(float)
    grid_df = grid_df.dropna(subset=['longitude', 'latitude'])
    for col in ('population', 'vs30', 'dem'):
        if col not in grid_df.columns:
            grid_df[col] = np.nan
    grid_df['hyp_dist'] = np.sqrt(
        _haversine(hypocenter_lon, hypocenter_lat, grid_df[['longitude', 'latitude']].values)**2 + depth**2
    )
    grid_df['mag'] = mag
    grid_df['vs30'] = grid_df['vs30'].fillna(600)

    if model_type == "ml" and ml_model_name:
        pga_values = _compute_ml_gmpe(grid_df, mag, depth, ml_model_name)
        for col, arr in {"PGA": pga_values}.items():
            grid_df[col] = arr
    else:
        imt_dict, _ = _compute_openquake_gmpe(
            grid_df, key1 or "", key2 or "",
            hypocenter_lon=hypocenter_lon, hypocenter_lat=hypocenter_lat, depth=depth
        )
        for col, arr in imt_dict.items():
            grid_df[col] = arr

    grid_df = grid_df[grid_df["PGA"] > pga_min_filter]

    out_dir = Path(grid_csv_path).parent
    out_path = out_dir / "grid_pga.csv"
    grid_df.to_csv(out_path, index=0)

    try:
        from gmfagent_tools.MP_DISP import save_pga_png
        save_pga_png(str(out_path), str(out_dir), hypocenter_lon, hypocenter_lat)
    except Exception as e:
        import warnings
        warnings.warn(f"save_pga_png failed: {e}")

    return str(out_path), grid_df


def _compute_openquake_gmpe(
    grid_df: pd.DataFrame,
    key1: str,
    key2: str,
    hypocenter_lon: float = None,
    hypocenter_lat: float = None,
    depth: float = None,
) -> tuple:
    from openquake.hazardlib.imt import PGA, SA

    key1 = key1[0] if isinstance(key1, (list, tuple)) else str(key1)
    key2 = key2[0] if isinstance(key2, (list, tuple)) else str(key2)
    depth = float(depth) if depth is not None else 0.0
    hypocenter_lon = float(hypocenter_lon) if hypocenter_lon is not None else 0.0
    hypocenter_lat = float(hypocenter_lat) if hypocenter_lat is not None else 0.0

    module = importlib.import_module(key1)
    model_class = getattr(module, key2)
    gmpe_model = model_class()

    req_sites = list(gmpe_model.REQUIRES_SITES_PARAMETERS) if hasattr(gmpe_model, 'REQUIRES_SITES_PARAMETERS') else []
    req_dist = list(gmpe_model.REQUIRES_DISTANCES) if hasattr(gmpe_model, 'REQUIRES_DISTANCES') else []
    req_rupture = list(gmpe_model.REQUIRES_RUPTURE_PARAMETERS) if hasattr(gmpe_model, 'REQUIRES_RUPTURE_PARAMETERS') else []

    def _has(name, lst):
        n = str(name).lower().replace('_', '')
        return any(str(x).lower().replace('_', '') == n for x in lst)

    dtype = [('mag', float), ('vs30', float)]
    hyp = np.asarray(grid_df['hyp_dist'].values, dtype=np.float64)
    site_lon = np.asarray(grid_df['longitude'].values, dtype=np.float64)
    site_lat = np.asarray(grid_df['latitude'].values, dtype=np.float64)

    if _has('rhypo', req_dist):
        dtype.append(('rhypo', float))
    if _has('rrup', req_dist):
        dtype.append(('rrup', float))
    if _has('rjb', req_dist):
        dtype.append(('rjb', float))
    if _has('ztor', req_rupture):
        dtype.append(('ztor', float))
    if _has('hypo_depth', req_rupture) or _has('hypdepth', req_rupture):
        dtype.append(('hypo_depth', float))
    if _has('rake', req_rupture):
        dtype.append(('rake', float))
    if _has('rvolc', req_dist):
        dtype.append(('rvolc', float))
    if any('hypo' in str(x).lower() and 'lon' in str(x).lower() for x in req_rupture):
        k = [x for x in req_rupture if 'hypo' in str(x).lower() and 'lon' in str(x).lower()][0]
        dtype.append((k, float))
    if any('hypo' in str(x).lower() and 'lat' in str(x).lower() for x in req_rupture):
        k = [x for x in req_rupture if 'hypo' in str(x).lower() and 'lat' in str(x).lower()][0]
        dtype.append((k, float))
    if any('lon' in str(x).lower() and 'hypo' not in str(x).lower() for x in req_sites):
        k = [x for x in req_sites if 'lon' in str(x).lower() and 'hypo' not in str(x).lower()][0]
        dtype.append((k, float))
    if any('lat' in str(x).lower() and 'hypo' not in str(x).lower() for x in req_sites):
        k = [x for x in req_sites if 'lat' in str(x).lower() and 'hypo' not in str(x).lower()][0]
        dtype.append((k, float))

    ctx = np.recarray(len(grid_df), dtype=dtype)
    ctx['mag'] = np.asarray(grid_df['mag'].values, dtype=np.float64)
    ctx['vs30'] = np.asarray(grid_df['vs30'].values, dtype=np.float64)

    for dname in [d[0] for d in dtype]:
        if dname == 'rhypo':
            ctx['rhypo'] = hyp
        elif dname == 'rrup' or dname == 'rjb':
            ctx[dname] = hyp
        elif dname == 'ztor':
            ctx['ztor'] = np.full(len(grid_df), depth)
        elif dname == 'hypo_depth':
            ctx['hypo_depth'] = np.full(len(grid_df), depth)
        elif dname == 'rake':
            ctx['rake'] = np.full(len(grid_df), 90.0)
        elif dname == 'rvolc':
            ctx['rvolc'] = np.zeros(len(grid_df))
        elif 'hypo' in dname.lower() and 'lon' in dname.lower():
            ctx[dname] = np.full(len(grid_df), hypocenter_lon)
        elif 'hypo' in dname.lower() and 'lat' in dname.lower():
            ctx[dname] = np.full(len(grid_df), hypocenter_lat)
        elif 'lon' in dname.lower() and 'hypo' not in dname.lower():
            ctx[dname] = site_lon
        elif 'lat' in dname.lower() and 'hypo' not in dname.lower():
            ctx[dname] = site_lat

    from openquake.hazardlib import contexts
    imts = [PGA()] + [SA(period=p) for p in IMT_SA_PERIODS]
    ret = contexts.get_mean_stds(gmpe_model, ctx, imts)
    mean_arr = ret[0]
    result = {}
    result["PGA"] = np.exp(mean_arr[0]) * 1000
    for i, p in enumerate(IMT_SA_PERIODS):
        col = f"SA_{str(p).replace('.', '_')}"
        result[col] = np.exp(mean_arr[i + 1]) * 1000
    return result, str(gmpe_model)


def _compute_ml_gmpe(grid_df: pd.DataFrame, mag: float, depth: float, model_name: str) -> np.ndarray:
    n = (model_name or "").strip().upper().replace("-", "_")
    if n in ("ML_XGB", "GMPE_ML_JP_CRUSTL_XGB") or n.endswith("XGB"):
        return _ml_xgb(grid_df, mag, depth)
    raise ValueError(f"Unknown ML model: {model_name}. Use ML_XGB.")


def _ml_xgb(grid_df: pd.DataFrame, mag: float, depth: float) -> np.ndarray:
    import pickle
    path = Path(ML_XGB)
    if not path.exists():
        raise FileNotFoundError(f"ML_XGB model file not found: {path}")
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    n = len(grid_df)
    depth_arr = np.full(n, depth, dtype=float)
    mag_arr = np.full(n, mag, dtype=float)
    dist = grid_df['hyp_dist'].values.astype(float)
    vs30 = grid_df['vs30'].values.astype(float)
    # 映射：0=depth, 18=mag, 27=hyp_dist, 31=vs30；部分模型训练时含第5列如 rake
    idx_map = {0: depth_arr, 18: mag_arr, 27: dist, 31: vs30}
    cols = [idx_map[i] for i in ML_XGB_FEATURE_INDICES if i in idx_map]
    preX = np.column_stack(cols)
    n_need = getattr(loaded, "n_features_in_", None)
    if n_need is None and hasattr(loaded, "get_booster"):
        try:
            n_need = loaded.get_booster().num_feature()
        except Exception:
            n_need = preX.shape[1]
    if n_need is not None and preX.shape[1] < n_need:
        # 补足第5列，常用 rake=90
        extra = np.full(n, 90.0, dtype=float)
        preX = np.column_stack([preX, extra])
    pred = np.asarray(loaded.predict(preX))
    if pred.ndim == 2:
        pred = pred[:, 0]
    return np.exp(pred.ravel())


def gmpe_compute_weighted_ensemble(
    grid_csv_path: str,
    mag: float,
    depth: float,
    hypocenter_lon: float,
    hypocenter_lat: float,
    model_weights: list[tuple[str, str, float]],
    usgs_id: str = None,
    pga_min_filter: float = 10,
) -> tuple[str, pd.DataFrame, dict]:
    """
    Compute PGA as weighted average of multiple GMPEs.
    model_weights: list of (key1, key2, weight).
    Returns: (output_csv_path, grid_df, weights_used_dict)
    """
    grid_df = pd.read_csv(grid_csv_path)
    grid_df['longitude'] = pd.to_numeric(grid_df['longitude'], errors='coerce').astype(float)
    grid_df['latitude'] = pd.to_numeric(grid_df['latitude'], errors='coerce').astype(float)
    grid_df = grid_df.dropna(subset=['longitude', 'latitude'])
    for col in ('population', 'vs30', 'dem'):
        if col not in grid_df.columns:
            grid_df[col] = np.nan
    grid_df['hyp_dist'] = np.sqrt(
        _haversine(hypocenter_lon, hypocenter_lat, grid_df[['longitude', 'latitude']].values)**2 + depth**2
    )
    grid_df['mag'] = mag
    grid_df['vs30'] = grid_df['vs30'].fillna(600)

    total_w = sum(w for _, _, w in model_weights)
    if total_w <= 0:
        total_w = 1.0
    weights = [(k1, k2, w / total_w) for k1, k2, w in model_weights if w > 0]

    if not weights:
        import warnings
        warnings.warn("No valid model weights, using default KuehnEtAl2020SSlab")
        weights = [("openquake.hazardlib.gsim.kuehn_2020", "KuehnEtAl2020SSlab", 1.0)]

    imt_sums = {}
    weights_used = {}
    for key1, key2, w in weights:
        try:
            if str(key1).lower() == "ml":
                pga_arr = _compute_ml_gmpe(grid_df, mag, depth, key2)
                imt_dict = {"PGA": pga_arr}
                label = f"ML:{key2}"
            else:
                imt_dict, _ = _compute_openquake_gmpe(
                    grid_df, key1, key2,
                    hypocenter_lon=hypocenter_lon, hypocenter_lat=hypocenter_lat, depth=depth
                )
                label = key2
            for col, arr in imt_dict.items():
                if col not in imt_sums:
                    imt_sums[col] = arr * w
                else:
                    imt_sums[col] = imt_sums[col] + arr * w
            weights_used[label] = w
        except Exception as e:
            import warnings
            warnings.warn(f"Model {key1}.{key2} failed: {e}, skipping")

    if "PGA" not in imt_sums:
        raise RuntimeError("All GMPEs failed")
    for col, arr in imt_sums.items():
        grid_df[col] = arr
    grid_df = grid_df[grid_df["PGA"] > pga_min_filter]

    total_used = sum(weights_used.values())
    if total_used > 0:
        weights_used = {k: v / total_used for k, v in weights_used.items()}

    out_dir = Path(grid_csv_path).parent
    out_path = out_dir / "grid_pga.csv"
    grid_df.to_csv(out_path, index=0)
    try:
        from gmfagent_tools.MP_DISP import save_pga_png
        save_pga_png(str(out_path), str(out_dir), hypocenter_lon, hypocenter_lat)
    except Exception:
        pass
    return str(out_path), grid_df, weights_used


def gmpe_select_and_compute(
    event: dict,
    grid_csv_path: str,
    agent=None,
    n_rounds: int = 1,
    use_ml: bool = False,
    ml_model_name: str = None,
    use_ai_select: bool = True,
    progress_callback=None,
    log_callback=None,
) -> tuple[str, pd.DataFrame, str]:
    """
    Select GMPE weights via AI (n_rounds) + RAG, compute weighted PGA.
    progress_callback(round_idx, total_rounds) called each round.
    Returns: (output_csv_path, grid_df, model_used_str)
    """
    if agent is None and use_ai_select:
        try:
            from agent import _create_gmpe_agent
            agent = _create_gmpe_agent()
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to create GMPE Agent: {e}", stacklevel=2)
            agent = None

    mag = event['mag']
    depth = event['depth']
    lon = event['lon']
    lat = event['lat']
    usgs_id = event.get('usgs_id', 'unknown')

    if agent is not None:
        try:
            from rag_retrieve import rag_retrieve
            telc = event.get("telc_class", "") or ""
            query = f"{telc} {event.get('nation','')} mag{depth} depth{mag} GMPE"
            rag_context = rag_retrieve(query)

            from collections import defaultdict
            all_rounds = []
            for r in range(n_rounds):
                try:
                    if progress_callback:
                        progress_callback(r + 1, n_rounds)
                    nation = event.get('nation', '') or ''
                    place = event.get('place', '') or ''
                    loc_str = ", ".join(filter(None, [nation, place])) or "unknown"
                    prompt = (
                        f"You are a seismic/earthquake engineering expert. "
                        f"Earthquake location: {loc_str}. "
                        f"Parameters: mag={mag}, depth={depth}km, telc_class={telc} (Crustal/Interface/Slab), "
                        f"mechanism={event.get('Mech','')}, hypocenter({lon},{lat}). "
                        "Return gmpe_models with key1, key2, weight (0~1, sum≈1). Use catalog and context."
                    )
                    if use_ml:
                        prompt += f" You may also assign weights to ML models: use key1='ml', key2 one of: {', '.join(ML_MODEL_NAMES)}."
                    if rag_context:
                        prompt = f"[RAG context]\n{rag_context[:RAG_PROMPT_MAX_CHARS]}\n\n[Task]\n{prompt}"
                    result = agent.run_sync(prompt)
                    out = getattr(result, 'output', None) or getattr(result, 'data', result)
                    round_weights = []
                    if hasattr(out, 'gmpe_models') and out.gmpe_models:
                        for m in out.gmpe_models:
                            w = getattr(m, 'weight', 1.0 / max(1, len(out.gmpe_models)))
                            round_weights.append((m.key1, m.key2, float(w)))
                    elif isinstance(out, list) and out:
                        for m in out:
                            w = getattr(m, 'weight', 1.0 / max(1, len(out)))
                            round_weights.append((getattr(m, 'key1', ''), getattr(m, 'key2', ''), float(w)))
                    all_rounds.append(round_weights)
                    if log_callback:
                        from datetime import datetime
                        log_callback(f"Round {r + 1}/{n_rounds} GMPE selection done — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception as er:
                    if log_callback:
                        from datetime import datetime
                        log_callback(f"Round {r + 1}/{n_rounds} failed: {er} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    import warnings
                    warnings.warn(f"LLM round {r + 1}/{n_rounds} failed: {er}", stacklevel=2)

            if all_rounds:
                avg = defaultdict(float)
                for round_weights in all_rounds:
                    round_dict = {(k1, k2): w for k1, k2, w in round_weights if k1 and k2}
                    for (k1, k2), w in round_dict.items():
                        avg[(k1, k2)] += w
                for k in avg:
                    avg[k] /= n_rounds
                total_w = sum(avg.values())
                if total_w <= 0:
                    total_w = 1.0
                final_weights = [(k1, k2, w / total_w) for (k1, k2), w in avg.items() if w > 0]
                if use_ml and ML_MODEL_NAMES and not any(str(k1).lower() == "ml" for k1, _, _ in final_weights):
                    if Path(ML_XGB).exists():
                        final_weights.append(("ml", ML_MODEL_NAMES[0], 0.25))
                    elif log_callback:
                        log_callback(f"ML model file not found, skipped: {ML_XGB}")
                total_w = sum(w for _, _, w in final_weights)
                if total_w <= 0:
                    total_w = 1.0
                final_weights = [(k1, k2, w / total_w) for k1, k2, w in final_weights]
                if final_weights:
                    out_path, grid_df, weights_used = gmpe_compute_weighted_ensemble(
                        grid_csv_path, mag, depth, lon, lat, final_weights, usgs_id=usgs_id
                    )
                    model_str = "\n".join(f"{k}: {v:.2f}" for k, v in weights_used.items())
                    return out_path, grid_df, model_str
        except Exception as e:
            import warnings
            warnings.warn(f"LLM weighted selection failed: {e}, using default", stacklevel=2)

    key1, key2 = "openquake.hazardlib.gsim.kuehn_2020", "KuehnEtAl2020SSlab"
    out_path, grid_df = gmpe_compute_with_model(
        grid_csv_path, mag, depth, lon, lat,
        model_type="openquake", key1=key1, key2=key2, usgs_id=usgs_id
    )
    return out_path, grid_df, key2
