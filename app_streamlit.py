# -*- coding: utf-8 -*-
"""GMFAgent Streamlit UI."""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

# Favicon: seismogram waveform (generate if missing)
ASSETS_DIR = PROJECT_ROOT / "assets"
ICON_PATH = ASSETS_DIR / "icon_seismogram.png"
if not ICON_PATH.exists():
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ASSETS_DIR.mkdir(exist_ok=True)
        np.random.seed(42)
        t = np.linspace(0, 4 * np.pi, 200)
        y = np.exp(-0.15 * t) * np.sin(2 * t) + 0.3 * np.sin(5 * t) + 0.05 * np.random.randn(len(t))
        y = (y - y.min()) / (y.max() - y.min() + 1e-8) * 0.9 + 0.05
        fig, ax = plt.subplots(figsize=(1, 1), dpi=64, facecolor="#0e1117")
        ax.set_facecolor("#0e1117")
        ax.plot(t, y, color="#00d4aa", linewidth=2)
        ax.set_xlim(0, 4 * np.pi)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.subplots_adjust(0, 0, 1, 1)
        fig.savefig(ICON_PATH, bbox_inches="tight", pad_inches=0, facecolor="#0e1117", edgecolor="none")
    except Exception:
        pass
    finally:
        try:
            plt.close()
        except NameError:
            pass
PAGE_ICON = str(ICON_PATH) if ICON_PATH.exists() else "〰️"

st.set_page_config(
    page_title="GMFAgent",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
    .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; padding-left: 1rem; padding-right: 1rem; max-width: 100%; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; margin-bottom: 0.5rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.4rem 1rem; font-size: 1.05rem; }
    [data-testid="stExpander"] { margin: 0.25rem 0; }
    [data-testid="stExpander"] details { padding: 0.25rem 0; }
    .stMarkdown p, .stMarkdown label, .stMarkdown span { font-size: 1.02rem !important; }
    [data-testid="stWidgetLabel"] { font-size: 1.02rem !important; }
    .stSlider label, .stNumberInput label, .stTextInput label, .stCheckbox label, .stRadio label { font-size: 1.02rem !important; }
    [data-testid="stVerticalBlock"] > div { padding: 0.2rem 0 !important; }
    [data-testid="stHorizontalBlock"] > div { padding: 0 0.25rem !important; }
    h1 { font-size: 1.75rem !important; margin: 0.3rem 0 !important; }
    h2, h3 { font-size: 1.25rem !important; margin: 0.25rem 0 !important; }
    .stCaptionContainer { font-size: 1rem !important; margin-bottom: 0.2rem !important; }
    [data-testid="stDataFrame"] { font-size: 0.98rem !important; }
    button[kind="primary"] { font-size: 1rem !important; padding: 0.35rem 0.8rem !important; }
    .stSpinner { margin: 0.2rem 0 !important; }
</style>
""", unsafe_allow_html=True)
st.title("🤖 GMFAgent")
st.caption("A Domain Knowledge-Driven Agent for Ground Motion Field Estimation")

try:
    from agent import run_full_pipeline, run_step_query, run_step_grid, run_step_gmpe, run_polling_cycle
    from config import OUTPUT_BASE, PROJECT_ROOT
    from gmfagent_tools.MP_DISP import make_interactive_map
except ImportError as e:
    st.error(f"Import failed: {e}. Run from GMFAgent dir: streamlit run app_streamlit.py")
    st.stop()


def _rel_path(path_str: str) -> str:
    """Display path as project-relative (no Chinese in project root)."""
    if not path_str:
        return path_str
    try:
        return str(Path(path_str).resolve().relative_to(Path(PROJECT_ROOT).resolve()))
    except (ValueError, TypeError):
        return path_str


# Event parameters: (key, label, default, help). place, nation, event_time are optional (can be empty).
PRESET_FIELDS = [
    ("usgs_id", "Event ID", "preset_001", "Used for output path"),
    ("mag", "Magnitude", 7.30, "Float"),
    ("depth", "Depth (km)", 41.0, "Float"),
    ("lon", "Longitude", 141.58, "Float"),
    ("lat", "Latitude", 37.71, "Float"),
    ("place", "Place (optional)", "Japan east coast", "String"),
    ("nation", "Nation (optional)", "Japan", "e.g. Japan, China"),
    ("telc_class", "Tectonic class", "Interface", "Crustal/Interface/Slab"),
    ("Mech", "Mechanism", "R", "R/O/U/S/N"),
    ("rake", "Rake", 90, "Integer"),
    ("event_time", "Event time (optional)", "2025-01-01T00:00:00.000Z", "ISO format"),
]
OPTIONAL_PRESET_KEYS = {"place", "nation", "event_time"}

LAYER_LABELS = {
    "population": "Population", "vs30": "Vs30", "dem": "Elevation",
    "pga": "PGA", "sa_0_3": "SA(0.3)", "sa_1_0": "SA(1.0)", "sa_3_0": "SA(3.0)",
}
LAYER_KEYS = list(LAYER_LABELS.keys())

# Event detection: monitoring region key -> display label (default Japan)
POLLING_REGION_OPTIONS = [
    ("japan", "Japan"),
    ("global", "Global"),
    ("china", "China"),
    ("indonesia", "Indonesia"),
    ("chile", "Chile"),
    ("turkey", "Turkey"),
    ("iran", "Iran"),
    ("philippines", "Philippines"),
    ("peru", "Peru"),
    ("new_zealand", "New Zealand"),
    ("mexico", "Mexico"),
]

# Ground Motion Intensity Parameter: filter options (label -> column name)
IMT_FILTER_OPTIONS = [
    ("PGA", "PGA"),
    ("SA(0.3)", "SA_0_3"),
    ("SA(1.0)", "SA_1_0"),
    ("SA(3.0)", "SA_3_0"),
]


def build_preset_event() -> dict:
    return {
        "usgs_id": st.session_state.get("preset_usgs_id", "preset_001"),
        "mag": float(st.session_state.get("preset_mag", 7.30)),
        "depth": float(st.session_state.get("preset_depth", 41.0)),
        "lon": float(st.session_state.get("preset_lon", 141.58)),
        "lat": float(st.session_state.get("preset_lat", 37.71)),
        "place": st.session_state.get("preset_place", "Japan east coast") or "",
        "nation": st.session_state.get("preset_nation", "Japan") or "",
        "telc_class": st.session_state.get("preset_telc_class", "Interface"),
        "Mech": st.session_state.get("preset_Mech", "R"),
        "rake": int(st.session_state.get("preset_rake", 90)),
        "event_time": st.session_state.get("preset_event_time", "2025-01-01T00:00:00.000Z") or "",
        "jiaodu": 0,
    }


def main():
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Scenario Earthquake",
        "🔄 Event Detection",
        "🔧 Phased Execution",
        "🗺️ Visualization",
    ])

    # ========== Scenario Earthquake ==========
    with tab1:
        with st.container():
            st.subheader("Custom event parameters")
            with st.expander("Event parameters", expanded=True):
                cols = st.columns(2)
                for i, (key, label, default, help_txt) in enumerate(PRESET_FIELDS):
                    with cols[i % 2]:
                        if isinstance(default, float):
                            st.number_input(label, key=f"preset_{key}", value=default, format="%.2f", help=help_txt)
                        elif isinstance(default, int) and key == "rake":
                            st.number_input(label, key=f"preset_{key}", value=default, step=1, help=help_txt)
                        else:
                            st.text_input(label, key=f"preset_{key}", value=str(default), help=help_txt)
            event = build_preset_event()

            st.subheader("Compute impact grid and peak ground motion")
            col1, col2 = st.columns(2)
            with col1:
                use_ml = st.checkbox("Use machine learning model", value=True, key="tab1_ml", help="When checked, all ML models are added as base model candidates; LLM assigns weights to GMPEs and ML models together")
                n_rounds = st.number_input("LLM decision rounds", 1, 10, 1, key="tab1_rounds", help="Average weights over N rounds (includes ML as candidates when enabled)")
            with col2:
                save_images = st.checkbox("Save local image files", value=False, key="tab1_plot", help="Save PNG images to output directory when checked")

            log_placeholder = st.empty()
            progress_placeholder = st.empty()
            if "scenario_break" not in st.session_state:
                st.session_state.scenario_break = False
            run_col, break_col = st.columns([1, 1])
            with run_col:
                run_clicked = st.button("▶ Run", type="primary", key="tab1_run")
            with break_col:
                break_clicked = st.button("⏹ Break", key="tab1_break", help="Interrupt the running task (takes effect at end of current LLM round)")
            if break_clicked:
                st.session_state.scenario_break = True
                st.info("Break requested. It will take effect at the end of the current LLM round.")
            if run_clicked:
                st.session_state.scenario_break = False
                log_lines = []
                def _log(msg):
                    log_lines.append(msg)
                    log_placeholder.code("\n".join(log_lines), language=None)
                def _progress(r, total):
                    if st.session_state.get("scenario_break"):
                        raise InterruptedError("Stopped by user")
                    progress_placeholder.info(f"Round {r}/{total} in progress...")
                with st.spinner("Running..."):
                    try:
                        res = run_full_pipeline(
                            event=event,
                            use_ai_select=True,
                            n_rounds=n_rounds,
                            use_ml=use_ml,
                            ml_model_name=None,
                            plot_grid=save_images,
                            progress_callback=_progress,
                            log_callback=_log,
                        )
                        if res.get("error"):
                            st.error(res["error"])
                        else:
                            st.success("Done")
                            if res.get("selected_event"):
                                st.write("**Processed:**", res["selected_event"].get("place", ""))
                            if res.get("grid_path"):
                                st.write("**Grid:**", _rel_path(res["grid_path"]))
                            if res.get("pga_path"):
                                st.write("**Peak ground motion:**", _rel_path(res["pga_path"]))
                            if res.get("model_used"):
                                st.write("**Final average weights:**")
                                for line in str(res["model_used"]).strip().split("\n"):
                                    if line.strip():
                                        st.markdown(f"- {line.strip()}")
                            if res.get("grid_df") is not None:
                                st.subheader("Peak ground motion preview")
                                df_disp = res["grid_df"].copy()
                                df_disp = df_disp.drop(columns=[c for c in ["name_zh", "mag"] if c in df_disp.columns], errors="ignore")
                                if "dem" in df_disp.columns:
                                    df_disp = df_disp.rename(columns={"dem": "Elevation"})
                                st.dataframe(df_disp.head(20), use_container_width=True)
                            st.session_state["last_result"] = res
                    except InterruptedError:
                        st.warning("Task interrupted by user.")
                    except Exception as e:
                        st.exception(e)
                    finally:
                        progress_placeholder.empty()

    # ========== Event Detection ==========
    with tab2:
        with st.container():
            st.subheader("Event Detection")
            st.caption("Poll earthquake catalog for new events and run pipeline automatically. Close page to stop.")
            if "polling_seen_ids" not in st.session_state:
                st.session_state.polling_seen_ids = set()
            if "polling_stop" not in st.session_state:
                st.session_state.polling_stop = False
            if "poll_log_lines" not in st.session_state:
                st.session_state.poll_log_lines = []

            poll_interval = st.number_input("Poll interval (seconds)", min_value=10, max_value=300, value=30, step=5, key="poll_interval", help="Check for new events every N seconds")
            poll_region_label = st.selectbox(
                "Monitoring region",
                [opt[1] for opt in POLLING_REGION_OPTIONS],
                index=0,
                key="poll_region",
                help="Region to monitor for earthquakes (default Japan)",
            )
            poll_region_key = next(k for k, lbl in POLLING_REGION_OPTIONS if lbl == poll_region_label)
            poll_min_mag = st.number_input(
                "Minimum magnitude",
                min_value=2.5,
                max_value=9.0,
                value=5.0,
                step=0.1,
                format="%.1f",
                key="poll_min_mag",
                help="Only trigger pipeline for events with magnitude >= this (default 5.0)",
            )
            use_ml_poll = st.checkbox("Use machine learning model", value=True, key="poll_ml", help="When checked, all ML models are added as candidates")
            n_rounds_poll = st.number_input("LLM decision rounds", 1, 10, 1, key="poll_rounds")

            start_btn = st.button("▶ Start polling", type="primary", key="poll_start")
            stop_btn = st.button("⏹ Stop", key="poll_stop")
            poll_log_placeholder = st.empty()
            if stop_btn:
                st.session_state.polling_stop = True
                st.rerun()
            if start_btn:
                st.session_state.polling_stop = False
                st.session_state.polling_active = True
                st.rerun()

            if st.session_state.get("polling_active") and not st.session_state.get("polling_stop"):
                if "poll_last_time" not in st.session_state:
                    st.session_state.poll_last_time = 0
                elapsed = time.time() - st.session_state.poll_last_time
                if elapsed >= poll_interval:
                    def _poll_log(msg):
                        st.session_state.poll_log_lines.append(msg)
                        poll_log_placeholder.code("\n".join(st.session_state.poll_log_lines), language=None)
                    from datetime import datetime
                    _poll_log(f"Checking catalog (region={poll_region_label}, min_mag>={poll_min_mag}) --- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    progress_placeholder = st.empty()
                    def _poll_progress(r, total):
                        progress_placeholder.info(f"LLM round {r}/{total}...")
                    results, st.session_state.polling_seen_ids = run_polling_cycle(
                        st.session_state.polling_seen_ids,
                        use_ai_select=True,
                        n_rounds=n_rounds_poll,
                        use_ml=use_ml_poll,
                        ml_model_name=None,
                        region=poll_region_key,
                        min_mag=float(poll_min_mag),
                        progress_callback=_poll_progress,
                        log_callback=_poll_log,
                    )
                    if progress_placeholder:
                        progress_placeholder.empty()
                    if results:
                        _poll_log(f"Processed {len(results)} new event(s).")
                        st.success(f"Processed {len(results)} new event(s)")
                        for r in results:
                            st.write("-", r.get("selected_event", {}).get("place", ""), "|", _rel_path(r.get("pga_path", "") or ""))
                        st.session_state["last_result"] = results[-1]
                    else:
                        st.info("No new events to process.")
                    st.session_state.poll_last_time = time.time()
                    poll_log_placeholder.code("\n".join(st.session_state.poll_log_lines), language=None)
                else:
                    remaining = max(1, int(poll_interval - elapsed))
                    st.info(f"Event detection active. Next poll in {remaining} s. Click **Stop** to end monitoring.")
                    if st.session_state.get("poll_log_lines"):
                        poll_log_placeholder.code("\n".join(st.session_state.poll_log_lines), language=None)
                # Refresh every 5 s when waiting so countdown updates more often; 1 s after poll
                wait_s = 5 if elapsed < poll_interval else 1
                time.sleep(wait_s)
                if st.session_state.get("polling_stop"):
                    st.session_state.polling_active = False
                st.rerun()
            elif st.session_state.get("polling_stop"):
                st.session_state.polling_active = False
                st.session_state.polling_stop = False

    # ========== Phased Execution ==========
    with tab3:
        with st.container():
            st.subheader("Phased Execution")
            step = st.radio("Step", ["1. Recent events or custom event", "2. Generate grid", "3. Compute Peak ground motion"])
            if step == "1. Recent events or custom event":
                src = st.radio("Source", ["Recent events", "Custom event parameters"], horizontal=True, key="step1_src")
                if src == "Recent events":
                    if st.button("Query recent events", key="step1_btn"):
                        with st.spinner("Querying..."):
                            events = run_step_query()
                            if events:
                                st.dataframe(pd.DataFrame(events), use_container_width=True)
                                st.session_state["events"] = events
                            else:
                                st.info("No events found")
                else:
                    with st.expander("Custom event parameters", expanded=True):
                        cols = st.columns(2)
                        for i, (key, label, default, help_txt) in enumerate(PRESET_FIELDS):
                            with cols[i % 2]:
                                if isinstance(default, float):
                                    st.number_input(label, key=f"step1_{key}", value=default, format="%.2f", help=help_txt)
                                elif isinstance(default, int) and key == "rake":
                                    st.number_input(label, key=f"step1_{key}", value=default, step=1, help=help_txt)
                                else:
                                    st.text_input(label, key=f"step1_{key}", value=str(default), help=help_txt)
                    if st.button("Use custom event", key="step1_preset_btn"):
                        ev = {
                            "usgs_id": st.session_state.get("step1_usgs_id", "preset_001"),
                            "mag": float(st.session_state.get("step1_mag", 7.30)),
                            "depth": float(st.session_state.get("step1_depth", 41.0)),
                            "lon": float(st.session_state.get("step1_lon", 141.58)),
                            "lat": float(st.session_state.get("step1_lat", 37.71)),
                            "place": st.session_state.get("step1_place", "Japan east coast") or "",
                            "nation": st.session_state.get("step1_nation", "Japan") or "",
                            "telc_class": st.session_state.get("step1_telc_class", "Interface"),
                            "Mech": st.session_state.get("step1_Mech", "R"),
                            "rake": int(st.session_state.get("step1_rake", 90)),
                            "event_time": st.session_state.get("step1_event_time", "") or "",
                            "jiaodu": 0,
                        }
                        st.session_state["events"] = [ev]
                        st.success("Custom event set")

            if step == "2. Generate grid":
                events = st.session_state.get("events", [])
                if events:
                    idx = st.selectbox("Event", range(len(events)), format_func=lambda i: f"{events[i].get('place','')} M{events[i].get('mag','')}", key="step2_sel")
                    event = events[idx]
                    if st.button("Generate grid", key="step2_btn"):
                        with st.spinner("Generating..."):
                            path = run_step_grid(event["usgs_id"], event)
                            st.success("Saved: " + _rel_path(path))
                            st.session_state["grid_path"] = path
                            st.session_state["selected_event"] = event
                            import json
                            ep = Path(path).parent / "event_info.json"
                            with open(ep, "w", encoding="utf-8") as f:
                                json.dump(event, f, ensure_ascii=False, indent=2)
                else:
                    st.warning("Run step 1 first")

            if step == "3. Compute Peak ground motion":
                event = st.session_state.get("selected_event")
                grid_path = st.session_state.get("grid_path")
                if event and grid_path:
                    use_ml_step = st.checkbox("Use machine learning model", value=True, key="step3_ml", help="When checked, all ML models are added as candidates")
                    n_rounds_step = st.number_input("LLM decision rounds", 1, 10, 1, key="step3_rounds")
                    step3_progress = st.empty()
                    step3_log = st.empty()
                    if st.button("Compute Peak ground motion", key="step3_btn"):
                        log_lines = []
                        def _step_log(msg):
                            log_lines.append(msg)
                            step3_log.code("\n".join(log_lines), language=None)
                        def _step_progress(r, total):
                            step3_progress.info(f"Round {r}/{total} in progress...")
                        with st.spinner("Computing..."):
                            pga_path, df, model = run_step_gmpe(
                                event, grid_path,
                                use_ai=True,
                                n_rounds=n_rounds_step,
                                use_ml=use_ml_step,
                                ml_model_name=None,
                                progress_callback=_step_progress,
                                log_callback=_step_log,
                            )
                            step3_progress.empty()
                            st.success("Done")
                            st.write("**Final average weights:**")
                            for line in str(model).strip().split("\n"):
                                if line.strip():
                                    st.markdown(f"- {line.strip()}")
                        df_disp = df.copy()
                        df_disp = df_disp.drop(columns=[c for c in ["name_zh", "mag"] if c in df_disp.columns], errors="ignore")
                        if "dem" in df_disp.columns:
                            df_disp = df_disp.rename(columns={"dem": "Elevation"})
                        st.dataframe(df_disp.head(30), use_container_width=True)
                        st.session_state["pga_path"] = pga_path
                        st.session_state["last_result"] = {"grid_path": grid_path, "pga_path": pga_path, "selected_event": event}
                else:
                    st.warning("Run steps 1 and 2 first")

    # ========== Visualization ==========
    with tab4:
        with st.container():
            st.subheader("Visualization")
            out = Path(OUTPUT_BASE)
            if not out.exists():
                st.info("No output directory")
            else:
                dirs = [d for d in sorted(out.iterdir()) if d.is_dir()]
                if not dirs:
                    st.info("No event outputs")
                else:
                    sel_dir = st.selectbox("Event directory", dirs, format_func=lambda x: x.name, key="map_dir")
                    grid_path = sel_dir / "grid.csv"
                    pga_path = sel_dir / "grid_pga.csv"
                    layer_options = [LAYER_LABELS[k] for k in LAYER_KEYS]
                    layer_sel = st.radio("Layer", layer_options, horizontal=True, key="map_layer")
                    layer_key = LAYER_KEYS[layer_options.index(layer_sel)]
                    st.caption("Filter: show only grid where selected parameter exceeds minimum")
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        imt_filter_label = st.selectbox(
                            "Ground Motion Intensity Parameter",
                            [opt[0] for opt in IMT_FILTER_OPTIONS],
                            index=0,
                            key="map_imt_filter",
                            help="Select which IMT to use for filtering",
                        )
                        imt_filter_col = next(c for lbl, c in IMT_FILTER_OPTIONS if lbl == imt_filter_label)
                    with fc2:
                        imt_min = st.number_input(
                            "Minimum (gal)",
                            min_value=1,
                            value=60,
                            step=10,
                            key="map_imt_min",
                            help="Minimum value for the selected parameter",
                        )
                    use_sampling = st.checkbox("Enable sampling", value=True, key="map_sampling", help="Downsample when too many points")
                    event_info = None
                    if (sel_dir / "event_info.json").exists():
                        import json
                        with open(sel_dir / "event_info.json", "r", encoding="utf-8") as f:
                            event_info = json.load(f)
                    epic_lon = event_info.get("lon") if event_info else None
                    epic_lat = event_info.get("lat") if event_info else None
                    if (sel_dir / "grid.csv").exists():
                        m, legend_info = make_interactive_map(
                            str(grid_path),
                            str(pga_path) if pga_path.exists() else None,
                            epic_lon, epic_lat,
                            layer_key,
                            max_points=3000 if use_sampling else None,
                            event_info=event_info,
                            imt_filter_col=imt_filter_col,
                            imt_min=float(imt_min),
                        )
                        if m is not None:
                            if legend_info:
                                lab = legend_info.get("label", "Value")
                                vmin_fmt = legend_info.get("vmin_fmt", "")
                                vmax_fmt = legend_info.get("vmax_fmt", "")
                                cb_html = f'''
                                <div style="margin-bottom:8px;font-size:13px;">
                                    <span style="font-weight:600;">{lab}</span>
                                    <div style="display:flex;align-items:center;gap:6px;margin:4px 0;width:fit-content;">
                                        <span style="font-size:11px;color:#555;">{vmin_fmt}</span>
                                        <div style="width:160px;height:14px;background:linear-gradient(to right,#0080ff 0%,#00ffff 25%,#00ff00 50%,#ffff00 75%,#ff0000 100%);border-radius:4px;"></div>
                                        <span style="font-size:11px;color:#555;">{vmax_fmt}</span>
                                    </div>
                                </div>
                                '''
                                st.markdown(cb_html, unsafe_allow_html=True)
                            import streamlit.components.v1 as components
                            components.html(m._repr_html_(), height=800, scrolling=False)
                        else:
                            st.warning("Install folium: pip install folium")
                    else:
                        st.warning("No grid.csv in this directory")


if __name__ == "__main__":
    main()
