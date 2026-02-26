# -*- coding: utf-8 -*-
"""GMFAgent Chat UI: natural language interface with multi-turn conversation."""
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

ASSETS_DIR = PROJECT_ROOT / "assets"
ICON_PATH = ASSETS_DIR / "icon_seismogram.png"
PAGE_ICON = str(ICON_PATH) if ICON_PATH.exists() else "〰️"

st.set_page_config(
    page_title="GMFAgent Chat",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",  # 展开侧边栏，监听时「停止监听」按钮在侧边栏顶部
)

st.title("🤖 GMFAgent")
st.caption("A Domain Knowledge-Driven Agent for Ground Motion Field Estimation — Chat with me in natural language.")

try:
    from agent import run_polling_cycle
    from agent_controller import create_controller_agent, show_map as controller_show_map, _app_state
    from config import OUTPUT_BASE, PROJECT_ROOT
except ImportError as e:
    st.error(f"Import failed: {e}. Run from GMFAgent dir: streamlit run app_chat.py")
    st.stop()


def _ensure_session_keys():
    """Ensure all session state keys exist (avoids attribute errors)."""
    defaults = {
        "chat_history": [],
        "polling_active": False,
        "polling_stop": False,
        "polling_seen_ids": set(),
        "poll_log_lines": [],
        "task_log_lines": [],
        "last_result": None,
        "last_map_html": None,
        "last_map_layer": "pga",
        "edit_last_prompt": False,
        "edit_last_content": "",
        "rerun_with_prompt": None,
        "pending_prompt": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_ensure_session_keys()

# Inject session into agent_controller for tools
_app_state["session"] = st.session_state

# Top bar: New chat only; stop monitoring appears in message and sidebar
if st.button("New chat", key="new_chat", help="Clear conversation and start over"):
    st.session_state.chat_history = []
    st.session_state.task_log_lines = []
    st.session_state.last_result = None
    st.session_state.last_map_html = None
    st.session_state.last_map_layer = "pga"
    st.session_state.edit_last_prompt = False
    st.session_state.edit_last_content = ""
    st.session_state.pending_prompt = None
    if "last_map" in _app_state:
        del _app_state["last_map"]
    st.rerun()

# Sidebar: render before prompt/polling so stop button is available during monitoring
with st.sidebar:
    st.subheader("Quick reference")
    st.markdown("""
**What you can ask**
- **Scenario**: mag, depth, lon/lat (optional: region name). *Example in the box below.*
- **Completed events**: list computed events, then "Show map for \`event_id\`"
- **Event detection**: start/stop regional monitoring (e.g. Japan, M≥6)
- **Recent catalog**: query earthquakes by region and magnitude
- **Map layers**: PGA, SA(0.3s/1.0s/3.0s), Population, Vs30, DEM
    """)
    poll_logs_sb = st.session_state.get("poll_log_lines") or []
    if poll_logs_sb:
        _log_title = "Event detection log (stopped)" if not st.session_state.get("polling_active") else "Event detection log"
        with st.expander(_log_title, expanded=True):
            st.code("\n".join(poll_logs_sb[-50:]), language=None)


def _log_cb(msg):
    lst = st.session_state.get("task_log_lines")
    if lst is None:
        st.session_state.task_log_lines = [msg]
    else:
        lst.append(msg)


def _poll_log_cb(msg):
    lst = st.session_state.get("poll_log_lines")
    if lst is None:
        st.session_state.poll_log_lines = [msg]
    else:
        lst.append(msg)


_app_state["log_callback"] = _log_cb

# Welcome: one concrete example (details are in the sidebar Quick reference)
if not st.session_state.chat_history:
    st.info("💡 **Try this**: *Run scenario Japan M7.3, lon 141.18, lat 37.71, depth 20 km* — then say **\"Show map\"** to view the PGA map. For more options see **Quick reference** in the sidebar.")

def _render_content(text):
    """Render text with preserved newlines (\\n -> line breaks in markdown)."""
    if not text:
        return
    st.markdown((text or "").replace("\n", "  \n"))


def _infer_layer_from_text(text: str) -> str:
    """从文本推断图层：pga, sa_0_3, sa_1_0, sa_3_0, population, vs30, dem."""
    if not text:
        return "pga"
    t = text.lower()
    if "人口" in text: return "population"
    if "场地" in text or "vs30" in t: return "vs30"
    if "高程" in text or "dem" in t: return "dem"
    if "0.3" in t or "sa_0_3" in t or "sa0.3" in t: return "sa_0_3"
    if "1.0" in t or "sa_1_0" in t or "sa1.0" in t: return "sa_1_0"
    if "3.0" in t or "sa_3_0" in t or "sa3.0" in t: return "sa_3_0"
    return "pga"


def _parse_monitoring_started(reply: str, user_prompt: str) -> dict | None:
    """若助手回复像「已启动监测」且用户是在请求开始监听，返回 polling_params；否则返回 None。"""
    if not reply or not user_prompt:
        return None
    import re
    r, p = reply.lower(), user_prompt.lower()
    started = (
        ("已启动" in reply or "已成功启动" in reply or "已开始" in reply or "监测开始" in reply or "监测已开始" in reply or "监测已开始运行" in reply
         or "event detection started" in r or "monitoring started" in r or "started for" in r)
        and ("监测" in reply or "监听" in reply or "event" in r or "monitor" in r)
    )
    # 用户明确说了「监听/启动/开始」+ 主题词
    want_start = any(x in p or x in user_prompt for x in ("start", "开始", "启动", "监听")) and any(
        x in p or x in user_prompt for x in ("event", "监测", "监听", "monitor", "japan", "日本", "earthquake", "地震", "事件")
    )
    # 或：回复明确说「已启动/已开始…监测」且用户输入含地区或震级（如「日本, 6.0级」）
    if not want_start and started:
        has_region = any(x in user_prompt for x in ("日本", "japan", "中国", "全球", "global", "阿根廷", "智利", "印尼", "china", "chile", "indonesia"))
        has_mag = bool(re.findall(r"[\d.]+\s*级|M\s*[\d.]+|[\d.]+\s*以上", user_prompt)) or bool(re.findall(r"\d+\.?\d*", user_prompt))
        if has_region or has_mag:
            want_start = True
    if not started or not want_start:
        return None
    region = "japan"
    for k, v in [("日本", "japan"), ("japan", "japan"), ("全球", "global"), ("global", "global"),
                 ("中国", "china"), ("china", "china"), ("阿根廷", "argentina"), ("argentina", "argentina"),
                 ("智利", "chile"), ("chile", "chile"), ("印尼", "indonesia"), ("indonesia", "indonesia")]:
        if k in reply or k in user_prompt:
            region = v
            break
    mag = 5.0
    for m in re.findall(r"M?\s*(\d+\.?\d*)", reply + " " + user_prompt):
        try:
            f = float(m)
            if 3 <= f <= 10:
                mag = f
                break
        except ValueError:
            pass
    interval = 30
    for s in re.findall(r"(\d+)\s*秒", reply) + re.findall(r"interval\s*(\d+)", r) + re.findall(r"(\d+)\s*秒", user_prompt) + re.findall(r"every\s*(\d+)\s*s", r) + re.findall(r"(\d+)\s*s\b", r):
        try:
            i = int(s)
            if 5 <= i <= 300:
                interval = i
                break
        except ValueError:
            pass
    return {"region": region, "min_mag": mag, "poll_interval": interval, "use_ml": True, "n_rounds": 1}


def _event_id_sort_key(name: str):
    """Sort key for event IDs: natural order (1, 2, ..., 10) by first number in name."""
    m = re.search(r"(\d+(?:\.\d+)?)", name)
    return (float(m.group(1)) if m else 0, name)


def _infer_event_id_from_prompt(prompt: str, completed_ids: list) -> str | None:
    """If user explicitly specified an event (e.g. event5, event 8, custom_event5, scenario_7.3_20), return that event_id from completed_ids; else None."""
    if not prompt or not completed_ids:
        return None
    p = prompt.strip().lower()
    # "event5", "event 5", "event 8" -> number 5, 8
    m = re.search(r"event\s*(\d+)", p, re.IGNORECASE)
    if m:
        num = m.group(1)
        # Prefer custom_event{N}_ or preset_00{N} or preset_0{N}
        for eid in completed_ids:
            if eid.startswith(f"custom_event{num}_") or eid == f"custom_event{num}":
                return eid
            if re.match(rf"preset_0*{num}\b", eid, re.IGNORECASE):
                return eid
        # Any id whose first number is N (e.g. scenario_7.3_20 if user said event7)
        for eid in completed_ids:
            if re.search(rf"(?:^|_){re.escape(num)}(?:_|$|\d)", eid):
                return eid
    # Full id or prefix: "custom_event5_...", "scenario_7.3_20", "preset_001"
    for eid in completed_ids:
        if eid.lower() in p or (eid.lower() in p.replace(" ", "")):
            return eid
        if p in eid.lower():
            return eid
    # "custom_event5" or "scenario_7.3" as prefix
    for eid in completed_ids:
        if eid.lower().startswith(p) or p in eid.lower():
            return eid
    return None


def _user_only_asked_for_list(prompt: str) -> bool:
    """用户是否只要求列出（历史/已完成）事件，而不是要看地图."""
    if not prompt:
        return False
    p = prompt.strip()
    list_keywords = (
        "历史事件", "已完成", "有哪些跑完", "跑完的事件", "列表", "读取列表",
        "计算完成", "已经计算完成", "完成的事件", "查看已经计算", "查看完成",
        "list completed", "completed events", "列出事件", "事件列表",
    )
    return any(k in p for k in list_keywords)


def _user_wants_map(prompt: str) -> bool:
    """用户是否在要求显示/查看地图（而非仅列事件列表）."""
    if not prompt or _user_only_asked_for_list(prompt):
        return False
    p = prompt.strip().lower()
    return any(k in p for k in ("地图", "显示", "查看", "map", "display", "show", "图层", "交互"))


def _user_wants_png(prompt: str) -> bool:
    """用户是否在要求 PNG/图片/结果图."""
    if not prompt:
        return False
    p = prompt.strip()
    return any(k in p for k in ("png", "图片", "picture", "image", "导出图", "静态图", "结果图", "看图", "显示图", "要图"))


def _collect_png_paths_from_last_result(ensure_generated: bool = False) -> list:
    """从 last_result 对应输出目录收集已存在的 PNG 路径（pga.png, sa_0_3.png 等）. 返回绝对路径列表.
    若 ensure_generated 且尚无 PNG，尝试调用 save_pga_png 生成."""
    r = st.session_state.get("last_result")
    if not r:
        return []
    pga = r.get("pga_path")
    if not pga:
        return []
    pga_path = Path(pga)
    if not pga_path.exists():
        return []
    out_dir = pga_path.parent.resolve()
    names = ["pga.png", "sa_0_3.png", "sa_1_0.png", "sa_3_0.png"]
    paths = [str((out_dir / n).resolve()) for n in names if (out_dir / n).exists()]
    if not paths and ensure_generated:
        try:
            from gmfagent_tools.MP_DISP import save_pga_png
            ev = r.get("selected_event", {})
            save_pga_png(str(pga_path), str(out_dir), ev.get("lon") or 0, ev.get("lat") or 0)
            paths = [str((out_dir / n).resolve()) for n in names if (out_dir / n).exists()]
        except Exception:
            pass
    return paths

# Chat display: 助手消息靠左，用户消息靠右；地图内嵌在对应助手消息内
_history = st.session_state.chat_history
_last_user_idx = next((i for i in range(len(_history) - 1, -1, -1) if _history[i]["role"] == "user"), None)
for _msg_i, msg in enumerate(_history):
    if msg["role"] == "user":
        col_left, col_right = st.columns([1, 1])
        with col_right:
            with st.chat_message("user"):
                _render_content(msg.get("content", ""))
                # 仅最后一条用户消息显示编辑入口（图标按钮，较隐蔽）
                if _last_user_idx is not None and _msg_i == _last_user_idx and not st.session_state.get("polling_active"):
                    if st.button("✏️", key=f"edit_last_btn_{_msg_i}", help="Edit and resend"):
                        st.session_state.edit_last_prompt = True
                        st.session_state.edit_last_content = msg.get("content", "")
                        st.rerun()
    else:
        with st.chat_message("assistant"):
            _render_content(msg.get("content", ""))
            if msg.get("logs"):
                with st.expander("📋 Execution log", expanded=True):
                    st.code("\n".join(msg["logs"]), language=None)
            if msg.get("monitoring_log"):
                with st.expander("📋 Event detection log", expanded=True):
                    st.code("\n".join(msg["monitoring_log"]), language=None)
            msg_map_html = msg.get("map_html")
            msg_map_layer = msg.get("map_layer", "pga")
            msg_ev_info = msg.get("map_event_info") or {}
            msg_eid = msg.get("map_event_id") or msg_ev_info.get("event_id", "")
            if msg_map_html:
                layer_label = {"pga": "PGA", "sa_0_3": "SA(0.3s)", "sa_1_0": "SA(1.0s)", "sa_3_0": "SA(3.0s)", "population": "Population", "vs30": "Vs30", "dem": "DEM"}.get(msg_map_layer, msg_map_layer)
                with st.expander(f"📍 Interactive Map — {layer_label}", expanded=True):
                    _place, _mag, _depth = msg_ev_info.get("place", ""), msg_ev_info.get("mag", ""), msg_ev_info.get("depth", "")
                    _cap = f"**Event ID**: `{msg_eid}`" + (f" · {_place}" if _place else "") + (f" · M{_mag}" if _mag else "") + (f" · depth {_depth} km" if _depth else "")
                    if _cap.strip():
                        st.caption(_cap)
                    import streamlit.components.v1 as components
                    components.html(msg_map_html, height=500, scrolling=False)
            for img_path in msg.get("png_paths") or []:
                p = Path(img_path).resolve()
                if p.exists():
                    try:
                        with open(p, "rb") as f:
                            st.image(f.read(), use_container_width=True, caption=p.name)
                    except Exception as e:
                        st.caption(f"Image: {p.name}")
                        st.markdown(f"*Render failed: {e}*")
            # During event detection: show stop button and log in this message
            if msg.get("show_stop_ui") and st.session_state.get("polling_active") and not st.session_state.get("polling_stop"):
                st.markdown("---")
                st.error("🔴 **Event detection active** — Click below to stop.")
                if st.button("⏹ Stop monitoring", type="primary", key=f"stop_in_msg_{_msg_i}", use_container_width=False):
                    st.session_state.polling_stop = True
                    st.rerun()
                st.caption("📋 Event detection log")
                _pl = st.session_state.get("poll_log_lines") or []
                st.code("\n".join(_pl[-60:]) if _pl else "Waiting for first poll…", language=None)


# 获取本轮提问：回转 / 待处理（上一轮提交） / 监听中不显示输入 / 编辑上条 / 正常输入
prompt = st.session_state.pop("rerun_with_prompt", None) or st.session_state.pop("pending_prompt", None)
if prompt is None:
    if st.session_state.polling_active and not st.session_state.polling_stop:
        st.text_input("", value="", disabled=True, placeholder="Event detection active. Click 'Stop monitoring' to type again.", key="polling_input_placeholder", label_visibility="collapsed")
        prompt = None
    elif st.session_state.get("edit_last_prompt"):
        with st.form("edit_last_form", clear_on_submit=False):
            _prefill = st.session_state.get("edit_last_content", "")
            st.text_area("Edit last message", value=_prefill, height=120, key="edit_last_ta", placeholder="Edit and click below to resend.")
            _col1, _col2, _ = st.columns([1, 1, 2])
            with _col1:
                _submitted = st.form_submit_button("Resend")
            with _col2:
                _cancel = st.form_submit_button("Cancel")
        if _submitted:
            _new_text = st.session_state.get("edit_last_ta", st.session_state.get("edit_last_content", ""))
            if _new_text.strip():
                _hist = st.session_state.chat_history
                st.session_state.chat_history = _hist[:-2] if len(_hist) >= 2 else []
                st.session_state.rerun_with_prompt = _new_text.strip()
            st.session_state.edit_last_prompt = False
            st.rerun()
        if _cancel:
            st.session_state.edit_last_prompt = False
            st.rerun()
        prompt = None
    else:
        prompt = st.chat_input("Run scenario, start event detection, list completed events, show map…")
        if prompt:
            st.session_state.pending_prompt = prompt
            st.rerun()

# While processing: show thinking state, hide input
if prompt is not None:
    st.caption("🤔 **Thinking…** Input disabled.")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.task_log_lines = []

    # 用户消息靠右
    _c1, _c2 = st.columns([1, 1])
    with _c2:
        with st.chat_message("user"):
            _render_content(prompt)

    with st.chat_message("assistant"):
        agent = create_controller_agent()
        if agent is None:
            msg = "Agent not available (check DEEPSEEK_API_KEY and pydantic-ai)."
            st.error(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
        else:
            with st.spinner("Thinking..."):
                try:
                    history_text = ""
                    for m in st.session_state.chat_history[:-1]:
                        role = "User" if m["role"] == "user" else "Assistant"
                        history_text += f"{role}: {m['content']}\n\n"
                    full_prompt = f"[Previous conversation]\n{history_text}\n[Current] User: {prompt}"
                    result = agent.run_sync(full_prompt)
                    out = getattr(result, "output", None) or getattr(result, "data", None) or str(result)
                    out = str(out) if out is not None else ""
                    task_logs = st.session_state.get("task_log_lines") or []
                    _render_content(out)
                    map_html = st.session_state.get("last_map_html")
                    map_layer = st.session_state.get("last_map_layer", "pga")
                    # If this turn ran a new scenario (task_logs) but the map we have is for a different event, discard stale map so we show the new run's map
                    if map_html and task_logs and st.session_state.get("last_result"):
                        r = st.session_state["last_result"]
                        pga = r.get("pga_path")
                        if pga:
                            try:
                                _p = Path(pga).resolve()
                                if not _p.exists() and PROJECT_ROOT:
                                    _p = (Path(PROJECT_ROOT) / pga).resolve()
                                if _p.exists():
                                    current_eid = _p.parent.name
                                    if current_eid and current_eid != st.session_state.get("last_map_event_id"):
                                        map_html = None
                                        st.session_state.last_map_html = None
                                        st.session_state.pop("last_map_event_id", None)
                                        st.session_state.pop("last_map_event_info", None)
                                        if "last_map" in _app_state:
                                            del _app_state["last_map"]
                            except Exception:
                                pass
                    # 用户只是要列表时，立刻丢弃地图状态，不再参与后续任何逻辑
                    if _user_only_asked_for_list(prompt):
                        if map_html:
                            map_html = None
                            st.session_state.last_map_html = None
                            st.session_state.pop("last_map_event_id", None)
                            st.session_state.pop("last_map_event_info", None)
                            if "last_map" in _app_state:
                                del _app_state["last_map"]
                    # 用户要求看地图或 agent 说已显示但未调工具时，补调 show_map（用用户输入优先推断图层）
                    want_map = (not _user_only_asked_for_list(prompt)) and (
                        _user_wants_map(prompt) or ("displayed" in out.lower() or "map ready" in out.lower())
                    )
                    if not map_html and want_map:
                        _layer = _infer_layer_from_text(prompt) if _user_wants_map(prompt) else _infer_layer_from_text(out)
                        eid = None
                        # Get completed_ids once for both user-specified and fallback
                        out_path = Path(OUTPUT_BASE).resolve()
                        if not out_path.exists() and PROJECT_ROOT:
                            out_path = Path(PROJECT_ROOT) / "output" if (Path(PROJECT_ROOT) / "output").exists() else out_path
                        completed_ids = []
                        if out_path.exists():
                            dirs = [d for d in out_path.iterdir() if d.is_dir() and (d / "grid_pga.csv").exists()]
                            dirs.sort(key=lambda d: _event_id_sort_key(d.name))
                            completed_ids = [d.name for d in dirs]
                        # If user explicitly specified an event (e.g. "show event5 vs30 map"), use that event
                        eid = _infer_event_id_from_prompt(prompt, completed_ids)
                        # Only if no explicit event: use last run (e.g. just ran scenario -> show latest)
                        if not eid and st.session_state.get("last_result"):
                            r = st.session_state["last_result"]
                            pga = r.get("pga_path")
                            if pga:
                                _p = Path(pga).resolve()
                                if _p.exists():
                                    eid = _p.parent.name
                                elif PROJECT_ROOT:
                                    _p_rel = Path(PROJECT_ROOT) / pga
                                    if _p_rel.resolve().exists():
                                        eid = _p_rel.resolve().parent.name
                        # Else match from agent reply text, or use latest in list
                        if not eid and completed_ids:
                            eid = next((x for x in completed_ids if x in out), None)
                            if not eid:
                                eid = completed_ids[-1]
                        if eid:
                            try:
                                controller_show_map(event_id_or_path=eid, layer=_layer)
                                map_html = st.session_state.get("last_map_html")
                                map_layer = st.session_state.get("last_map_layer", _layer)
                            except Exception as _e:
                                if st.session_state.get("task_log_lines") is not None:
                                    _log_cb(f"Fallback map failed: {_e}")
                    # 用户要求看 PNG/图片/结果图时，收集输出目录下的 PNG 并在页面中展示（若无则尝试生成）
                    png_paths = []
                    if _user_wants_png(prompt) or ".png" in out or "png" in out.lower() or "image" in out.lower() or "path" in out.lower():
                        png_paths = _collect_png_paths_from_last_result(ensure_generated=_user_wants_png(prompt))
                    msg_entry = {
                        "role": "assistant",
                        "content": out,
                        "logs": task_logs,
                    }
                    # 用户只是要列表时，即使 Agent 误调了 show_map 也不显示地图，并清掉遗留状态
                    if _user_only_asked_for_list(prompt) and map_html:
                        map_html = None
                        st.session_state.last_map_html = None
                        st.session_state.pop("last_map_event_id", None)
                        st.session_state.pop("last_map_event_info", None)
                        if "last_map" in _app_state:
                            del _app_state["last_map"]
                    if map_html:
                        map_event_id = st.session_state.pop("last_map_event_id", None)
                        map_event_info = st.session_state.pop("last_map_event_info", None) or {}
                        msg_entry["map_html"] = map_html
                        msg_entry["map_layer"] = map_layer
                        msg_entry["map_event_id"] = map_event_id
                        msg_entry["map_event_info"] = map_event_info
                        st.session_state.last_map_html = None
                        if "last_map" in _app_state:
                            del _app_state["last_map"]
                        # 本轮回复中直接渲染地图，避免 rerun 后丢失或未绘制；并展示事件 ID 与基本参数
                        layer_label = {"pga": "PGA", "sa_0_3": "SA(0.3s)", "sa_1_0": "SA(1.0s)", "sa_3_0": "SA(3.0s)", "population": "Population", "vs30": "Vs30", "dem": "DEM"}.get(map_layer, map_layer)
                        _eid, _place, _mag, _depth = map_event_info.get("event_id", "") or map_event_id or "", map_event_info.get("place", ""), map_event_info.get("mag", ""), map_event_info.get("depth", "")
                        _event_caption = f"**Event ID**: `{_eid}`" + (f" · {_place}" if _place else "") + (f" · M{_mag}" if _mag else "") + (f" · depth {_depth} km" if _depth else "")
                        with st.expander(f"📍 Interactive Map — {layer_label}", expanded=True):
                            if _event_caption.strip():
                                st.caption(_event_caption)
                            import streamlit.components.v1 as components
                            components.html(map_html, height=500, scrolling=False)
                    if png_paths:
                        msg_entry["png_paths"] = png_paths
                    # 工具已设置 polling_active，或从回复推断「已启动监测」并补设状态，确保 Stop 一定出现
                    if st.session_state.get("polling_active"):
                        msg_entry["show_stop_ui"] = True
                        _pl = st.session_state.get("poll_log_lines") or []
                        if len(_pl) == 0:
                            from datetime import datetime
                            _pi = st.session_state.get("polling_params", {}).get("poll_interval", 30)
                            _poll_log_cb(f"[{datetime.now().strftime('%H:%M:%S')}] Event detection started; polling every {_pi} s.")
                    else:
                        params = _parse_monitoring_started(out, prompt)
                        if params:
                            st.session_state.polling_active = True
                            st.session_state.polling_stop = False
                            st.session_state.poll_last_time = time.time()
                            st.session_state["last_wait_log_time"] = 0
                            st.session_state.polling_params = params
                            st.session_state.polling_seen_ids = st.session_state.get("polling_seen_ids", set())
                            msg_entry["show_stop_ui"] = True
                            _pl = st.session_state.get("poll_log_lines") or []
                            if len(_pl) == 0:
                                from datetime import datetime
                                _poll_log_cb(f"[{datetime.now().strftime('%H:%M:%S')}] Event detection started; polling every {params.get('poll_interval', 30)} s.")
                    st.session_state.chat_history.append(msg_entry)
                    # 只保留最近 3 条带地图的消息，避免内存无限增长；更早的消息去掉 map_html/map_layer
                    with_map = [i for i, m in enumerate(st.session_state.chat_history) if m.get("map_html")]
                    if len(with_map) > 3:
                        for i in with_map[:-3]:
                            st.session_state.chat_history[i].pop("map_html", None)
                            st.session_state.chat_history[i].pop("map_layer", None)
                            st.session_state.chat_history[i].pop("map_event_id", None)
                            st.session_state.chat_history[i].pop("map_event_info", None)
                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": err,
                        "logs": st.session_state.get("task_log_lines") or [],
                    })

    # 若本轮对话后已开启监听：写一条「监测已启动」日志后直接 rerun，停止按钮与日志仅在对话内显示
    if st.session_state.get("polling_active") and not st.session_state.get("polling_stop"):
        from datetime import datetime
        _params = st.session_state.get("polling_params", {})
        _interval = _params.get("poll_interval", 30)
        _poll_log_cb(f"[{datetime.now().strftime('%H:%M:%S')}] Event detection started; polling every {_interval} s.")
        st.rerun()
    else:
        # 正常回答完毕，刷新一次以恢复输入框
        st.rerun()

# Polling loop (when event detection active)
elif st.session_state.polling_active and not st.session_state.polling_stop:
    from datetime import datetime
    params = st.session_state.get("polling_params", {})
    poll_interval = params.get("poll_interval", 30)
    if "poll_last_time" not in st.session_state:
        st.session_state.poll_last_time = 0
    elapsed = time.time() - st.session_state.poll_last_time
    # 未到轮询间隔：最多每 10 秒打印一次「等待下次检查」，避免刷屏
    plines = st.session_state.get("poll_log_lines") or []
    if elapsed < poll_interval:
        if len(plines) == 0:
            _poll_log_cb(f"[{datetime.now().strftime('%H:%M:%S')}] Event detection started; polling every {poll_interval} s.")
        remain = max(0, int(poll_interval - elapsed))
        last_wait = st.session_state.get("last_wait_log_time", 0)
        if time.time() - last_wait >= 10 or last_wait == 0:
            _poll_log_cb(f"[{datetime.now().strftime('%H:%M:%S')}] Next poll in ~{remain} s.")
            st.session_state.last_wait_log_time = time.time()
    if elapsed >= poll_interval:
        from datetime import datetime
        _poll_log_cb(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Polling earthquake catalog…")
        results, st.session_state.polling_seen_ids = run_polling_cycle(
            st.session_state.polling_seen_ids,
            use_ai_select=True,
            n_rounds=params.get("n_rounds", 1),
            use_ml=params.get("use_ml", True),
            region=params.get("region", "japan"),
            min_mag=params.get("min_mag", 5.0),
            log_callback=_poll_log_cb,
        )
        if results:
            st.session_state.last_result = results[-1]
            _poll_log_cb(f"Processed {len(results)} event(s).")
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Detected and processed {len(results)} new event(s). Latest: {results[-1].get('selected_event', {}).get('place', '')}",
            })
        else:
            from datetime import datetime
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            _poll_log_cb(f"[{ts}] No new events; next poll in {poll_interval} s.")
        st.session_state.poll_last_time = time.time()
        st.session_state.last_wait_log_time = 0  # 下次等待阶段可再打印
    # 未到轮询间隔：每 10 秒 rerun 一次，减少刷新频率
    if elapsed < poll_interval:
        sleep_sec = min(10, max(1, int(poll_interval - elapsed)))
        time.sleep(sleep_sec)
    st.rerun()
elif st.session_state.polling_stop:
    from datetime import datetime
    _poll_log_cb(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] User stopped monitoring.")
    _saved_log = list(st.session_state.get("poll_log_lines") or [])
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": "Event detection stopped.",
        "monitoring_log": _saved_log,
    })
    st.session_state.polling_active = False
    st.session_state.polling_stop = False
    st.session_state["last_wait_log_time"] = 0  # 下次再启动时允许立即打印「等待下次检查」
    st.rerun()


