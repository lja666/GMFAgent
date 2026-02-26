# -*- coding: utf-8 -*-
"""
调试脚本：逐步执行 Japan M7.3 场景并打印每步耗时，用于定位卡住的位置。
在项目根目录运行: python scripts/run_scenario_debug.py

若某一步后长时间无新输出，说明卡在该步。
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _log(msg):
    t = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{t}] {msg}", flush=True)

def main():
    _log("=== 开始调试 Japan M7.3 场景 ===")

    _log("Step 0: 加载 config...")
    try:
        from config import OUTPUT_BASE, DEEPSEEK_API_KEY
        _log(f"  OUTPUT_BASE={OUTPUT_BASE}, DEEPSEEK_API_KEY={'已设置' if DEEPSEEK_API_KEY else '未设置'}")
    except Exception as e:
        _log(f"  FAIL: {e}")
        return

    event = {
        "mag": 7.3,
        "depth": 20,
        "lon": 141.18,
        "lat": 37.71,
        "nation": "Japan",
        "place": "",
        "usgs_id": "scenario_7.3_20",
        "telc_class": "Interface",
        "Mech": "R",
        "rake": 90,
        "event_time": "2025-01-01T00:00:00.000Z",
    }

    _log("Step 1: run_full_pipeline (含 get_grid -> RAG/LLM -> PGA 计算)...")
    _log("  若卡在 'Generating grid...' 后，问题在 get_grid；")
    _log("  若卡在 'Grid done. LLM selecting...' 后，问题在 LLM 或 OpenQuake。")
    t0 = time.time()
    timings = []  # (elapsed_s, message_prefix)

    def _log_cb(msg):
        elapsed = time.time() - t0
        timings.append((elapsed, msg[:50]))
        _log(f"  >> {msg}")

    try:
        from agent import run_full_pipeline
        result = run_full_pipeline(
            event=event,
            use_ai_select=True,
            n_rounds=1,
            use_ml=True,
            plot_grid=False,
            log_callback=_log_cb,
        )
        total = time.time() - t0
        _log(f"  OK in {total:.1f}s")
        if result.get("error"):
            _log(f"  result.error: {result['error']}")
        else:
            _log(f"  pga_path={result.get('pga_path')}")

        if timings:
            _log("--- 各阶段耗时（从上一条 >> 到本条 >> 的间隔）---")
            for i, (elapsed, msg) in enumerate(timings):
                prev = timings[i - 1][0] if i > 0 else 0
                delta = elapsed - prev
                _log(f"  +{delta:.0f}s  {msg}")
            _log(f"  总耗时: {total:.0f}s (~{total/60:.1f} 分钟)")
    except Exception as e:
        _log(f"  FAIL in {time.time()-t0:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return

    _log("=== 结束：若某步后长时间无新输出，即卡在该步 ===")

if __name__ == "__main__":
    main()
