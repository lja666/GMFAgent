# -*- coding: utf-8 -*-
"""Generate seismogram-style favicon for GMFAgent. Run once: python -m assets.icon_seismogram"""
from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib required: pip install matplotlib")

OUT_DIR = Path(__file__).resolve().parent
SIZE = 64  # favicon size

def main():
    t = np.linspace(0, 4 * np.pi, 200)
    # Seismogram-like waveform: damped oscillation + small noise
    y = np.exp(-0.15 * t) * np.sin(2 * t) + 0.3 * np.sin(5 * t) + 0.05 * np.random.randn(len(t))
    y = (y - y.min()) / (y.max() - y.min() + 1e-8) * 0.9 + 0.05

    fig, ax = plt.subplots(figsize=(1, 1), dpi=SIZE, facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.plot(t, y, color="#00d4aa", linewidth=2, solid_capstyle="round")
    ax.set_xlim(0, 4 * np.pi)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    out_path = OUT_DIR / "icon_seismogram.png"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, facecolor="#0e1117", edgecolor="none")
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
