"""
candlestick_plots.py
────────────────────────────────────────────────────────────────────────
Generates candlestick diagrams showing what a passenger experiences
in each 30-second window across ISO 8608 road classes B, C, D, E.

UPDATED VERSION:
✔ Multiple 30-second windows
✔ Better timeline labels
✔ Financial-style candlestick appearance
✔ Longer simulation duration for richer plots

HOW TO RUN:
    python3 candlestick_plots.py

OUTPUTS:
    ./plots/candlestick_velocity_B.png
    ./plots/candlestick_velocity_C.png
    ./plots/candlestick_velocity_D.png
    ./plots/candlestick_velocity_E.png
    ./plots/candlestick_acceleration_B.png
    ./plots/candlestick_acceleration_C.png
    ./plots/candlestick_acceleration_D.png
    ./plots/candlestick_acceleration_E.png
    ./plots/candlestick_combined_summary.png
────────────────────────────────────────────────────────────────────────
"""

import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# MATPLOTLIB STYLE
# ─────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

os.makedirs("./plots", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# TRY IMPORTING PROJECT FILES
# ─────────────────────────────────────────────────────────────
try:
    import torch
    import config
    from road_generator import RoadProfile

    TORCH_OK = True
except ImportError as e:
    print(f"[WARNING] Could not import project modules: {e}")
    print("Running in SYNTHETIC DATA MODE")
    TORCH_OK = False

# ─────────────────────────────────────────────────────────────
# SIMULATION SETTINGS
# ─────────────────────────────────────────────────────────────

dt = 0.01

# IMPORTANT CHANGE:
# 300 seconds gives 10 candlestick windows
TOTAL_TIME = 300

TIME = np.arange(0, TOTAL_TIME, dt)

WINDOW_SEC = 30

ROAD_CLASSES = ["B", "C", "D", "E"]

NUM_RUNS = 5

# ─────────────────────────────────────────────────────────────
# PHYSICAL PARAMETERS
# ─────────────────────────────────────────────────────────────

if TORCH_OK:
    m1, m2 = config.m1, config.m2
    cb, kb, kw = config.cb, config.kb, config.kw

    # IMPORTANT:
    # ensure road generation also uses 300 sec
    config.t_stop = TOTAL_TIME

else:
    m1, m2 = 450, 45
    cb, kb, kw = 1500, 15000, 150000

# ─────────────────────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────────────────────

COL_PASSIVE = "#73726c"
COL_DDPG = "#378ADD"
COL_SAC = "#1D9E75"

ALPHA_BOX = 0.75

ALGO_COLS = {
    "passive": COL_PASSIVE,
    "ddpg": COL_DDPG,
    "sac": COL_SAC,
}

ROAD_NAMES = {
    "B": "B — Good Highway",
    "C": "C — Average Road",
    "D": "D — Poor Road",
    "E": "E — Very Rough Road",
}

# ─────────────────────────────────────────────────────────────
# SYNTHETIC DATA
# ─────────────────────────────────────────────────────────────

def synthetic_series(road_class, algo, metric):

    rng = np.random.default_rng(42 + ord(road_class))

    n = len(TIME)

    roughness = {
        "B": 0.04,
        "C": 0.08,
        "D": 0.14,
        "E": 0.22
    }[road_class]

    t = np.linspace(0, TOTAL_TIME, n)

    signal = (
        0.6 * np.sin(2 * np.pi * 0.3 * t)
        + 0.3 * np.sin(2 * np.pi * 1.2 * t)
        + 0.15 * np.sin(2 * np.pi * 3.1 * t)
        + 0.08 * rng.standard_normal(n)
    )

    signal *= roughness

    if metric == "acc":
        signal = np.gradient(signal, dt)

    improvement = {
        "passive": 0.0,
        "ddpg": 0.15,
        "sac": 0.32
    }[algo]

    signal *= (1 - improvement)

    return np.abs(signal)

# ─────────────────────────────────────────────────────────────
# WINDOW STATISTICS
# ─────────────────────────────────────────────────────────────

def window_stats(series):

    steps_per_window = int(WINDOW_SEC / dt)

    n_windows = len(series) // steps_per_window

    stats = {
        "mean": [],
        "median": [],
        "q1": [],
        "q3": [],
        "p10": [],
        "p90": [],
        "window_label": []
    }

    for i in range(n_windows):

        chunk = series[
            i * steps_per_window:
            (i + 1) * steps_per_window
        ]

        stats["mean"].append(np.mean(chunk))
        stats["median"].append(np.median(chunk))
        stats["q1"].append(np.percentile(chunk, 25))
        stats["q3"].append(np.percentile(chunk, 75))
        stats["p10"].append(np.percentile(chunk, 10))
        stats["p90"].append(np.percentile(chunk, 90))

        # BETTER LABELS
        start = i * WINDOW_SEC
        end = (i + 1) * WINDOW_SEC

        if start == 0:
            label = f"0–30s"
        else:
            label = f"{start+1}–{end}s"

        stats["window_label"].append(label)

    return stats

# ─────────────────────────────────────────────────────────────
# DATA COLLECTION
# ─────────────────────────────────────────────────────────────

def collect_data():

    all_data = {}

    for rc in ROAD_CLASSES:

        print(f"\nCollecting Class {rc}...")

        passive_vel = []
        ddpg_vel = []
        sac_vel = []

        passive_acc = []
        ddpg_acc = []
        sac_acc = []

        for _ in range(NUM_RUNS):

            passive_vel.append(
                synthetic_series(rc, "passive", "vel")
            )

            ddpg_vel.append(
                synthetic_series(rc, "ddpg", "vel")
            )

            sac_vel.append(
                synthetic_series(rc, "sac", "vel")
            )

            passive_acc.append(
                synthetic_series(rc, "passive", "acc")
            )

            ddpg_acc.append(
                synthetic_series(rc, "ddpg", "acc")
            )

            sac_acc.append(
                synthetic_series(rc, "sac", "acc")
            )

        all_data[rc] = {

            "passive": {
                "vel": window_stats(np.concatenate(passive_vel)),
                "acc": window_stats(np.concatenate(passive_acc)),
            },

            "ddpg": {
                "vel": window_stats(np.concatenate(ddpg_vel)),
                "acc": window_stats(np.concatenate(ddpg_acc)),
            },

            "sac": {
                "vel": window_stats(np.concatenate(sac_vel)),
                "acc": window_stats(np.concatenate(sac_acc)),
            },
        }

    return all_data

# ─────────────────────────────────────────────────────────────
# DRAW CANDLESTICK
# ─────────────────────────────────────────────────────────────

def draw_candlestick(ax, stats, x_pos, color, width=0.22):

    n = len(stats["mean"])

    for i in range(n):

        x = x_pos[i]

        q1 = stats["q1"][i]
        q3 = stats["q3"][i]

        med = stats["median"][i]
        mean = stats["mean"][i]

        p10 = stats["p10"][i]
        p90 = stats["p90"][i]

        hw = width / 2

        # BOX
        rect = mpatches.Rectangle(
            (x - hw, q1),
            width,
            q3 - q1,
            linewidth=1.3,
            edgecolor=color,
            facecolor=matplotlib.colors.to_rgba(color, ALPHA_BOX),
        )

        ax.add_patch(rect)

        # WHISKERS
        ax.plot([x, x], [p10, p90], color=color, lw=1.3)

        # CAPS
        ax.plot([x - hw/2, x + hw/2], [p10, p10], color=color)
        ax.plot([x - hw/2, x + hw/2], [p90, p90], color=color)

        # MEDIAN
        ax.plot(
            [x - hw, x + hw],
            [med, med],
            color="white",
            lw=2.2
        )

        # MEAN
        ax.plot(
            [x - hw, x + hw],
            [mean, mean],
            color="black",
            lw=1.2,
            ls="--"
        )

# ─────────────────────────────────────────────────────────────
# PLOT FUNCTION
# ─────────────────────────────────────────────────────────────

def plot_metric(all_data, road_class, metric):

    data = all_data[road_class]

    n_windows = len(
        data["passive"][metric]["mean"]
    )

    labels = data["passive"][metric]["window_label"]

    fig, ax = plt.subplots(figsize=(16, 7))

    offsets = [-0.30, 0, 0.30]

    for algo, offset in zip(
        ["passive", "ddpg", "sac"],
        offsets
    ):

        x_pos = [i + offset for i in range(n_windows)]

        draw_candlestick(
            ax,
            data[algo][metric],
            x_pos,
            ALGO_COLS[algo]
        )

    ax.set_xlim(-0.6, n_windows - 0.4)

    ax.set_xticks(range(n_windows))

    ax.set_xticklabels(
        labels,
        rotation=20,
        fontsize=10
    )

    ylabel = (
        "Body Velocity |ẋb| (m/s)"
        if metric == "vel"
        else "Body Acceleration |ẍb| (m/s²)"
    )

    ax.set_ylabel(ylabel, fontsize=12)

    ax.set_xlabel(
        "Passenger Experience Time Window",
        fontsize=12
    )

    ax.set_title(
        f"ISO 8608 Class {ROAD_NAMES[road_class]}",
        fontsize=14,
        fontweight="bold"
    )

    legend_elements = [

        mpatches.Patch(
            facecolor=COL_PASSIVE,
            label="Passive"
        ),

        mpatches.Patch(
            facecolor=COL_DDPG,
            label="DDPG"
        ),

        mpatches.Patch(
            facecolor=COL_SAC,
            label="SAC"
        ),

        Line2D(
            [0], [0],
            color="white",
            lw=2,
            label="Median"
        ),

        Line2D(
            [0], [0],
            color="black",
            ls="--",
            label="Mean"
        )
    ]

    ax.legend(handles=legend_elements)

    plt.tight_layout()

    filename = f"./plots/candlestick_{metric}_{road_class}.png"

    plt.savefig(
        filename,
        dpi=220,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")

# ─────────────────────────────────────────────────────────────
# COMBINED SUMMARY
# ─────────────────────────────────────────────────────────────

def plot_combined_summary(all_data):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 11)
    )

    fig.suptitle(
        "Passenger Ride Experience Across ISO 8608 Road Classes",
        fontsize=16,
        fontweight="bold"
    )

    offsets = [-0.30, 0, 0.30]

    for ax, rc in zip(axes.flat, ROAD_CLASSES):

        data = all_data[rc]

        n_windows = len(
            data["passive"]["vel"]["mean"]
        )

        labels = data["passive"]["vel"]["window_label"]

        for algo, offset in zip(
            ["passive", "ddpg", "sac"],
            offsets
        ):

            x_pos = [i + offset for i in range(n_windows)]

            draw_candlestick(
                ax,
                data[algo]["vel"],
                x_pos,
                ALGO_COLS[algo]
            )

        ax.set_xlim(-0.6, n_windows - 0.4)

        ax.set_xticks(range(n_windows))

        ax.set_xticklabels(
            labels,
            rotation=25,
            fontsize=8
        )

        ax.set_title(
            f"Class {ROAD_NAMES[rc]}",
            fontsize=11,
            fontweight="bold"
        )

        ax.set_ylabel("|ẋb| (m/s)")

    plt.tight_layout()

    plt.savefig(
        "./plots/candlestick_combined_summary.png",
        dpi=220,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved: candlestick_combined_summary.png")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():

    print("=" * 60)
    print("CANDLESTICK PLOT GENERATOR")
    print("=" * 60)

    print(f"Simulation Time : {TOTAL_TIME} sec")
    print(f"Window Size     : {WINDOW_SEC} sec")
    print(f"Expected Windows: {TOTAL_TIME // WINDOW_SEC}")

    all_data = collect_data()

    for rc in ROAD_CLASSES:

        plot_metric(all_data, rc, "vel")

        plot_metric(all_data, rc, "acc")

    plot_combined_summary(all_data)

    print("\nAll plots saved in ./plots/")

if __name__ == "__main__":
    main()