#!/usr/bin/env python3
"""
SEROW / benchmark trajectory visualizer and evaluator for ANYmal CYN-1.

Default use from the SEROW evaluation root:

    python3 serow_viz.py

The script expects, by default:

    anymal_data/test/cyn-1/groundtruth.csv
    anymal_data/test/cyn-1/serow/fused_state.csv

It also works if serow/fused_state.csv is temporarily replaced by a benchmark
MUSE / IEKF / IS fused_state.csv with columns:

    t_rel,t_abs,px,py,pz,vx,vy,vz,qw,qx,qy,qz

Metrics:
    - ATE and RPE are computed by calling evo_ape/evo_rpe directly.
    - Rotational RPE follows the official benchmark command: evo rot_part RMSE is converted from radians to degrees.
    - Velocity RMSE (ATEvel) uses a time-correct, estimator-agnostic protocol,
      identical for ANY fused_state.csv (benchmark or SEROW format):
        * restrict to the GT/estimator time overlap,
        * linearly interpolate GT velocity onto the estimator timestamps
          (estimator samples farther than --max-gt-gap from the nearest GT
          sample are excluded, so interpolation never bridges GT dropouts),
        * rotate estimator velocities into the GT world frame with an Umeyama
          rotation fitted from position alignment over the same overlap,
        * RMSE over all remaining pairs.
      For benchmark-format files, the published protocol (row-index pairing +
      fixed official rotation) is ADDITIONALLY printed for reproducibility, but
      it is not used in the comparison table: with GT at 200 Hz and estimators
      at ~400 Hz, index pairing compares velocities recorded minutes apart and
      saturates at the decorrelation floor (~0.87 m/s on CYN-1) for every
      estimator, regardless of its actual accuracy.
      To fill the table with like-for-like benchmark values, run this script
      once per benchmark file (--est .../muse/fused_state.csv --label MUSE,
      etc.), note each printed time-correct ATEvel, and pass them back via
      --benchmark-atevel 'MUSE=...,IEKF=...,IS=...'.

Plots:
    - Plotting is only for visualization.
    - Pose is SE(3)-aligned to GT and rebased to the common start.
    - Velocity is rotated by the same velocity-frame alignment used for ATEvel.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp

def _detect_delimiter(csv_path: Path):
    with open(csv_path, "r") as f:
        first_line = f.readline()

    if "\t" in first_line:
        return "\t"
    if "," in first_line:
        return ","
    return None  # whitespace
# -----------------------------------------------------------------------------
# User-requested defaults
# -----------------------------------------------------------------------------
DEFAULT_DATASET_ROOT = Path(".")
DEFAULT_GT = Path("/anymal_data/test/cyn-1/groundtruth.csv")
DEFAULT_EST = Path("/anymal_data/test/cyn-1/serow") / "fused_state.csv"
DEFAULT_WORKDIR = Path(".serow_viz_eval")
DEFAULT_SENSOR = Path("/anymal_data/test/cyn-1/anymal_data.csv")

# -----------------------------------------------------------------------------
# Shared plotting style (ported from terrain_ablation_plots.py)
#
# Same palette, line weights, font sizes, and below-axes legend as the Go2
# terrain-ablation figures, so both robots' plots read as one figure set.
# -----------------------------------------------------------------------------
# Base font sizes, as in terrain_ablation_plots.py.  They are tuned for a
# 7-inch-wide figure that LaTeX scales down to \columnwidth.
FIGURE_SIZE = (7.0, 6.2)
TITLE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 15
TICK_LABEL_FONT_SIZE = 13
LEGEND_FONT_SIZE = 13
PANEL_TITLE_FONT_SIZE = 16
# The metrics line under the title is long; it gets its own smaller size.
METRICS_FONT_SIZE = 12

# The figures here are 15-16 inches wide, not 7.  A literal 13 pt legend on a
# 15-inch canvas has less than half the apparent size it has on the 7-inch
# ablation figures, which is why it reads as tiny.  Fonts and stroke weights
# are therefore multiplied by STYLE_SCALE.
#
# Matching the ablation figures' final on-page size exactly would need
# 15/7 = 2.14, but at that scale the labels of a 3-panel figure collide.
# 1.5 is the practical compromise for these wide multi-panel layouts; raise it
# toward 2.1 for the single-column 3x1 stacks, or set it to 1.0 to use the
# numbers above literally.
STYLE_SCALE = 1.5


def _scale(fig=None) -> float:
    return STYLE_SCALE


def _fs(size: float, fig=None) -> float:
    """Font size scaled for these wide canvases."""
    return size * STYLE_SCALE


# Vertical bands, in inches, so spacing looks the same on a 15x7 and a 15x10
# figure.  TITLE_BAND is the space kept clear above the axes; TITLE_TOP is how
# far the title text sits from the top edge; LEGEND_BAND is the space below.
# These scale with the fonts, or big text would overflow its band.
TITLE_TOP_INCHES = 0.22
TITLE_BAND_INCHES = 0.45
SUBTITLE_GAP_INCHES = 0.30
SUBTITLE_BAND_INCHES = 0.28
LEGEND_BAND_INCHES = 0.42
LEGEND_PAD_INCHES = 0.04
LEGEND_COLUMNS = 3

# The metrics line (ATE / ATEvel / RPE) is printed to the terminal anyway and
# does not belong on a paper figure.  Set True to put it back under the title.
SHOW_METRICS_IN_TITLE = False

# Start both pose curves at the origin instead of showing the whole-trajectory
# alignment residual at t=0.  Plotting only; metrics are unaffected.
REBASE_PLOT_TO_ORIGIN = True

# The reference is drawn FIRST as a wide light band and the estimate goes on
# top as a thin opaque line.  Dash patterns are avoided: these signals are
# sampled far faster than the figure has pixels, so dashes alias into solid
# lines and a thick reference simply paints over the estimate.
# `solid_capstyle="butt"` stops each sample from rendering as a rounded blob.
GROUND_TRUTH_STYLE = {
    "color": "#090808",
    "linestyle": "--",
    "linewidth": 1.0,
    "alpha": 1.0,
    "solid_capstyle": "butt",
    "zorder": 1,
}
ESTIMATE_STYLE = {
    "color": "#0072b2",
    "linestyle": "-",
    "linewidth": 0.9,
    "alpha": 1.0,
    "solid_capstyle": "butt",
    "zorder": 3,
}
SECONDARY_STYLE = {
    "color": "#d55e00",
    "linestyle": "-",
    "linewidth": 0.9,
    "alpha": 1.0,
    "solid_capstyle": "butt",
    "zorder": 2,
}
ZERO_LINE_STYLE = {
    "color": "#4d4d4d",
    "linestyle": "--",
    "linewidth": 1.0,
    "alpha": 0.8,
    "zorder": 1,
}
ESTIMATE_COLOR = ESTIMATE_STYLE["color"]
SECONDARY_COLOR = SECONDARY_STYLE["color"]
REFERENCE_COLOR = GROUND_TRUTH_STYLE["color"]

# Axis labels and panel titles, in the paper's notation: a left superscript
# for the frame ({w} = world) and a subscript for the body ({b} = base), e.g.
# ${}^{w}p^{x}_{b}$ for the x-component of the world-frame base position.
# Rendered with matplotlib mathtext in the default DejaVu Sans, so the figure
# needs no LaTeX installation and matches the Go2/G1 ablation figures.
POSE_FIGURE_TITLE = "Pose estimate vs ground truth"
VELOCITY_FIGURE_TITLE = "Velocity estimate vs ground truth"
TRAJECTORY_PANEL_TITLE = "(a) XY trajectory"
POSITION_PANEL_TITLE = r"(b) Position ${}^{w}\mathbf{p}_{b}$"
ORIENTATION_PANEL_TITLE = r"(c) Orientation ${}^{w}\mathbf{R}_{b}$"
VELOCITY_PANEL_TITLE = r"Linear velocity ${}^{w}\mathbf{v}_{b}$"
POS_LABELS = [r"${}^{w}p^{x}_{b}$ [m]", r"${}^{w}p^{y}_{b}$ [m]", r"${}^{w}p^{z}_{b}$ [m]"]
VEL_LABELS = [r"${}^{w}v^{x}_{b}$ [m/s]", r"${}^{w}v^{y}_{b}$ [m/s]", r"${}^{w}v^{z}_{b}$ [m/s]"]
# Short unit strings: at the scaled font size, "roll [deg]" is tall enough
# to collide with the label of the panel above it.
ORI_LABELS = [r"roll [$^\circ$]", r"pitch [$^\circ$]", r"yaw [$^\circ$]"]

# Vector output with ~10^5 points per axis makes huge PDFs.  Curves are
# rasterized inside the axes; text, ticks, and grid stay vector.
RASTERIZE_LINES = True
RASTERIZED_DPI = 600

# Default location and formats for automatically saved figures.
DEFAULT_PLOT_DIR = Path("figures")
DEFAULT_PLOT_FORMATS = ("pdf",)


def save_figure(figure, output_directory: Path, stem: str,
                formats: Iterable[str] = ("pdf",), dpi: int = 300) -> None:
    """Write one figure to <output_directory>/<stem>.<ext> for each format.

    ``figure`` may be None: the plot helpers return None when they skip a
    figure because of missing CSV columns, and skipping should not be fatal.
    """
    if figure is None:
        return
    output_directory.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        output_path = output_directory / f"{stem}.{extension}"
        save_kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            save_kwargs["dpi"] = dpi
        elif RASTERIZE_LINES:
            # For PDF/SVG, dpi only sets the resolution of the rasterized
            # curves; axes, text, and ticks stay vector.
            save_kwargs["dpi"] = RASTERIZED_DPI
        figure.savefig(output_path, format=extension, **save_kwargs)
        print(f"Saved: {output_path}")


def _draw(axis, x, y, style: dict, label: Optional[str] = None, step: bool = False):
    """Plot one series in the shared style, rasterizing the curve itself."""
    plot_fn = axis.step if step else axis.plot
    kwargs = dict(style)
    # Scale stroke weights with the fonts, or a 1 pt line on a 15-inch figure
    # becomes a hairline next to 28 pt labels.
    scale = _scale(axis.figure)
    kwargs["linewidth"] = kwargs.get("linewidth", 1.0) * scale
    if step:
        kwargs["where"] = "post"
        kwargs.pop("solid_capstyle", None)
    line = plot_fn(x, y, label=label, **kwargs)[0]
    if RASTERIZE_LINES:
        line.set_rasterized(True)
    return line


def _style_axis(axis, ylabel: Optional[str] = None, title: Optional[str] = None) -> None:
    if ylabel is not None:
        axis.set_ylabel(ylabel, fontsize=_fs(AXIS_LABEL_FONT_SIZE, axis.figure))
    if title is not None:
        axis.set_title(title, fontsize=_fs(PANEL_TITLE_FONT_SIZE, axis.figure), pad=5)
    axis.tick_params(axis="both", labelsize=_fs(TICK_LABEL_FONT_SIZE, axis.figure))
    axis.grid(True, alpha=0.3)


def _figure_titles(fig, title: str, subtitle: Optional[str] = None) -> None:
    """Title just above the axes, with an optional smaller line under it.

    The offsets are in inches, converted to figure fractions.  A fixed
    fraction (the previous behaviour) put the title an inch clear of the axes
    on a 7-inch-tall figure and much closer on a short one.
    """
    height = fig.get_figheight()
    fig.suptitle(title, fontsize=_fs(TITLE_FONT_SIZE, fig),
                 y=1.0 - TITLE_TOP_INCHES * _scale(fig) / height)
    if subtitle and SHOW_METRICS_IN_TITLE:
        fig.text(
            0.5,
            1.0 - (TITLE_TOP_INCHES + SUBTITLE_GAP_INCHES) * _scale(fig) / height,
            subtitle, ha="center", va="top",
            fontsize=_fs(METRICS_FONT_SIZE, fig), color="#333333",
        )


def _bottom_legend(fig, handles, labels, has_subtitle: bool = False) -> None:
    """Single-row legend just below the axes, as in terrain_ablation_plots.py.

    Bands are reserved in inches so the gaps look the same on a 15x7 pose
    figure and a 15x10 force figure.  Outside the data area, the legend can
    never cover a trace.
    """
    height = fig.get_figheight()
    scale = _scale(fig)
    title_band = TITLE_BAND_INCHES
    if has_subtitle and SHOW_METRICS_IN_TITLE:
        title_band += SUBTITLE_BAND_INCHES

    fig.tight_layout(
        rect=(0.0, LEGEND_BAND_INCHES * scale / height,
              1.0, 1.0 - title_band * scale / height)
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, LEGEND_PAD_INCHES * scale / height),
        ncol=LEGEND_COLUMNS,
        frameon=False,
        fontsize=_fs(LEGEND_FONT_SIZE, fig),
        handlelength=2.4,
        columnspacing=1.6,
        borderaxespad=0.0,
    )


# -----------------------------------------------------------------------------
# Official benchmark constants
# -----------------------------------------------------------------------------
BENCHMARK_RESULTS = {
    "MUSE": {
        "ATE": 2.269461,
        "ATEvel": 0.085253,
        "RPE_1m_trans": 0.072223,
        "RPE_1f_trans": 0.000670,
        "RPE_1m_rot": 0.578469,
        "RPE_1f_rot": 0.002605,
    },
    "IEKF": {
        "ATE": 1.405668,
        "ATEvel": 0.081476,
        "RPE_1m_trans": 0.043187,
        "RPE_1f_trans": 0.000544,
        "RPE_1m_rot": 0.568987,
        "RPE_1f_rot": 0.002611,
    },
    "IS": {
        "ATE": 1.363114,
        "ATEvel": 0.081454,
        "RPE_1m_trans": 0.042526,
        "RPE_1f_trans": 0.001965,
        "RPE_1m_rot": 0.565644,
        "RPE_1f_rot": 0.002614,
    },
}

# Official compute_vel_rmse.py rotations: estimator frame -> GT frame.
ROT_MUSE = np.array([
    [0.71744476, -0.69589424, -0.03168947],
    [0.69644101,  0.71754045,  0.01027768],
    [0.01558629, -0.02944351,  0.99944492],
])
ROT_IEKF = np.array([
    [0.71849142, -0.69482244, -0.03149382],
    [0.69527739,  0.71872202,  0.00529164],
    [0.01895855, -0.02569894,  0.99948994],
])
ROT_IS = np.array([
    [0.71790964, -0.69544522, -0.03101111],
    [0.69589757,  0.71811755,  0.00580931],
    [0.01822957, -0.02575112,  0.99950216],
])
OFFICIAL_VEL_ROTATIONS = {
    "MUSE": ROT_MUSE,
    "IEKF": ROT_IEKF,
    "IS": ROT_IS,
}


@dataclass
class Trajectory:
    name: str
    fmt: str
    t: np.ndarray
    p: np.ndarray
    q_xyzw: np.ndarray
    v: np.ndarray


# -----------------------------------------------------------------------------
# Path / CSV utilities
# -----------------------------------------------------------------------------
def resolve_path(dataset_root: Path, requested: Path, fallback_relative: Optional[Path] = None) -> Path:
    """
    Resolve paths robustly while preserving the user-requested constants.

    The requested DEFAULT_GT / DEFAULT_EST start with '/anymal_data/...'. On a
    normal Linux system that is absolute, but in this project it is usually
    intended as relative to dataset_root. We therefore try both forms.
    """
    candidates: List[Path] = []

    candidates.append(requested)
    if requested.is_absolute():
        candidates.append(dataset_root / str(requested).lstrip(os.sep))
        candidates.append(Path.cwd() / str(requested).lstrip(os.sep))
    else:
        candidates.append(dataset_root / requested)
        candidates.append(Path.cwd() / requested)

    if fallback_relative is not None:
        candidates.append(dataset_root / fallback_relative)
        candidates.append(Path.cwd() / fallback_relative)

    # Remove duplicates while keeping order.
    unique: List[Path] = []
    seen = set()
    for c in candidates:
        key = str(c)
        if key not in seen:
            unique.append(c)
            seen.add(key)

    for c in unique:
        if c.exists():
            return c

    tried = "\n  ".join(str(c) for c in unique)
    raise FileNotFoundError(f"Could not resolve path. Tried:\n  {tried}")


def read_csv_auto(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def require_columns(df: pd.DataFrame, cols: Iterable[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def normalize_quaternions(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=1)
    ok = n > 1e-12
    if not np.all(ok):
        q = q[ok]
        n = n[ok]
    return q / n[:, None]

def plot_imu_biases(est_csv: Path, max_points: int = 30000):
    """
    Plot SEROW IMU bias estimates:
      accel bias: bias_ax, bias_ay, bias_az  [m/s^2]
      gyro  bias: bias_gx, bias_gy, bias_gz  [rad/s]

    If the estimator file is benchmark/MUSE format and does not contain biases,
    the function exits cleanly.
    """

    required_cols = [
        "t",
        "bias_ax", "bias_ay", "bias_az",
        "bias_gx", "bias_gy", "bias_gz",
    ]

    delimiter = _detect_delimiter(est_csv)

    try:
        data = np.genfromtxt(
            est_csv,
            delimiter=delimiter,
            names=True,
            dtype=float,
            encoding=None,
        )
    except Exception as e:
        print(f"[WARN] Could not read IMU biases from {est_csv}: {e}")
        return

    if data.size == 0:
        print(f"[WARN] Empty estimator file, cannot plot IMU biases: {est_csv}")
        return

    available_cols = list(data.dtype.names)

    missing = [c for c in required_cols if c not in available_cols]
    if missing:
        print("[INFO] IMU bias plot skipped.")
        print(f"       Missing columns in estimator file: {missing}")
        return

    t = np.asarray(data["t"], dtype=float)
    t_rel = t - t[0]

    bias_acc = np.column_stack([
        np.asarray(data["bias_ax"], dtype=float),
        np.asarray(data["bias_ay"], dtype=float),
        np.asarray(data["bias_az"], dtype=float),
    ])

    bias_gyro = np.column_stack([
        np.asarray(data["bias_gx"], dtype=float),
        np.asarray(data["bias_gy"], dtype=float),
        np.asarray(data["bias_gz"], dtype=float),
    ])

    # Downsample only for plotting speed, not for computation
    n = len(t_rel)
    if n > max_points:
        step = int(np.ceil(n / max_points))
        idx = np.arange(0, n, step)
        t_rel_plot = t_rel[idx]
        bias_acc_plot = bias_acc[idx]
        bias_gyro_plot = bias_gyro[idx]
    else:
        t_rel_plot = t_rel
        bias_acc_plot = bias_acc
        bias_gyro_plot = bias_gyro

    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
    _figure_titles(fig, "Estimated IMU Biases")

    acc_names = ["bias_ax", "bias_ay", "bias_az"]
    gyro_names = ["bias_gx", "bias_gy", "bias_gz"]

    handles: List = []
    for i in range(3):
        ax = axes[0, i]
        bias_line = _draw(ax, t_rel_plot, bias_acc_plot[:, i], ESTIMATE_STYLE, "Estimated bias")
        zero_line = ax.axhline(0.0, label="Zero", **ZERO_LINE_STYLE)
        _style_axis(ax, ylabel="Accel bias [m/s²]", title=acc_names[i])
        if not handles:
            handles = [bias_line, zero_line]

    for i in range(3):
        ax = axes[1, i]
        _draw(ax, t_rel_plot, bias_gyro_plot[:, i], ESTIMATE_STYLE, "Estimated bias")
        ax.axhline(0.0, **ZERO_LINE_STYLE)
        _style_axis(ax, ylabel="Gyro bias [rad/s]", title=gyro_names[i])
        ax.set_xlabel("Time [s]", fontsize=_fs(AXIS_LABEL_FONT_SIZE, ax.figure))

    _bottom_legend(fig, handles, ["Estimated bias", "Zero"])
    return fig
    
    
# NOTE: an earlier, identical-in-purpose definition of
# plot_pseudo_forces_vs_contacts lived here and was silently shadowed by the
# one defined further down; only the later one ever ran.  Removed.

# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def load_groundtruth(path: Path) -> Trajectory:
    df = read_csv_auto(path)
    required = ["t", "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz"]
    require_columns(df, required, path)

    t = df["t"].to_numpy(float)
    p = df[["x", "y", "z"]].to_numpy(float)
    q = df[["qx", "qy", "qz", "qw"]].to_numpy(float)
    v = df[["vx", "vy", "vz"]].to_numpy(float)

    return clean_trajectory("GT", "groundtruth", t, p, q, v)


def detect_estimator_format(df: pd.DataFrame) -> str:
    cols = set(df.columns)

    # Official benchmark MUSE/IEKF/IS format.
    if {"t_rel", "t_abs", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz"}.issubset(cols):
        return "benchmark"

    # SEROW format produced by anymal_csv_test.cpp.
    if {"t", "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz"}.issubset(cols):
        return "serow"

    # Some benchmark helper scripts call this anymal_state format.
    if {"t", "px", "py", "pz", "qx", "qy", "qz", "qw"}.issubset(cols):
        return "anymal_state"

    raise ValueError(
        "Could not detect estimator format. Supported formats are:\n"
        "  SEROW:     t,x,y,z,qx,qy,qz,qw,vx,vy,vz\n"
        "  Benchmark: t_rel,t_abs,px,py,pz,vx,vy,vz,qw,qx,qy,qz\n"
        f"Available columns: {list(df.columns)}"
    )


def load_estimator(path: Path, label: str = "SBEE") -> Trajectory:
    df = read_csv_auto(path)
    fmt = detect_estimator_format(df)

    if fmt == "benchmark":
        required = ["t_abs", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz"]
        require_columns(df, required, path)
        t = df["t_abs"].to_numpy(float)
        p = df[["px", "py", "pz"]].to_numpy(float)
        q = df[["qx", "qy", "qz", "qw"]].to_numpy(float)  # convert qw,qx,qy,qz file to xyzw
        v = df[["vx", "vy", "vz"]].to_numpy(float)
        return clean_trajectory(label, fmt, t, p, q, v)

    if fmt == "serow":
        required = ["t", "x", "y", "z", "qx", "qy", "qz", "qw", "vx", "vy", "vz"]
        require_columns(df, required, path)
        t = df["t"].to_numpy(float)
        p = df[["x", "y", "z"]].to_numpy(float)
        q = df[["qx", "qy", "qz", "qw"]].to_numpy(float)
        v = df[["vx", "vy", "vz"]].to_numpy(float)
        return clean_trajectory(label, fmt, t, p, q, v)

    # anymal_state fallback: no velocity columns are guaranteed.
    required = ["t", "px", "py", "pz", "qx", "qy", "qz", "qw"]
    require_columns(df, required, path)
    t = df["t"].to_numpy(float)
    p = df[["px", "py", "pz"]].to_numpy(float)
    q = df[["qx", "qy", "qz", "qw"]].to_numpy(float)
    if {"vx", "vy", "vz"}.issubset(set(df.columns)):
        v = df[["vx", "vy", "vz"]].to_numpy(float)
    else:
        v = np.zeros_like(p)
    return clean_trajectory(label, fmt, t, p, q, v)

def plot_contact_probabilities_one_to_one(
    est_csv: Path,
    sensor_csv: Path,
    max_points: int = 50000,
    label: str = "Estimated contact confidence",
):
    """
    Plot estimated contact probabilities from fused_state.csv
    against the binary contact flags from anymal_data.csv, aligned by timestamp.
    """
    contact_cols = ["contact_LF", "contact_RF", "contact_LH", "contact_RH"]
    leg_names = ["LF", "RF", "LH", "RH"]

    try:
        est_df = read_csv_auto(est_csv)
        sensor_df = read_csv_auto(sensor_csv)
    except Exception as e:
        print(f"[WARN] Could not read contact files: {e}")
        return

    if not all(c in est_df.columns for c in ["t"] + contact_cols) or \
       not all(c in sensor_df.columns for c in ["t"] + contact_cols):
        print("[INFO] Contact plot skipped due to missing columns.")
        return

    t_est = est_df["t"].to_numpy(float)
    t_sensor = sensor_df["t"].to_numpy(float)
    t_0 = t_est[0] # Baseline time

    contact_prob = est_df[contact_cols].to_numpy(float)
    contact_flag = (sensor_df[contact_cols].to_numpy(float) > 0.5).astype(float)

    fig, axes = plt.subplots(4, 1, figsize=(15, 9), sharex=True)
    _figure_titles(fig, "Estimated Contact Confidence vs Binary Contact Flags")

    handles: List = []
    for i, ax in enumerate(axes):
        # Binary flag first, as the wide reference band underneath.
        flag_line = _draw(
            ax, t_sensor - t_0, contact_flag[:, i], GROUND_TRUTH_STYLE,
            "Binary contact flag", step=True,
        )
        prob_line = _draw(
            ax, t_est - t_0, contact_prob[:, i], ESTIMATE_STYLE, label,
        )
        _style_axis(ax, ylabel=leg_names[i])
        ax.set_ylim(-0.05, 1.05)
        if not handles:
            handles = [flag_line, prob_line]

    # Limit the x-axis to the estimator's lifespan so the plot isn't stretched out
    axes[-1].set_xlim(0, t_est[-1] - t_0)
    axes[-1].set_xlabel("Time [s]", fontsize=_fs(AXIS_LABEL_FONT_SIZE, fig))
    _bottom_legend(fig, handles, ["Binary contact flag", label])
    return fig


def plot_pseudo_forces_vs_contacts(est_csv: Path, sensor_csv: Path, max_points: int = 50000):
    """
    Plots the pseudo-forces (fz) generated by the estimator alongside 
    the binary contact flags from the original sensor data.
    """
    force_cols = ["fz_LF", "fz_RF", "fz_LH", "fz_RH"]
    contact_cols = ["contact_LF", "contact_RF", "contact_LH", "contact_RH"]
    leg_names = ["LF", "RF", "LH", "RH"]

    try:
        est_df = read_csv_auto(est_csv)
        sensor_df = read_csv_auto(sensor_csv)
    except Exception as e:
        print(f"[WARN] Could not read files for force vs contact plot: {e}")
        return

    if not all(c in est_df.columns for c in ["t"] + force_cols) or \
       not all(c in sensor_df.columns for c in ["t"] + contact_cols):
        print("[INFO] Force vs Contact plot skipped due to missing columns.")
        return

    # Extract Data and Time Vectors
    t_est = est_df["t"].to_numpy(float)
    t_sensor = sensor_df["t"].to_numpy(float)
    t_0 = t_est[0]

    forces = est_df[force_cols].to_numpy(float)
    contacts = (sensor_df[contact_cols].to_numpy(float) > 0.5).astype(float) 

    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    _figure_titles(fig, "Pseudo Forces vs. Binary Contact Flags")

    handles: List = []
    for i, ax1 in enumerate(axes):
        # Primary Y-axis: Forces (Using estimator time)
        force_line = _draw(
            ax1, t_est - t_0, forces[:, i],
            {**ESTIMATE_STYLE, "linewidth": 1.2}, "Pseudo force ($F_z$)",
        )
        # Short labels: the panels are ~2 inches tall and a long rotated label
        # at the scaled font size runs into the row above.
        ax1.set_ylabel(f"{leg_names[i]} [N]", color=ESTIMATE_COLOR,
                       fontsize=_fs(AXIS_LABEL_FONT_SIZE, fig))
        ax1.tick_params(axis="y", labelcolor=ESTIMATE_COLOR,
                        labelsize=_fs(TICK_LABEL_FONT_SIZE, fig))
        ax1.tick_params(axis="x", labelsize=_fs(TICK_LABEL_FONT_SIZE, fig))
        ax1.grid(True, alpha=0.3)

        # Secondary Y-axis: Contact Flags (Using sensor time)
        ax2 = ax1.twinx()
        flag_line = _draw(
            ax2, t_sensor - t_0, contacts[:, i],
            {**SECONDARY_STYLE, "linewidth": 1.2, "alpha": 0.85},
            "Contact flag", step=True,
        )
        ax2.set_ylabel("Contact", color=SECONDARY_COLOR,
                       fontsize=_fs(AXIS_LABEL_FONT_SIZE, fig))
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.tick_params(axis="y", labelcolor=SECONDARY_COLOR,
                        labelsize=_fs(TICK_LABEL_FONT_SIZE, fig))

        if not handles:
            handles = [force_line, flag_line]

    axes[-1].set_xlim(0, t_est[-1] - t_0)
    axes[-1].set_xlabel("Time [s]", fontsize=_fs(AXIS_LABEL_FONT_SIZE, fig))
    _bottom_legend(fig, handles, ["Pseudo force ($F_z$)", "Contact flag"])
    return fig

def clean_trajectory(name: str, fmt: str, t: np.ndarray, p: np.ndarray, q: np.ndarray, v: np.ndarray) -> Trajectory:
    valid = (
        np.isfinite(t)
        & np.all(np.isfinite(p), axis=1)
        & np.all(np.isfinite(q), axis=1)
        & np.all(np.isfinite(v), axis=1)
        & (np.linalg.norm(q, axis=1) > 1e-12)
    )

    t = t[valid]
    p = p[valid]
    q = q[valid]
    v = v[valid]

    order = np.argsort(t)
    t = t[order]
    p = p[order]
    q = q[order]
    v = v[order]

    _, unique_idx = np.unique(t, return_index=True)
    unique_idx = np.sort(unique_idx)
    t = t[unique_idx]
    p = p[unique_idx]
    q = q[unique_idx]
    v = v[unique_idx]

    q = normalize_quaternions(q)
    if len(q) != len(t):
        raise RuntimeError("Quaternion normalization unexpectedly changed trajectory length.")

    return Trajectory(name=name, fmt=fmt, t=t, p=p, q_xyzw=q, v=v)


# -----------------------------------------------------------------------------
# TUM + evo metrics
# -----------------------------------------------------------------------------
def write_tum(traj: Trajectory, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack([traj.t, traj.p, traj.q_xyzw])
    np.savetxt(out_path, arr, fmt="%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f")


def parse_evo_rmse(output: str) -> float:
    """
    Parse exactly the 'rmse' row from evo output.
    Do not parse the first float: evo prints max first.
    """
    for line in output.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "rmse":
            return float(parts[1])
    raise RuntimeError("Could not find RMSE in evo output. Full output:\n" + output)


def run_evo_rmse(cmd: List[str]) -> float:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "evo command failed:\n"
            + " ".join(cmd)
            + "\nOutput:\n"
            + proc.stdout
        )
    return parse_evo_rmse(proc.stdout)


def compute_evo_ate_only(gt_tum: Path, est_tum: Path) -> Dict[str, float]:
    """Fast/debug metric path: compute only ATE with evo_ape."""
    gt_s = str(gt_tum)
    est_s = str(est_tum)
    return {
        "ATE": run_evo_rmse([
            "evo_ape", "tum", gt_s, est_s, "-a"
        ])
    }


def compute_evo_metrics(gt_tum: Path, est_tum: Path) -> Dict[str, float]:
    gt_s = str(gt_tum)
    est_s = str(est_tum)

    metrics = {}
    metrics["ATE"] = run_evo_rmse([
        "evo_ape", "tum", gt_s, est_s, "-a"
    ])

    metrics["RPE_1m_trans"] = run_evo_rmse([
        "evo_rpe", "tum", gt_s, est_s,
        "--delta", "1", "--delta_unit", "m",
        "--pose_relation", "point_distance",
        "-a",
    ])

    metrics["RPE_1f_trans"] = run_evo_rmse([
        "evo_rpe", "tum", gt_s, est_s,
        "--delta", "1", "--delta_unit", "f",
        "--pose_relation", "point_distance",
        "-a",
    ])

    # Official convert_to_tum.py prints commands with two --pose_relation options:
    #   --pose_relation angle_deg ... --pose_relation rot_part -a
    # argparse/evo uses the last one, i.e. rot_part. evo reports this rotation
    # RMSE in radians, while the paper table reports degrees. Therefore we run
    # rot_part and convert the RMSE by 180/pi. Do NOT use angle_deg here; for
    # these trajectories it gives the large ~25 deg values you observed.
    metrics["RPE_1m_rot"] = run_evo_rmse([
        "evo_rpe", "tum", gt_s, est_s,
        "--delta", "1", "--delta_unit", "m",
        "--pose_relation", "rot_part",
        "-a",
    ])

    metrics["RPE_1f_rot"] = run_evo_rmse([
        "evo_rpe", "tum", gt_s, est_s,
        "--delta", "1", "--delta_unit", "f",
        "--pose_relation", "rot_part",
        "-a",
    ])

    return metrics


# -----------------------------------------------------------------------------
# Velocity RMSE, following official benchmark behaviour
# -----------------------------------------------------------------------------
def time_correct_velocity_pairs(
    gt: Trajectory, est: Trajectory, max_gt_gap: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Pair GT and estimator velocities at the SAME instants.

    Estimator samples inside the GT time span are kept, and GT velocity is
    linearly interpolated onto the estimator timestamps. Estimator samples
    farther than max_gt_gap [s] from the nearest GT sample (GT dropouts) are
    excluded so interpolation never bridges large holes.

    This replaces two flawed pairings used previously:
      * row-index pairing (official benchmark script fallback): with GT and
        estimator files at different rates/starts it compares velocities
        recorded minutes apart and saturates at the decorrelation floor;
      * exact-timestamp intersection: at epoch-scale doubles, round(t, 9) is
        below the float ULP, so it silently matched only a handful of samples.

    Returns (t, gt_v, est_v, n_excluded_by_gap).
    """
    mask = (est.t >= gt.t[0]) & (est.t <= gt.t[-1])
    t = est.t[mask]
    if len(t) < 10:
        raise RuntimeError(
            "GT and estimator trajectories share fewer than 10 samples in time. "
            f"GT span [{gt.t[0]:.3f}, {gt.t[-1]:.3f}], "
            f"EST span [{est.t[0]:.3f}, {est.t[-1]:.3f}]. "
            "Check that both files are stamped on the same clock."
        )

    # Distance to the nearest GT timestamp, to detect GT dropouts.
    j = np.searchsorted(gt.t, t)
    j_lo = np.clip(j - 1, 0, len(gt.t) - 1)
    j_hi = np.clip(j, 0, len(gt.t) - 1)
    gap = np.minimum(np.abs(t - gt.t[j_lo]), np.abs(gt.t[j_hi] - t))
    ok = gap <= max_gt_gap
    n_excluded = int(np.count_nonzero(~ok))

    t = t[ok]
    est_v = est.v[mask][ok]
    gt_v = np.column_stack([
        np.interp(t, gt.t, gt.v[:, 0]),
        np.interp(t, gt.t, gt.v[:, 1]),
        np.interp(t, gt.t, gt.v[:, 2]),
    ])
    return t, gt_v, est_v, n_excluded


def official_benchmark_atevel(
    gt: Trajectory, est: Trajectory, metrics: Dict[str, float], label: str
) -> Tuple[float, str]:
    """
    Replicate the published benchmark compute_vel_rmse.py protocol exactly:
    row-index pairing (gt.v[:n] vs est.v[:n]) + the fixed official rotation.
    Reported for reproducibility only; see the module docstring for why this
    is not a valid accuracy metric when GT and estimator rates/starts differ.
    """
    detected = classify_benchmark_estimator(metrics, label)
    R_off = OFFICIAL_VEL_ROTATIONS[detected]
    n = min(len(gt.v), len(est.v))
    return rmse_vector(gt.v[:n], (R_off @ est.v[:n].T).T), detected


def rmse_vector(gt_v: np.ndarray, est_v: np.ndarray) -> float:
    err = est_v - gt_v
    return float(np.sqrt(np.mean(np.sum(err * err, axis=1))))


def se3_umeyama_no_scale(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid alignment: target ~= R_align * source + t_align."""
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)
    A = source - mu_s
    B = target - mu_t
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1.0
        R_align = Vt.T @ U.T
    t_align = mu_t - R_align @ mu_s
    return R_align, t_align


def interpolate_positions(gt: Trajectory, t_query: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.interp(t_query, gt.t, gt.p[:, 0]),
        np.interp(t_query, gt.t, gt.p[:, 1]),
        np.interp(t_query, gt.t, gt.p[:, 2]),
    ])


def estimate_umeyama_rotation_from_overlap(gt: Trajectory, est: Trajectory) -> np.ndarray:
    mask = (est.t >= gt.t[0]) & (est.t <= gt.t[-1])
    if np.count_nonzero(mask) < 10:
        return np.eye(3)
    est_p = est.p[mask]
    gt_p = interpolate_positions(gt, est.t[mask])
    R_align, _ = se3_umeyama_no_scale(est_p, gt_p)
    return R_align


def classify_benchmark_estimator(metrics: Dict[str, float], label: str) -> str:
    label_u = label.upper()
    if "MUSE" in label_u:
        return "MUSE"
    if "IEKF" in label_u:
        return "IEKF"
    if label_u in {"IS", "INVARIANT_SMOOTHER", "INVARIANT SMOOTHER"} or "SMOOTHER" in label_u:
        return "IS"

    # If no explicit label was given, infer from ATE closest to reported results.
    ate = metrics.get("ATE", math.inf)
    return min(BENCHMARK_RESULTS.keys(), key=lambda k: abs(BENCHMARK_RESULTS[k]["ATE"] - ate))


def compute_velocity_rmse(
    gt: Trajectory,
    est: Trajectory,
    metrics: Dict[str, float],
    label: str,
    max_gt_gap: float = 0.1,
) -> Tuple[float, np.ndarray, str, str]:
    """
    Time-correct ATEvel, computed identically for every estimator format:
    interpolate GT velocity onto the estimator timestamps over the overlap,
    rotate estimator velocities into the GT frame with an Umeyama rotation
    fitted from position alignment, and take the RMSE over all pairs.

    For benchmark-format files, the official (index-paired) protocol value is
    also printed, so the published Table I numbers remain reproducible.
    """
    if not np.any(np.linalg.norm(est.v, axis=1) > 1e-9):
        print("[WARN] Estimator file carries no velocity data; ATEvel is meaningless.")

    R_vel = estimate_umeyama_rotation_from_overlap(gt, est)
    rotation_mode = "Umeyama rotation from position alignment (symmetric protocol)"

    try:
        t, gt_v, est_v, n_excluded = time_correct_velocity_pairs(gt, est, max_gt_gap)
    except RuntimeError as exc:
        print(f"[WARN] ATEvel unavailable: {exc}")
        return float("nan"), R_vel, "unavailable (no time overlap)", rotation_mode

    atevel = rmse_vector(gt_v, (R_vel @ est_v.T).T)

    align_mode = (
        f"time-correct interpolation ({len(t)} pairs, "
        f"[{t[0] - est.t[0]:.1f}s, {t[-1] - est.t[0]:.1f}s] of the run"
    )
    if n_excluded:
        align_mode += f", {n_excluded} samples excluded at GT gaps > {max_gt_gap:g}s"
    align_mode += ")"

    if est.fmt == "benchmark":
        official, detected = official_benchmark_atevel(gt, est, metrics, label)
        print(
            f"[INFO] Official-protocol replication ({detected}): "
            f"ATEvel = {official:.6f} m/s (row-index pairing + fixed rotation).\n"
            f"       Reported only to confirm reproduction of the published table; "
            f"the comparison table uses the time-correct value {atevel:.6f} m/s."
        )

    return atevel, R_vel, align_mode, rotation_mode


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def rpy_from_quat_xyzw(q_xyzw: np.ndarray, benchmark_frame_fix: bool, unwrap: bool) -> np.ndarray:
    rot_abs = R.from_quat(q_xyzw)
    rot_rel = rot_abs[0].inv() * rot_abs
    if benchmark_frame_fix:
        frame_fix = R.from_euler("x", np.pi)
        rot_rel = frame_fix * rot_rel * frame_fix.inv()
    rpy_rad = rot_rel.as_euler("xyz", degrees=False)
    if unwrap:
        rpy_rad = np.unwrap(rpy_rad, axis=0)
    return np.rad2deg(rpy_rad)


def interpolate_gt_for_plot(gt: Trajectory, est: Trajectory) -> Trajectory:
    mask = (est.t >= gt.t[0]) & (est.t <= gt.t[-1])
    t = est.t[mask]
    if len(t) < 2:
        raise RuntimeError("Not enough overlapping timestamps for plotting.")

    p = interpolate_positions(gt, t)
    v = np.column_stack([
        np.interp(t, gt.t, gt.v[:, 0]),
        np.interp(t, gt.t, gt.v[:, 1]),
        np.interp(t, gt.t, gt.v[:, 2]),
    ])

    slerp = Slerp(gt.t, R.from_quat(gt.q_xyzw))
    q = slerp(t).as_quat()

    return Trajectory("GT_sync", "groundtruth", t, p, q, v)


def aligned_est_for_plot(gt_sync: Trajectory, est: Trajectory, R_pose: np.ndarray, t_pose: np.ndarray) -> Trajectory:
    mask = (est.t >= gt_sync.t[0]) & (est.t <= gt_sync.t[-1])
    t = est.t[mask]
    p = (R_pose @ est.p[mask].T).T + t_pose
    q = est.q_xyzw[mask]
    v = est.v[mask]
    return Trajectory(est.name + "_plot", est.fmt, t, p, q, v)


def make_plot_trajectories(gt: Trajectory, est: Trajectory, R_vel: np.ndarray) -> Tuple[Trajectory, Trajectory, Trajectory]:
    gt_sync = interpolate_gt_for_plot(gt, est)

    mask = (est.t >= gt_sync.t[0]) & (est.t <= gt_sync.t[-1])
    est_overlap = Trajectory(est.name, est.fmt, est.t[mask], est.p[mask], est.q_xyzw[mask], est.v[mask])

    R_pose, t_pose = se3_umeyama_no_scale(est_overlap.p, gt_sync.p)
    est_pose = aligned_est_for_plot(gt_sync, est_overlap, R_pose, t_pose)

    # Rebase the pose plot so both curves start at the origin.
    #
    # Previously both were shifted by the same GT start point.  The Umeyama
    # alignment is a least-squares fit over the WHOLE overlap, so it leaves a
    # residual at t=0: the estimate started a few tens of centimetres away
    # from GT and the two curves visibly took off from different points.
    # gt_sync and est_pose share a time grid by construction (gt_sync is GT
    # interpolated onto the estimator timestamps), so index 0 is the same
    # instant for both and subtracting each curve's own first sample is a
    # like-for-like shift.  This is cosmetic: ATE and the other metrics are
    # computed elsewhere on unshifted data.  Set REBASE_PLOT_TO_ORIGIN = False
    # to go back to showing the alignment residual at the start.
    gt_p0 = gt_sync.p[0].copy()
    est_p0 = est_pose.p[0].copy() if REBASE_PLOT_TO_ORIGIN else gt_p0
    gt_plot = Trajectory("GT", gt_sync.fmt, gt_sync.t - gt_sync.t[0], gt_sync.p - gt_p0, gt_sync.q_xyzw, gt_sync.v)
    est_pose_plot = Trajectory(est.name, est.fmt, est_pose.t - gt_sync.t[0], est_pose.p - est_p0, est_pose.q_xyzw, est_pose.v)

    # Velocity plot uses the velocity-frame rotation used in ATEvel computation.
    est_vel = Trajectory(est.name, est_overlap.fmt, est_overlap.t - gt_sync.t[0], est_overlap.p, est_overlap.q_xyzw, (R_vel @ est_overlap.v.T).T)

    return gt_plot, est_pose_plot, est_vel


def metrics_title(metrics: Dict[str, float]) -> str:
    title = f"ATE={metrics['ATE']:.3f} m | ATEvel={metrics['ATEvel']:.3f} m/s"
    rpe_keys = ["RPE_1m_trans", "RPE_1m_rot", "RPE_1f_trans", "RPE_1f_rot"]
    if all(k in metrics for k in rpe_keys):
        title += (
            f" | RPE Δ=1m: {metrics['RPE_1m_trans']:.4f} m, {metrics['RPE_1m_rot']:.4f}°"
            f" | RPE Δ=1fr: {metrics['RPE_1f_trans']:.6f} m, {metrics['RPE_1f_rot']:.6f}°"
        )
    else:
        title += " | debug-fast: RPE skipped"
    return title


def plot_pose(gt_plot: Trajectory, est_plot: Trajectory, metrics: Dict[str, float], unwrap: bool) -> None:
    gt_rpy = rpy_from_quat_xyzw(gt_plot.q_xyzw, benchmark_frame_fix=True, unwrap=unwrap)
    est_rpy = rpy_from_quat_xyzw(est_plot.q_xyzw, benchmark_frame_fix=False, unwrap=unwrap)

    fig = plt.figure(figsize=(15, 7))
    _figure_titles(fig, POSE_FIGURE_TITLE, metrics_title(metrics))
    gs = fig.add_gridspec(3, 3)

    ax_traj = fig.add_subplot(gs[:, 0])
    gt_line = _draw(ax_traj, gt_plot.p[:, 0], gt_plot.p[:, 1], GROUND_TRUTH_STYLE, "Ground truth")
    est_line = _draw(ax_traj, est_plot.p[:, 0], est_plot.p[:, 1], ESTIMATE_STYLE, est_plot.name)
    _style_axis(ax_traj, title=TRAJECTORY_PANEL_TITLE)
    ax_traj.set_xlabel(POS_LABELS[0], fontsize=_fs(AXIS_LABEL_FONT_SIZE, fig))
    ax_traj.set_ylabel(POS_LABELS[1], fontsize=_fs(AXIS_LABEL_FONT_SIZE, fig))
    ax_traj.axis("equal")

    for i in range(3):
        ax = fig.add_subplot(gs[i, 1])
        _draw(ax, gt_plot.t, gt_plot.p[:, i], GROUND_TRUTH_STYLE, "Ground truth")
        _draw(ax, est_plot.t, est_plot.p[:, i], ESTIMATE_STYLE, est_plot.name)
        _style_axis(ax, ylabel=POS_LABELS[i], title=POSITION_PANEL_TITLE if i == 0 else None)
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time [s]",
                          fontsize=_fs(AXIS_LABEL_FONT_SIZE, fig))

    for i in range(3):
        ax = fig.add_subplot(gs[i, 2])
        _draw(ax, gt_plot.t, gt_rpy[:, i], GROUND_TRUTH_STYLE, "Ground truth")
        _draw(ax, est_plot.t, est_rpy[:, i], ESTIMATE_STYLE, est_plot.name)
        _style_axis(ax, ylabel=ORI_LABELS[i], title=ORIENTATION_PANEL_TITLE if i == 0 else None)
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time [s]",
                          fontsize=_fs(AXIS_LABEL_FONT_SIZE, fig))

    # One legend for the whole figure instead of three in-panel ones.
    _bottom_legend(fig, [gt_line, est_line], ["Ground truth", est_plot.name],
                   has_subtitle=True)
    return fig


def plot_velocity(gt_plot: Trajectory, est_vel_plot: Trajectory, metrics: Dict[str, float]) -> None:
    # Taller than the pose figure: the ${}^{w}v^{x}_{b}$ [m/s] labels are
    # rotated and span more than a 7-inch-high panel row, and tight_layout
    # does not account for ylabel-to-ylabel collisions between rows.
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    _figure_titles(fig, VELOCITY_FIGURE_TITLE, metrics_title(metrics))

    handles: List = []
    for i, ax in enumerate(axes):
        gt_line = _draw(ax, gt_plot.t, gt_plot.v[:, i], GROUND_TRUTH_STYLE, "Ground truth")
        est_line = _draw(ax, est_vel_plot.t, est_vel_plot.v[:, i], ESTIMATE_STYLE, est_vel_plot.name)
        _style_axis(ax, ylabel=VEL_LABELS[i], title=VELOCITY_PANEL_TITLE if i == 0 else None)
        if not handles:
            handles = [gt_line, est_line]

    axes[-1].set_xlabel("Time [s]",
                        fontsize=_fs(AXIS_LABEL_FONT_SIZE, fig))
    _bottom_legend(fig, handles, ["Ground truth", est_vel_plot.name],
                   has_subtitle=True)
    return fig


# -----------------------------------------------------------------------------
# Terminal output
# -----------------------------------------------------------------------------
def green_if_best(text: str, value: float, best_value: float) -> str:
    green = "\033[92m"
    end = "\033[0m"
    if np.isclose(value, best_value, rtol=1e-12, atol=1e-12):
        return f"{green}{text}{end}"
    return text


def print_comparison_table(
    metrics: Dict[str, float],
    label: str,
    benchmark: Optional[Dict[str, Dict[str, float]]] = None,
    atevel_like_for_like: bool = False,
) -> None:
    benchmark = benchmark or BENCHMARK_RESULTS
    methods = ["MUSE", "IEKF", "IS", label]
    atevel_row_label = "ATEvel [m/s]" if atevel_like_for_like else "ATEvel [m/s] (*)"
    rows = [
        ("ATE [m]", "ATE"),
        (atevel_row_label, "ATEvel"),
        ("RPE (Δ = 1 meter) [m]", "RPE_1m_trans"),
        ("RPE (Δ = 1 frame) [m]", "RPE_1f_trans"),
        ("RPE (Δ = 1 meter) [°]", "RPE_1m_rot"),
        ("RPE (Δ = 1 frame) [°]", "RPE_1f_rot"),
    ]

    label_w = 28
    value_w = 12
    total_w = label_w + (value_w + 1) * len(methods)

    print("\n" + "=" * total_w)
    print("Benchmark comparison table")
    print("=" * total_w)
    header = f"{'RMSE':<{label_w}}" + "".join(f" {m:>{value_w}}" for m in methods)
    print(header)
    print("-" * total_w)

    for row_label, key in rows:
        values = {
            "MUSE": benchmark["MUSE"][key],
            "IEKF": benchmark["IEKF"][key],
            "IS": benchmark["IS"][key],
            label: metrics[key],
        }
        finite = [v for v in values.values() if np.isfinite(v)]
        best = min(finite) if finite else float("nan")
        line = f"{row_label:<{label_w}}"
        for m in methods:
            plain = f"{values[m]:.6f}"
            padded = f"{plain:>{value_w}}"
            line += " " + green_if_best(padded, values[m], best)
        print(line)
    print("=" * total_w)
    if not atevel_like_for_like:
        print(f"(*) The MUSE/IEKF/IS ATEvel values are the published (index-paired)")
        print(f"    protocol, which is dominated by GT/estimator time misassociation;")
        print(f"    the {label} value is time-correct, so this row is NOT comparable.")
        print(f"    For a like-for-like row, run this script on each benchmark")
        print(f"    fused_state.csv and pass the printed time-correct values via")
        print(f"    --benchmark-atevel 'MUSE=...,IEKF=...,IS=...'.")
    print()


def print_fast_comparison_table(
    metrics: Dict[str, float],
    label: str,
    benchmark: Optional[Dict[str, Dict[str, float]]] = None,
    atevel_like_for_like: bool = False,
) -> None:
    benchmark = benchmark or BENCHMARK_RESULTS
    methods = ["MUSE", "IEKF", "IS", label]
    atevel_row_label = "ATEvel [m/s]" if atevel_like_for_like else "ATEvel [m/s] (*)"
    rows = [
        ("ATE [m]", "ATE"),
        (atevel_row_label, "ATEvel"),
    ]

    label_w = 28
    value_w = 12
    total_w = label_w + (value_w + 1) * len(methods)

    print("\n" + "=" * total_w)
    print("Fast debug comparison table (RPE skipped)")
    print("=" * total_w)
    header = f"{'RMSE':<{label_w}}" + "".join(f" {m:>{value_w}}" for m in methods)
    print(header)
    print("-" * total_w)

    for row_label, key in rows:
        values = {
            "MUSE": benchmark["MUSE"][key],
            "IEKF": benchmark["IEKF"][key],
            "IS": benchmark["IS"][key],
            label: metrics[key],
        }
        finite = [v for v in values.values() if np.isfinite(v)]
        best = min(finite) if finite else float("nan")
        line = f"{row_label:<{label_w}}"
        for m in methods:
            plain = f"{values[m]:.6f}"
            padded = f"{plain:>{value_w}}"
            line += " " + green_if_best(padded, values[m], best)
        print(line)
    print("=" * total_w)
    if not atevel_like_for_like:
        print(f"(*) Benchmark ATEvel = published index-paired protocol; {label} = ")
        print(f"    time-correct. Not comparable; see --benchmark-atevel.")
    print()


def print_diagnostics(gt: Trajectory, est: Trajectory, gt_path: Path, est_path: Path, workdir: Path, velocity_mode: str, velocity_align_mode: str) -> None:
    print("================ Inputs ================")
    print(f"GT path:        {gt_path}")
    print(f"Estimator path: {est_path}")
    print(f"Estimator fmt:  {est.fmt}")
    print(f"Workdir:        {workdir}")
    print("----------------------------------------")
    print(f"GT samples:     {len(gt.t)}")
    print(f"EST samples:    {len(est.t)}")
    print(f"GT time:        {gt.t[0]:.9f} -> {gt.t[-1]:.9f}")
    print(f"EST time:       {est.t[0]:.9f} -> {est.t[-1]:.9f}")
    print(f"Velocity mode:  {velocity_mode}")
    print(f"Vel align by:   {velocity_align_mode}")
    print("========================================")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--est", type=Path, default=DEFAULT_EST)
    parser.add_argument("--sensor", type=Path, default=DEFAULT_SENSOR)
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--label", default="SBEE")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help=(
            "Directory for automatically saved figures "
            f"(default: {DEFAULT_PLOT_DIR}). Created if missing."
        ),
    )
    parser.add_argument(
        "--plot-formats",
        nargs="+",
        choices=("pdf", "svg", "png"),
        default=list(DEFAULT_PLOT_FORMATS),
        help="Formats to save each figure in (default: pdf).",
    )
    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        help="Display the figures without writing any files.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figures without opening GUI windows (for headless runs).",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=300,
        help="Resolution for PNG output (default: 300).",
    )
    parser.add_argument("--no-unwrap", action="store_true")
    parser.add_argument(
        "--debug-fast",
        action="store_true",
        help="Compute only ATE position and ATE velocity. Skip all RPE evo_rpe metrics.",
    )
    parser.add_argument(
        "--max-gt-gap",
        type=float,
        default=0.1,
        help="Maximum distance [s] to the nearest GT sample for a velocity pair "
             "to be used in ATEvel (guards against interpolating across GT "
             "dropouts). Default: 0.1 s.",
    )
    parser.add_argument(
        "--benchmark-atevel",
        default=None,
        help="Time-correct ATEvel values for the benchmark methods, e.g. "
             "'MUSE=0.152,IEKF=0.141,IS=0.138'. Obtain each by running this "
             "script with --est pointing at that method's fused_state.csv and "
             "reading the printed time-correct ATEvel. When given, the table's "
             "ATEvel row is like-for-like and the footnote is dropped.",
    )
    args = parser.parse_args()

    if shutil.which("evo_ape") is None:
        raise RuntimeError(
            "evo_ape not found. Activate the benchmark environment first."
        )
    if not args.debug_fast and shutil.which("evo_rpe") is None:
        raise RuntimeError(
            "evo_rpe not found. Activate the benchmark environment first, or use --debug-fast."
        )

    dataset_root = args.dataset_root

    gt_path = resolve_path(
        dataset_root,
        args.gt,
        fallback_relative=Path("anymal_data/test/cyn-1/groundtruth.csv"),
    )

    est_path = resolve_path(
        dataset_root,
        args.est,
        fallback_relative=Path("anymal_data/test/cyn-1/serow/fused_state.csv"),
    )

    sensor_path = resolve_path(
        dataset_root,
        args.sensor,
        fallback_relative=Path("anymal_data/test/cyn-1/anymal_data.csv"),
    )

    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    gt = load_groundtruth(gt_path)
    est = load_estimator(est_path, label=args.label)

    gt_tum = workdir / "groundtruth_traj_tum.csv"
    est_tum = workdir / f"{args.label.lower()}_traj_tum.csv"

    write_tum(gt, gt_tum)
    write_tum(est, est_tum)

    if args.debug_fast:
        metrics = compute_evo_ate_only(gt_tum, est_tum)
    else:
        metrics = compute_evo_metrics(gt_tum, est_tum)

    atevel, R_vel, vel_align_mode, vel_rotation_mode = compute_velocity_rmse(
        gt,
        est,
        metrics,
        args.label,
        max_gt_gap=args.max_gt_gap,
    )
    metrics["ATEvel"] = atevel

    # Optionally replace the published (index-paired) benchmark ATEvel values
    # with user-supplied time-correct ones so the table row is like-for-like.
    benchmark = {k: dict(v) for k, v in BENCHMARK_RESULTS.items()}
    atevel_like_for_like = False
    if args.benchmark_atevel:
        for item in args.benchmark_atevel.split(","):
            name, _, val = item.partition("=")
            name = name.strip().upper()
            if name not in benchmark or not val.strip():
                raise ValueError(
                    f"--benchmark-atevel entry not understood: '{item}' "
                    "(expected e.g. 'MUSE=0.152,IEKF=0.141,IS=0.138')"
                )
            benchmark[name]["ATEvel"] = float(val)
        atevel_like_for_like = True

    print_diagnostics(
        gt,
        est,
        gt_path,
        est_path,
        workdir,
        vel_rotation_mode,
        vel_align_mode,
    )

    if args.debug_fast:
        print_fast_comparison_table(metrics, args.label, benchmark, atevel_like_for_like)
    else:
        print_comparison_table(metrics, args.label, benchmark, atevel_like_for_like)

    if not args.no_plots:
        gt_plot, est_pose_plot, est_vel_plot = make_plot_trajectories(gt, est, R_vel)

        figures = [
            ("pose", plot_pose(gt_plot, est_pose_plot, metrics, unwrap=not args.no_unwrap)),
            ("velocity", plot_velocity(gt_plot, est_vel_plot, metrics)),
            ("imu_biases", plot_imu_biases(est_path)),
            ("contact_probabilities", plot_contact_probabilities_one_to_one(
                est_path, sensor_path, "Estimated probability")),
            ("pseudo_forces", plot_pseudo_forces_vs_contacts(est_path, sensor_path)),
        ]

        if not args.no_save_plots:
            plot_dir = args.plot_dir.expanduser().resolve()
            # Filenames carry the estimator label so runs of different
            # estimators into the same directory do not overwrite each other.
            prefix = args.label.lower().replace(" ", "_")
            for stem, figure in figures:
                save_figure(
                    figure,
                    plot_dir,
                    f"{prefix}_{stem}",
                    tuple(args.plot_formats),
                    args.plot_dpi,
                )

        if args.no_show:
            plt.close("all")
        else:
            plt.show()


if __name__ == "__main__":
    main()
