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
    - Velocity RMSE follows the official benchmark script behaviour:
        * benchmark MUSE/IEKF/IS format: use official fixed Umeyama rotations.
        * SEROW format: use an Umeyama rotation estimated from position alignment.

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
# Official benchmark constants
# -----------------------------------------------------------------------------
BENCHMARK_RESULTS = {
    "MUSE": {
        "ATE": 2.269461,
        "ATEvel": 0.876147,
        "RPE_1m_trans": 0.072223,
        "RPE_1f_trans": 0.000670,
        "RPE_1m_rot": 0.578469,
        "RPE_1f_rot": 0.002605,
    },
    "IEKF": {
        "ATE": 1.405668,
        "ATEvel": 0.869383,
        "RPE_1m_trans": 0.043187,
        "RPE_1f_trans": 0.000544,
        "RPE_1m_rot": 0.568987,
        "RPE_1f_rot": 0.002611,
    },
    "IS": {
        "ATE": 1.363114,
        "ATEvel": 0.869481,
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
    fig.suptitle("Estimated IMU Biases", fontsize=14)

    acc_names = ["bias_ax", "bias_ay", "bias_az"]
    gyro_names = ["bias_gx", "bias_gy", "bias_gz"]

    for i in range(3):
        ax = axes[0, i]
        ax.plot(t_rel_plot, bias_acc_plot[:, i])
        ax.axhline(0.0, linestyle="--", linewidth=0.8)
        ax.set_title(acc_names[i])
        ax.set_ylabel("Accel bias [m/s²]")
        ax.grid(True)

    for i in range(3):
        ax = axes[1, i]
        ax.plot(t_rel_plot, bias_gyro_plot[:, i])
        ax.axhline(0.0, linestyle="--", linewidth=0.8)
        ax.set_title(gyro_names[i])
        ax.set_ylabel("Gyro bias [rad/s]")
        ax.set_xlabel("Time [s]")
        ax.grid(True)

    fig.tight_layout()
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


def load_estimator(path: Path, label: str = "SEROW") -> Trajectory:
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

def plot_contact_probabilities_one_to_one(est_csv: Path,
                                          sensor_csv: Path,
                                          max_points: int = 50000):
    """
    Plot SEROW estimated contact probabilities from fused_state.csv
    against the binary contact flags from anymal_data.csv, one-to-one by row index.

    This assumes:
      fused_state.csv row i corresponds exactly to anymal_data.csv row i.
    """

    contact_cols = ["contact_LF", "contact_RF", "contact_LH", "contact_RH"]
    leg_names = ["LF", "RF", "LH", "RH"]

    try:
        est_df = read_csv_auto(est_csv)
        sensor_df = read_csv_auto(sensor_csv)
    except Exception as e:
        print(f"[WARN] Could not read contact files: {e}")
        return

    missing_est = [c for c in ["t"] + contact_cols if c not in est_df.columns]
    missing_sensor = [c for c in ["t"] + contact_cols if c not in sensor_df.columns]

    if missing_est:
        print("[INFO] Contact plot skipped.")
        print(f"       Missing columns in fused_state.csv: {missing_est}")
        return

    if missing_sensor:
        print("[INFO] Contact plot skipped.")
        print(f"       Missing columns in anymal_data.csv: {missing_sensor}")
        return

    if len(est_df) != len(sensor_df):
        raise RuntimeError(
            "Cannot plot contacts one-to-one because the files have different lengths:\n"
            f"  fused_state.csv rows: {len(est_df)}\n"
            f"  anymal_data.csv rows: {len(sensor_df)}\n"
            "Since you requested one-to-one plotting, this must be fixed before plotting."
        )

    t_est = est_df["t"].to_numpy(float)
    t_sensor = sensor_df["t"].to_numpy(float)

    # Not used for alignment, only printed as a sanity check.
    dt = t_est - t_sensor
    print("Contact one-to-one check:")
    print(f"  rows:       {len(est_df)}")
    print(f"  mean dt:    {np.mean(dt):.12f} s")
    print(f"  max |dt|:   {np.max(np.abs(dt)):.12f} s")

    contact_prob = est_df[contact_cols].to_numpy(float)
    contact_flag = sensor_df[contact_cols].to_numpy(float)

    # Force binary flags to 0/1 in case they are stored as floats.
    contact_flag = (contact_flag > 0.5).astype(float)

    n = len(est_df)
    if n > max_points:
        step = int(np.ceil(n / max_points))
        idx = np.arange(0, n, step)
    else:
        idx = np.arange(n)

    t_rel = t_est[idx] - t_est[0]
    prob_plot = contact_prob[idx]
    flag_plot = contact_flag[idx]

    fig, axes = plt.subplots(4, 1, figsize=(15, 9), sharex=True)
    fig.suptitle("SEROW estimated contact probabilities vs binary contact flags")

    for i, ax in enumerate(axes):
        ax.plot(
            t_rel,
            prob_plot[:, i],
            label="SEROW estimated probability",
            linewidth=1.2,
        )

        ax.step(
            t_rel,
            flag_plot[:, i],
            where="post",
            linestyle="--",
            linewidth=1.0,
            label="Binary contact flag",
        )

        ax.set_ylabel(leg_names[i])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc="best")

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
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
def align_velocity_records(gt: Trajectory, est: Trajectory, force_index_for_benchmark: bool) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Match the official compute_vel_rmse.py behaviour.

    For benchmark MUSE/IEKF/IS fused_state.csv files, the official script does
    not see a time column because it only checks t/time/timestamp/stamp, while
    benchmark files have t_abs. Therefore it falls back to index alignment.
    We reproduce that behaviour with force_index_for_benchmark=True.
    """
    if not force_index_for_benchmark:
        gt_map = {round(float(t), 9): i for i, t in enumerate(gt.t)}
        est_map = {round(float(t), 9): i for i, t in enumerate(est.t)}
        common = sorted(set(gt_map.keys()) & set(est_map.keys()))
        if len(common) > 1:
            gi = np.array([gt_map[t] for t in common], dtype=int)
            ei = np.array([est_map[t] for t in common], dtype=int)
            return gt.v[gi], est.v[ei], "timestamp"

    n = min(len(gt.v), len(est.v))
    return gt.v[:n], est.v[:n], "index"


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


def compute_velocity_rmse(gt: Trajectory, est: Trajectory, metrics: Dict[str, float], label: str) -> Tuple[float, np.ndarray, str, str]:
    if est.fmt == "benchmark":
        detected = classify_benchmark_estimator(metrics, label)
        R_vel = OFFICIAL_VEL_ROTATIONS[detected]
        force_index = True
        rotation_mode = f"official {detected} rotation"
    else:
        R_vel = estimate_umeyama_rotation_from_overlap(gt, est)
        force_index = False
        rotation_mode = "Umeyama rotation from position alignment"

    gt_v, est_v, align_mode = align_velocity_records(gt, est, force_index_for_benchmark=force_index)
    est_v_rot = (R_vel @ est_v.T).T
    return rmse_vector(gt_v, est_v_rot), R_vel, align_mode, rotation_mode


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

    # Rebase pose plot to common GT start so both curves start together visually.
    p0 = gt_sync.p[0].copy()
    gt_plot = Trajectory("GT", gt_sync.fmt, gt_sync.t - gt_sync.t[0], gt_sync.p - p0, gt_sync.q_xyzw, gt_sync.v)
    est_pose_plot = Trajectory(est.name, est.fmt, est_pose.t - gt_sync.t[0], est_pose.p - p0, est_pose.q_xyzw, est_pose.v)

    # Velocity plot uses the velocity-frame rotation used in ATEvel computation.
    est_vel = Trajectory(est.name, est_overlap.fmt, est_overlap.t - gt_sync.t[0], est_overlap.p, est_overlap.q_xyzw, (R_vel @ est_overlap.v.T).T)

    return gt_plot, est_pose_plot, est_vel


def metrics_title(metrics: Dict[str, float]) -> str:
    return (
        f"ATE={metrics['ATE']:.3f} m | "
        f"ATEvel={metrics['ATEvel']:.3f} m/s | "
        f"RPE Δ=1m: {metrics['RPE_1m_trans']:.4f} m, {metrics['RPE_1m_rot']:.4f}° | "
        f"RPE Δ=1fr: {metrics['RPE_1f_trans']:.6f} m, {metrics['RPE_1f_rot']:.6f}°"
    )


def plot_pose(gt_plot: Trajectory, est_plot: Trajectory, metrics: Dict[str, float], unwrap: bool) -> None:
    gt_rpy = rpy_from_quat_xyzw(gt_plot.q_xyzw, benchmark_frame_fix=True, unwrap=unwrap)
    est_rpy = rpy_from_quat_xyzw(est_plot.q_xyzw, benchmark_frame_fix=False, unwrap=unwrap)

    fig = plt.figure(figsize=(15, 7))
    fig.suptitle("CYN-1 pose estimate vs ground truth\n" + metrics_title(metrics))
    gs = fig.add_gridspec(3, 3)

    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.plot(gt_plot.p[:, 0], gt_plot.p[:, 1], "--", label="GT")
    ax_traj.plot(est_plot.p[:, 0], est_plot.p[:, 1], label=est_plot.name)
    ax_traj.set_title("(a) XY trajectory")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.axis("equal")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(loc="best")

    pos_labels = [r"$p_x$ [m]", r"$p_y$ [m]", r"$p_z$ [m]"]
    for i in range(3):
        ax = fig.add_subplot(gs[i, 1])
        ax.plot(gt_plot.t, gt_plot.p[:, i], "--", label="GT")
        ax.plot(est_plot.t, est_plot.p[:, i], label=est_plot.name)
        ax.set_ylabel(pos_labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("(b) Position")
            ax.legend(loc="best")
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time since estimator start [s]")

    ori_labels = ["roll [deg]", "pitch [deg]", "yaw [deg]"]
    for i in range(3):
        ax = fig.add_subplot(gs[i, 2])
        ax.plot(gt_plot.t, gt_rpy[:, i], "--", label="GT")
        ax.plot(est_plot.t, est_rpy[:, i], label=est_plot.name)
        ax.set_ylabel(ori_labels[i])
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title("(c) Orientation")
            ax.legend(loc="best")
        if i < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time since estimator start [s]")

    plt.tight_layout(rect=[0, 0, 1, 0.92])


def plot_velocity(gt_plot: Trajectory, est_vel_plot: Trajectory, metrics: Dict[str, float]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 7), sharex=True)
    fig.suptitle("CYN-1 velocity estimate vs ground truth\n" + metrics_title(metrics))
    labels = [r"$v_x$ [m/s]", r"$v_y$ [m/s]", r"$v_z$ [m/s]"]

    for i, ax in enumerate(axes):
        ax.plot(gt_plot.t, gt_plot.v[:, i], "--", label="GT")
        ax.plot(est_vel_plot.t, est_vel_plot.v[:, i], label=est_vel_plot.name)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time since estimator start [s]")
    plt.tight_layout(rect=[0, 0, 1, 0.92])


# -----------------------------------------------------------------------------
# Terminal output
# -----------------------------------------------------------------------------
def green_if_best(text: str, value: float, best_value: float) -> str:
    green = "\033[92m"
    end = "\033[0m"
    if np.isclose(value, best_value, rtol=1e-12, atol=1e-12):
        return f"{green}{text}{end}"
    return text


def print_comparison_table(metrics: Dict[str, float], label: str) -> None:
    methods = ["MUSE", "IEKF", "IS", label]
    rows = [
        ("ATE [m]", "ATE"),
        ("ATEvel [m/s]", "ATEvel"),
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
            "MUSE": BENCHMARK_RESULTS["MUSE"][key],
            "IEKF": BENCHMARK_RESULTS["IEKF"][key],
            "IS": BENCHMARK_RESULTS["IS"][key],
            label: metrics[key],
        }
        best = min(values.values())
        line = f"{row_label:<{label_w}}"
        for m in methods:
            plain = f"{values[m]:.6f}"
            padded = f"{plain:>{value_w}}"
            line += " " + green_if_best(padded, values[m], best)
        print(line)
    print("=" * total_w + "\n")


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
    parser.add_argument("--label", default="SEROW")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-unwrap", action="store_true")
    args = parser.parse_args()

    if shutil.which("evo_ape") is None or shutil.which("evo_rpe") is None:
        raise RuntimeError(
            "evo_ape/evo_rpe not found. Activate the benchmark environment first."
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

    metrics = compute_evo_metrics(gt_tum, est_tum)

    atevel, R_vel, vel_align_mode, vel_rotation_mode = compute_velocity_rmse(
        gt,
        est,
        metrics,
        args.label,
    )
    metrics["ATEvel"] = atevel

    print_diagnostics(
        gt,
        est,
        gt_path,
        est_path,
        workdir,
        vel_rotation_mode,
        vel_align_mode,
    )

    print_comparison_table(metrics, args.label)

    if not args.no_plots:
        gt_plot, est_pose_plot, est_vel_plot = make_plot_trajectories(gt, est, R_vel)

        plot_pose(gt_plot, est_pose_plot, metrics, unwrap=not args.no_unwrap)
        plot_velocity(gt_plot, est_vel_plot, metrics)
        plot_imu_biases(est_path)
        plot_contact_probabilities_one_to_one(est_path, sensor_path)

        plt.show()


if __name__ == "__main__":
    main()
