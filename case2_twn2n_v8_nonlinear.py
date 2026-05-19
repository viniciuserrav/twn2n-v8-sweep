"""Class 2 — nonlinear pendulum sweep with LINEAR (angular) sensors (v8 focal).

Same physics as Class 3 (case3_twn2n_v8_nonlinear.py): single or double pendulum
with sin(theta) gravity restoring torque and tangential viscous damping; three
excitation regimes (a) free vibration, (b) one known shaker, (c) random
unmeasured forces.  The only difference is the measurement model: instead of
the Cartesian projection of the bobs (which is a nonlinear map of the state),
the denoiser observes the angular state variables directly: theta and
theta_dot for each pendulum.  The measurement map is therefore linear (it just
selects rows of the state vector), making this the genuine Class~2 setting
(nonlinear dynamics + linear measurement) of the framework in Section~3.

Channel pool per model:
  * n_models = 1 (single pendulum) : (theta, theta_dot)            -> max n_z = 2
  * n_models = 2 (double pendulum) : (theta_1, theta_dot_1,
                                        theta_2, theta_dot_2)      -> max n_z = 4
Order is round-robin per pendulum so the focal channel is always theta_1.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
# Re-use TWN2N training / SNR scaling / gain functions from Class~1
from case1_twn2n_v8_anchor import (
    DT, N_S, R_OVERSAMPLE, NPERSEG,
    F_MIN_BRACKET, F_MAX_BRACKET, CHIRP_F0, CHIRP_F1,
    SNR_DB, P_SAMPLES, P_GRID, N_EXP_PER_CELL,
    LATENT_RHO, LEARNING_RATE, LOSS, VAL_FRACTION, PATIENCE, EPOCHS,
    BATCH_SIZE,
    denoise_anchor_multi, gain_db, add_noise_per_channel, ntfy,
)
# Re-use pendulum integrator + force generator from Class~3 (same physics)
from case3_twn2n_v8_nonlinear import (
    simulate_trajectory as _c3_simulate_trajectory,
    G as _G_DEFAULT, L_LINK as _L_DEFAULT, M_BOB as _M_DEFAULT,
    C_TANG as _C_DEFAULT, THETA0_MAX as _T0_DEFAULT,
    FORCE_AMP_B as _FB_DEFAULT, FORCE_AMP_C as _FC_DEFAULT,
)

# ============================================================================
# Configuration
# ============================================================================
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "vinicius-claude-alert")

OUT_DIR = HERE / "case2_v8_angular_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "case2_v8_angular_results.csv"
XLSX_PATH = OUT_DIR / "case2_v8_angular_results.xlsx"
LOG_PATH = OUT_DIR / "case2_v8_angular_run.log"

# Sweep grid
CASES = ("a", "b", "c")
# Seed offsets distinct from Class 1 and Class 3 so the random realisations
# of the three classes are statistically independent for the same seed index.
_CASE_SEED_OFFSET = {"a": 20, "b": 21, "c": 22}
N_MODELS = (1, 2)

# Physical parameters: inherited from Class 3 verbatim so the two pendulum
# studies share the same dynamics (only the sensor model differs).
G = _G_DEFAULT
L_LINK = _L_DEFAULT
M_BOB = _M_DEFAULT
C_TANG = _C_DEFAULT
THETA0_MAX = _T0_DEFAULT
FORCE_AMP_B = _FB_DEFAULT
FORCE_AMP_C = _FC_DEFAULT


def n_sensors_grid_case2(n_models: int) -> list[int]:
    """Channel-count grid: 2 angular sensors per pendulum (theta, theta_dot)."""
    return list(range(1, 2 * n_models + 1))


def simulate_trajectory(n_models: int, case: str,
                        rng: np.random.Generator,
                        n_steps: int = N_S, dt: float = DT) -> dict[str, np.ndarray]:
    """Same dynamics as Class 3 (re-uses its integrator).

    The Class-3 ``simulate_trajectory`` already returns ``theta`` of shape
    ``(n_models, n_steps)`` together with the Cartesian ``x`` and ``y``; we
    just ignore the Cartesian channels and compute ``theta_dot`` by central
    differences on ``theta``.
    """
    traj = _c3_simulate_trajectory(n_models, case, rng, n_steps=n_steps, dt=dt)
    theta = traj["theta"]
    if theta.ndim == 1:
        theta = theta[None, :]
    theta_dot = np.zeros_like(theta)
    # Centred finite difference, with one-sided differences at the boundaries.
    theta_dot[:, 1:-1] = (theta[:, 2:] - theta[:, :-2]) / (2 * dt)
    theta_dot[:, 0] = (theta[:, 1] - theta[:, 0]) / dt
    theta_dot[:, -1] = (theta[:, -1] - theta[:, -2]) / dt
    return {"theta": theta, "theta_dot": theta_dot}


def channel_order_for_case2(n_models: int) -> list[tuple[str, int]]:
    """Round-robin per pendulum: [(theta, 0), (theta_dot, 0), (theta, 1), ...]
    so the focal channel (index 0) is theta_1.
    """
    order = []
    for k in range(n_models):
        order.append(("theta", k))
        order.append(("theta_dot", k))
    return order


def select_channels_from_traj_case2(traj: dict[str, np.ndarray],
                                    channels_spec: list[tuple[str, int]]
                                    ) -> tuple[np.ndarray, list[str]]:
    rows, names = [], []
    for ctype, idx in channels_spec:
        if ctype == "theta":
            rows.append(traj["theta"][idx])
            names.append(f"theta_pend{idx}")
        elif ctype == "theta_dot":
            rows.append(traj["theta_dot"][idx])
            names.append(f"thetadot_pend{idx}")
        else:
            raise ValueError(f"Unknown channel type {ctype!r}")
    return np.vstack(rows), names


def run_cell(case: str, snr_db: float, n_models: int, n_z: int,
             p_samples: int, seed: int) -> dict[str, Any]:
    rng_sys = np.random.default_rng(
        seed * 100003 + 7 * n_models + _CASE_SEED_OFFSET[case])
    full_order = channel_order_for_case2(n_models)
    if n_z > len(full_order):
        raise ValueError(
            f"n_z={n_z} > max channels {len(full_order)} for case={case}, "
            f"n_models={n_models}")
    spec = full_order[:n_z]

    clean_list, noisy_list, ch_names = [], [], []
    for exp_id in range(N_EXP_PER_CELL):
        rng_traj = np.random.default_rng(
            seed * 100003 + 7 * n_models + 1000 + 11 * exp_id
            + _CASE_SEED_OFFSET[case])
        traj = simulate_trajectory(n_models, case, rng_traj)
        chs, names = select_channels_from_traj_case2(traj, spec)
        chs_noisy, _ = add_noise_per_channel(chs, snr_db, rng_traj)
        clean_list.append(chs)
        noisy_list.append(chs_noisy)
        if not ch_names:
            ch_names = names

    latent_dim = max(2, int(round(LATENT_RHO * 2 * n_models)))
    den_list, train_info = denoise_anchor_multi(
        noisy_list, p_samples, latent_dim, seed)

    per_traj: list[dict[str, float]] = []
    for noisy_mat, den_arr, clean_mat in zip(noisy_list, den_list, clean_list):
        per_traj.append(gain_db(noisy_mat[0], den_arr, clean_mat[0]))

    anchor_noisy_pool = np.concatenate([n[0] for n in noisy_list])
    anchor_clean_pool = np.concatenate([c[0] for c in clean_list])
    anchor_den_pool = np.concatenate(den_list)
    pooled = gain_db(anchor_noisy_pool, anchor_den_pool, anchor_clean_pool)

    f_max_sys = float(np.sqrt(G / L_LINK) / (2 * np.pi))
    f_min_sys = f_max_sys
    row: dict[str, Any] = {
        "case": case,
        "snr_db": snr_db,
        "n_modes": n_models,
        "n_z": int(n_z),
        "p_samples": int(p_samples),
        "P_grid_nominal": p_samples / R_OVERSAMPLE,
        "P_sys": float(p_samples * DT * f_max_sys),
        "latent_dim": int(latent_dim),
        "rho": float(latent_dim / (2 * n_models)),
        "f_max_sys": f_max_sys,
        "f_min_sys": f_min_sys,
        "anchor_name": ch_names[0],
        "all_channels": ";".join(ch_names),
        "seed": int(seed),
        **pooled,
        **train_info,
    }
    for k, m in enumerate(per_traj):
        row[f"G_NMSE_dB_traj{k}"] = m["G_NMSE_dB"]
        row[f"NMSE_in_traj{k}"] = m["NMSE_in"]
        row[f"NMSE_out_traj{k}"] = m["NMSE_out"]
    return row


def build_cells(smoke: bool = False,
                cases_filter: tuple = CASES,
                snr_filter: tuple = SNR_DB,
                n_models_filter: tuple = N_MODELS) -> list[tuple]:
    cells = []
    for case in cases_filter:
        for n_models in n_models_filter:
            for snr in snr_filter:
                for n_z in n_sensors_grid_case2(n_models):
                    for p in P_SAMPLES:
                        cells.append((case, snr, n_models, n_z, p))
    if smoke:
        cells = cells[:4]
    return cells


def load_already_done(csv_path: Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return set()
    out = set()
    for _, r in df.iterrows():
        out.add((str(r["case"]),
                 float(r["snr_db"]),
                 int(r["n_modes"]),
                 int(r["n_z"]),
                 int(r["p_samples"]),
                 int(r["seed"])))
    return out


def append_row(row: dict[str, Any]):
    write_header = not CSV_PATH.exists()
    df = pd.DataFrame([row])
    df.to_csv(CSV_PATH, mode="a", header=write_header, index=False)


def save_excel(rows: list[dict[str, Any]],
               xlsx_path: Path, run_config: dict[str, Any]):
    df = pd.DataFrame(rows) if rows else pd.read_csv(CSV_PATH)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as wr:
        df.to_excel(wr, sheet_name="results", index=False)
        if len(df):
            grp = df.groupby(["case", "n_modes", "snr_db", "n_z", "p_samples"],
                             as_index=False)["G_NMSE_dB"].agg(
                ["mean", "std", "count"]).reset_index()
            grp.to_excel(wr, sheet_name="agg_by_cell", index=False)
        meta = pd.DataFrame([{"key": k, "value": json.dumps(v, default=str)}
                             for k, v in run_config.items()])
        meta.to_excel(wr, sheet_name="run_config", index=False)


def _parse_csv_list(s, cast):
    return tuple(cast(x) for x in s.split(",") if x.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--snr-only", default=",".join(str(s) for s in SNR_DB))
    parser.add_argument("--n-modes-only",
                        default=",".join(str(n) for n in N_MODELS))
    args = parser.parse_args()

    cases_filter = tuple(c.strip() for c in args.cases.split(",") if c.strip())
    snr_filter = _parse_csv_list(args.snr_only, int)
    n_models_filter = _parse_csv_list(args.n_modes_only, int)

    cells = build_cells(args.smoke, cases_filter=cases_filter,
                        snr_filter=snr_filter, n_models_filter=n_models_filter)
    n_seeds = 1 if args.smoke else args.seeds
    total = len(cells) * n_seeds

    if args.restart and CSV_PATH.exists():
        CSV_PATH.unlink()
        print(f"--restart: deleted existing {CSV_PATH.name}", flush=True)
    already_done = load_already_done(CSV_PATH)

    run_config = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "ntfy_topic": NTFY_TOPIC,
        "variant": "class2-v8-nonlinear-angular (pendulum, theta+theta_dot sensors)",
        "smoke": bool(args.smoke),
        "restart": bool(args.restart),
        "n_seeds": n_seeds,
        "n_cells": len(cells),
        "n_trainings": total,
        "n_already_done": len(already_done),
        "n_exp_per_cell": N_EXP_PER_CELL,
        "dt": DT, "N_s": N_S,
        "f_min_bracket": F_MIN_BRACKET, "f_max_bracket": F_MAX_BRACKET,
        "R_oversample": R_OVERSAMPLE, "nperseg": NPERSEG,
        "CASES": CASES, "SNR_DB": SNR_DB, "N_MODELS": N_MODELS,
        "P_SAMPLES": P_SAMPLES,
        "n_sensors_per_n_models": {n: n_sensors_grid_case2(n) for n in N_MODELS},
        "EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE, "PATIENCE": PATIENCE,
        "LATENT_RHO": LATENT_RHO,
        "pendulum_params": {"G": G, "L_LINK": L_LINK, "M_BOB": M_BOB,
                            "C_TANG": C_TANG, "THETA0_MAX": THETA0_MAX,
                            "FORCE_AMP_B": FORCE_AMP_B,
                            "FORCE_AMP_C": FORCE_AMP_C},
        "out_dir": str(OUT_DIR),
    }
    print(json.dumps(run_config, indent=2, default=str), flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"== class2 v8 angular sweep started {run_config['started_at']}\n")

    t_start = time.time()
    rows: list[dict[str, Any]] = []
    n_skipped = 0
    for seed in range(args.seed_start, args.seed_start + n_seeds):
        for ci, (case, snr, n_models, n_z, p) in enumerate(cells, 1):
            key = (case, float(snr), int(n_models), int(n_z), int(p), int(seed))
            if key in already_done:
                n_skipped += 1
                continue
            t0 = time.time()
            try:
                row = run_cell(case, snr, n_models, n_z, p, seed)
            except Exception as e:
                tb = traceback.format_exc()
                with LOG_PATH.open("a", encoding="utf-8") as fh:
                    fh.write(f"ERROR seed={seed} cell={ci}/{len(cells)} "
                             f"case={case} snr={snr} N={n_models} "
                             f"n_z={n_z} p={p}: {e}\n{tb}\n")
                continue
            row["wall_s"] = time.time() - t0
            rows.append(row)
            append_row(row)
            print(f"[seed={seed} cell {ci}/{len(cells)}] case={case} "
                  f"snr={snr} N={n_models} n_z={n_z} p={p}  "
                  f"G_NMSE={row['G_NMSE_dB']:+.2f} dB  "
                  f"({row['wall_s']:.1f} s)", flush=True)

    elapsed = time.time() - t_start
    msg = (f"class2 v8 angular sweep DONE in {elapsed/60:.1f} min. "
           f"Wrote {len(rows)} new rows, skipped {n_skipped}.")
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")
    try:
        save_excel(rows, XLSX_PATH, run_config)
    except Exception as e:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"XLSX save failed: {e}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
