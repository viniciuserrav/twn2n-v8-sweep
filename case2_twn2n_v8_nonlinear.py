"""Case 2 — nonlinear pendulum sweep (v8 focal architecture).

Two physical models:
* ``n_modes = 1``: single pendulum (mass m, length L, tangential damping c).
  Measurements: [x, y] = [L sin θ, L cos θ].  Max channel pool = 2.
* ``n_modes = 2``: planar double pendulum (m1 = m2, L1 = L2, tangential
  damping on each angle).  Measurements: [x1, y1, x2, y2].  Max pool = 4.

Three excitation cases (matching Case 1):
* (a) free vibration from a random initial angle;
* (b) one known tangential shaker force applied to pendulum 1, log-linear
      chirp in the range used by Case 1;
* (c) random Gaussian "unmeasured" tangential forces on every pendulum.

The focal-architecture TWN2N denoiser and the training/eval pipeline are
imported from :mod:`case1_twn2n_v8_anchor` so the model definition is shared.

The CSV schema is identical to Case 1's (same column set, including
``n_modes``, which here means "number of pendulums in the chain"), so the
multi-seed study notebook can be reused with only minor changes.
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
from case1_twn2n_v8_anchor import (
    DT, N_S, R_OVERSAMPLE, NPERSEG,
    F_MIN_BRACKET, F_MAX_BRACKET, CHIRP_F0, CHIRP_F1,
    SNR_DB, P_SAMPLES, P_GRID, N_EXP_PER_CELL,
    LATENT_RHO, LEARNING_RATE, LOSS, VAL_FRACTION, PATIENCE, EPOCHS,
    BATCH_SIZE,
    denoise_anchor_multi, gain_db, add_noise_per_channel, ntfy,
)

# ============================================================================
# Configuration
# ============================================================================
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "vinicius-claude-alert")

OUT_DIR = HERE / "case2_v8_nonlinear_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "case2_v8_nonlinear_results.csv"
XLSX_PATH = OUT_DIR / "case2_v8_nonlinear_results.xlsx"
LOG_PATH = OUT_DIR / "case2_v8_nonlinear_run.log"

# Sweep grid — case2 axes
CASES = ("a", "b", "c")
_CASE_SEED_OFFSET = {"a": 10, "b": 11, "c": 12}  # offset by +10 from case1
N_MODELS = (1, 2)  # n_models = 1 -> single pendulum, n_models = 2 -> double

# Pendulum parameters (fixed across the sweep)
G = 9.81
L_LINK = 1.0          # length of each rod
M_BOB = 1.0           # mass of each bob
C_TANG = 0.02         # tangential damping coefficient (kg m^2 / s on each angle)
THETA0_MAX = 0.6      # max initial angle, rad — large enough to exercise sin nonlinearity

# Force amplitude for cases (b) and (c).  Scaled so the pendulum response is
# comparable to the case-1 amplitude scale (otherwise low-SNR cells become
# all-noise).
FORCE_AMP_B = 0.30
FORCE_AMP_C = 0.30


def n_sensors_grid_case2(n_models: int) -> list[int]:
    """Channel-count grid for case 2 (2 Cartesian sensors per pendulum)."""
    return list(range(1, 2 * n_models + 1))


# ============================================================================
# Pendulum dynamics (RK4, with sub-stepping for the nonlinear regime)
# ============================================================================
RK4_SUBSTEPS = 4  # internal sub-steps per macro dt


def _rk4_step(rhs, state, dt, *args):
    k1 = rhs(state, *args)
    k2 = rhs(state + 0.5 * dt * k1, *args)
    k3 = rhs(state + 0.5 * dt * k2, *args)
    k4 = rhs(state + dt * k3, *args)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _rhs_single(state, F_ext: float):
    """state = [theta, theta_dot].  F_ext = tangential force on the bob."""
    th, w = state
    a = -G / L_LINK * np.sin(th) - C_TANG * w / (M_BOB * L_LINK ** 2) \
        + F_ext / (M_BOB * L_LINK ** 2)
    return np.array([w, a])


def _rhs_double(state, F1: float, F2: float):
    """state = [theta1, theta2, w1, w2]. F1, F2: tangential torques on each bob.

    Lagrangian formulation: M(q) q̈ + C(q,q̇) q̇ + G(q) = τ ,
    with τ_i = -c θ̇_i + F_i.  Direct 2×2 inversion of M.
    """
    t1, t2, w1, w2 = state
    m1, m2, L1, L2, c = M_BOB, M_BOB, L_LINK, L_LINK, C_TANG
    cdiff = np.cos(t1 - t2)
    sdiff = np.sin(t1 - t2)
    M11 = (m1 + m2) * L1 ** 2
    M12 = m2 * L1 * L2 * cdiff
    M22 = m2 * L2 ** 2
    Cv1 = m2 * L1 * L2 * sdiff * w2 * w2
    Cv2 = -m2 * L1 * L2 * sdiff * w1 * w1
    Gv1 = (m1 + m2) * G * L1 * np.sin(t1)
    Gv2 = m2 * G * L2 * np.sin(t2)
    tau1 = -c * w1 + F1
    tau2 = -c * w2 + F2
    det = M11 * M22 - M12 * M12
    rhs1 = tau1 - Cv1 - Gv1
    rhs2 = tau2 - Cv2 - Gv2
    a1 = (M22 * rhs1 - M12 * rhs2) / det
    a2 = (M11 * rhs2 - M12 * rhs1) / det
    return np.array([w1, w2, a1, a2])


def _force_signal(case: str, n_steps: int, dt: float,
                  rng: np.random.Generator, n_force_channels: int):
    """Build the external-force time series for a given case.

    Returns ``F[c, k]`` of shape ``(n_force_channels, n_steps)``.
    """
    F = np.zeros((n_force_channels, n_steps))
    if case == "a":
        return F
    if case == "b":
        # Random-phase chirp so that different seeds give different
        # realisations even at the same n_models.
        phi = rng.uniform(0.0, 2 * np.pi)
        t = np.arange(n_steps) * dt
        chirp = signal.chirp(t, f0=CHIRP_F0, f1=CHIRP_F1,
                              t1=float(t[-1]), method="linear",
                              phi=np.degrees(phi))
        F[0] = FORCE_AMP_B * chirp
        return F
    if case == "c":
        F[:] = FORCE_AMP_C * rng.standard_normal((n_force_channels, n_steps))
        return F
    raise ValueError(f"Unknown case '{case}'")


def simulate_trajectory(n_models: int, case: str,
                        rng: np.random.Generator,
                        n_steps: int = N_S, dt: float = DT) -> dict[str, np.ndarray]:
    """Integrate the pendulum chain and return Cartesian channels.

    Output dict:
        ``x`` : (n_models, n_steps)   horizontal position of each bob
        ``y`` : (n_models, n_steps)   vertical   position of each bob
        ``theta`` : (n_models, n_steps)   angle of each link
    """
    sub_dt = dt / RK4_SUBSTEPS
    if n_models == 1:
        state = np.array([rng.uniform(-THETA0_MAX, THETA0_MAX), 0.0])
        F = _force_signal(case, n_steps, dt, rng, 1)
        theta = np.zeros(n_steps)
        omega = np.zeros(n_steps)
        theta[0], omega[0] = state[0], state[1]
        for k in range(n_steps - 1):
            for _ in range(RK4_SUBSTEPS):
                state = _rk4_step(_rhs_single, state, sub_dt, F[0, k])
            theta[k + 1], omega[k + 1] = state[0], state[1]
        x = (L_LINK * np.sin(theta))[None, :]
        y = (L_LINK * np.cos(theta))[None, :]
        return {"x": x, "y": y, "theta": theta[None, :]}
    if n_models == 2:
        state = np.array([rng.uniform(-THETA0_MAX, THETA0_MAX),
                          rng.uniform(-THETA0_MAX, THETA0_MAX),
                          0.0, 0.0])
        # Case (b) drives only pendulum 1 (single shaker on bob 1).
        # Case (c) drives both bobs independently.
        n_force = 2
        F = _force_signal(case, n_steps, dt, rng, n_force)
        if case == "b":
            F[1] = 0.0  # only one force
        theta = np.zeros((2, n_steps))
        theta[:, 0] = state[:2]
        for k in range(n_steps - 1):
            for _ in range(RK4_SUBSTEPS):
                state = _rk4_step(_rhs_double, state, sub_dt,
                                  F[0, k], F[1, k])
            theta[:, k + 1] = state[:2]
        x1 = L_LINK * np.sin(theta[0])
        y1 = L_LINK * np.cos(theta[0])
        x2 = x1 + L_LINK * np.sin(theta[1])
        y2 = y1 + L_LINK * np.cos(theta[1])
        x = np.stack([x1, x2], axis=0)
        y = np.stack([y1, y2], axis=0)
        return {"x": x, "y": y, "theta": theta}
    raise ValueError(f"Unsupported n_models={n_models} for case 2")


# ============================================================================
# Channel ordering (round-robin x, y per pendulum)
# ============================================================================
def channel_order_for_case2(n_models: int) -> list[tuple[str, int]]:
    """Returns ``[(x, 0), (y, 0), (x, 1), (y, 1), ...]`` truncated to
    ``2 * n_models`` entries.  Index 0 is the focal channel (x of pendulum 0).
    """
    order = []
    for k in range(n_models):
        order.append(("x", k))
        order.append(("y", k))
    return order


def select_channels_from_traj_case2(traj: dict[str, np.ndarray],
                                    channels_spec: list[tuple[str, int]]
                                    ) -> tuple[np.ndarray, list[str]]:
    rows, names = [], []
    for ctype, idx in channels_spec:
        if ctype == "x":
            rows.append(traj["x"][idx])
            names.append(f"x_pend{idx}")
        elif ctype == "y":
            rows.append(traj["y"][idx])
            names.append(f"y_pend{idx}")
        else:
            raise ValueError(f"Unknown channel type {ctype!r}")
    return np.vstack(rows), names


# ============================================================================
# Single-cell runner
# ============================================================================
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

    # Approximate "system" instantaneous frequency: small-angle linearisation
    # gives omega_n = sqrt(g/L) for the single pendulum, and the two normal
    # modes of the symmetric double pendulum are at sqrt(g/L) and sqrt(3 g/L).
    # We report sqrt(g/L)/(2π) as f_max_sys for both, so the P_sys = p*dt*f_max_sys
    # is comparable to case 1.
    f_max_sys = float(np.sqrt(G / L_LINK) / (2 * np.pi))
    f_min_sys = f_max_sys  # single dominant freq for both models

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


# ============================================================================
# Sweep / resume / persistence
# ============================================================================
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


# ============================================================================
# Main
# ============================================================================
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
    if already_done:
        print(f"Resume: found {len(already_done)} cells already in "
              f"{CSV_PATH.name}; they will be skipped.", flush=True)

    run_config = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "ntfy_topic": NTFY_TOPIC,
        "variant": "case2-v8-nonlinear (pendulum + double pendulum)",
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
        fh.write(f"== case2 v8 sweep started {run_config['started_at']} "
                 f"(resume: {len(already_done)} cells) ==\n")

    ntfy(message=(f"Starting case2 v8 sweep on {run_config['host']}: "
                  f"{total} trainings, {len(already_done)} already done."),
         title="TWN2N case2 — sweep started",
         priority="low", tags="hourglass_flowing_sand")

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
                ntfy(f"case2 v8 cell FAILED: seed={seed} case={case} "
                     f"snr={snr} N={n_models} n_z={n_z} p={p} — {e}",
                     title="TWN2N case2 — cell failed",
                     priority="default", tags="warning")
                continue
            row["wall_s"] = time.time() - t0
            rows.append(row)
            append_row(row)
            print(f"[seed={seed} cell {ci}/{len(cells)}] case={case} "
                  f"snr={snr} N={n_models} n_z={n_z} p={p}  "
                  f"G_NMSE={row['G_NMSE_dB']:+.2f} dB  "
                  f"({row['wall_s']:.1f} s)", flush=True)

    elapsed = time.time() - t_start
    msg = (f"case2 v8 sweep DONE in {elapsed/60:.1f} min. "
           f"Wrote {len(rows)} new rows, skipped {n_skipped}.")
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(msg + "\n")
    ntfy(message=msg, title="TWN2N case2 — sweep done",
         priority="default", tags="white_check_mark")

    try:
        save_excel(rows, XLSX_PATH, run_config)
    except Exception as e:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(f"XLSX save failed: {e}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
