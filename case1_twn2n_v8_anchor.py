"""Anchor-channel TWN2N variant (v8) — extended N and SNR grids.

Same anchor architecture as v7:
  * one network per cell, scalar output (the anchor channel at the central instant);
  * anchor channel sees its own past + future neighbours (blind spot, current excluded);
  * other channels supply their FULL window (past + current + future).

What changed vs v7
------------------
  * N axis  : (1, 2, 4, 6, 8, 16)         was (2, 4, 8, 16); N=1 and N=6 added.
  * SNR axis: (5, 10, 15, 20, 25, 30, 35) was (5, ..., 25); 30 and 35 dB added.
  * n_z grid: 1 .. 2 N for N in {1, 2, 4, 6}   (full enumeration);
              12 evenly spaced values from 1 to 2 N for N in {8, 16}.
              No load cell channel in either case; case (a) and case (b)
              differ only in excitation (free vibration / single-shaker chirp)
              so the case-(b) results map the OMA-like regime in which the
              force is not measured by the denoiser.
  * case (c): random unmeasured forces w driving every modal coordinate
              with independent Gaussian white noise (broadband disturbance,
              no measurable input). The channel pool of case (c) is
              accelerations + velocities + positions per DOF in round-robin
              order (max 3 N channels), so at n_z = 3 the focal network sees
              one fully-instrumented DOF.

Sweep size
----------
  N            = (1, 2, 4, 6, 8, 16)
  n_z per N    = (2, 4, 8, 12, 12, 12)        total 50 values
  SNR          = (5, 10, 15, 20, 25, 30, 35)
  p_samples    = (1, 2, 3, 4, 5, 6)
  cases        = (a, b, c)
  cells/seed   = 3 * 7 * 50 * 6 = 6300

Outputs (parallel-safe with v6 and v7)
--------------------------------------
  case1_v8_anchor_results/case1_v8_anchor_results.csv     (incremental)
  case1_v8_anchor_results/case1_v8_anchor_results.xlsx    (final)
  case1_v8_anchor_results/case1_v8_anchor_run.log

ntfy notifications go to https://ntfy.sh/vinicius-claude-alert
(override with NTFY_TOPIC env var).
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
from scipy import linalg, signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ============================================================================
# Configuration
# ============================================================================
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "vinicius-claude-alert")

OUT_DIR = Path(__file__).parent / "case1_v8_anchor_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "case1_v8_anchor_results.csv"
XLSX_PATH = OUT_DIR / "case1_v8_anchor_results.xlsx"
LOG_PATH = OUT_DIR / "case1_v8_anchor_run.log"

# Sampling / record length (same as v6 / v7, uniform across all systems)
F_MAX_BRACKET = 1.0
F_MIN_BRACKET = 0.1
R_OVERSAMPLE = 16
M_CYCLES = 200
NPERSEG = 2048

FS = R_OVERSAMPLE * F_MAX_BRACKET
DT = 1.0 / FS
T_REC = M_CYCLES / F_MIN_BRACKET
N_S = int(T_REC * FS)

ZETA_MIN = 0.0015
ZETA_MAX = 0.05

# *** v8: extended axes ***
CASES = ("a", "b", "c")
_CASE_SEED_OFFSET = {"a": 0, "b": 1, "c": 2}
SNR_DB = (5, 10, 15, 20, 25, 30, 35)
N_MODES = (1, 2, 4, 6, 8, 16)
P_SAMPLES = (1, 2, 3, 4, 5, 6)
P_GRID = tuple(p / R_OVERSAMPLE for p in P_SAMPLES)


def n_sensors_grid(N: int, case: str = "a") -> list[int]:
    """n_z grid for a given modal count N and case.

    For N in {1, 2, 4, 6}: full enumeration 1 .. 2 N.
    For N in {8, 16}     : 12 evenly spaced values from 1 to 2 N (rounded,
                           duplicates removed).
    Case (a) and case (c) use the acc+vel (and pos, for c) pool only.
    Case (b) additionally exposes the measurable shaker force as one extra
    channel, so its n_z grid gets one extra value at the top (the cell that
    includes the force channel).
    """
    if N <= 6:
        base = list(range(1, 2 * N + 1))
    else:
        base = list(np.unique(np.round(np.linspace(1, 2 * N, 12)).astype(int)))
    if case == "b":
        base = base + [base[-1] + 1]
    return base


# Network / training (uniform-width architecture v2)
HIDDEN_WIDTH = 96      # all hidden layers have this many units
N_HIDDEN_LAYERS = 4    # number of hidden layers between input and output
LATENT_RHO = 1.0       # retained for CSV backward compatibility only
LEARNING_RATE = 1e-3
LOSS = "mean_absolute_error"
VAL_FRACTION = 0.10
PATIENCE = 50
EPOCHS = 500
BATCH_SIZE = 4096

N_EXP_PER_CELL = 3

CHIRP_F0 = F_MIN_BRACKET
CHIRP_F1 = F_MAX_BRACKET


# ============================================================================
# ntfy
# ============================================================================
def ntfy(message: str, title: str = "TWN2N v8 sweep",
         priority: str = "default", tags: str = "robot") -> None:
    if not NTFY_TOPIC:
        return
    try:
        import requests
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": tags},
            timeout=10,
        )
    except Exception:
        pass


# ============================================================================
# Random-system generator (identical to v7)
# ============================================================================
def random_orthonormal(N: int, rng: np.random.Generator) -> np.ndarray:
    G = rng.standard_normal(size=(N, N))
    Q, R = np.linalg.qr(G)
    sgn = np.sign(np.diag(R))
    sgn[sgn == 0] = 1.0
    return Q * sgn


def sample_system(N: int, rng: np.random.Generator) -> dict[str, Any]:
    log_w = rng.uniform(np.log(2 * np.pi * F_MIN_BRACKET),
                        np.log(2 * np.pi * F_MAX_BRACKET), size=N)
    omegas = np.exp(log_w)
    log_z = rng.uniform(np.log(ZETA_MIN), np.log(ZETA_MAX), size=N)
    zetas = np.exp(log_z)

    Omega = np.diag(omegas)
    Z = np.diag(zetas)
    A = np.block([
        [np.zeros((N, N)), np.eye(N)],
        [-Omega @ Omega, -2.0 * Z @ Omega],
    ])

    Psi = random_orthonormal(N, rng)
    S_u = Psi[:, :1]
    B_u = np.vstack([np.zeros((N, 1)), S_u])
    B_w = np.vstack([np.zeros((N, N)), np.eye(N)])

    return {
        "N": N, "omegas": omegas, "zetas": zetas,
        "f_max_sys": float(omegas.max() / (2 * np.pi)),
        "f_min_sys": float(omegas.min() / (2 * np.pi)),
        "A": A, "B_u": B_u, "B_w": B_w,
        "Psi": Psi, "S_u": S_u,
    }


# ============================================================================
# Trajectory simulation (identical to v7)
# ============================================================================
def discretise(A: np.ndarray, B: np.ndarray, dt: float
               ) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[0]
    m = B.shape[1] if B.ndim == 2 else 0
    if m == 0:
        return linalg.expm(A * dt), np.zeros((n, 0))
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = linalg.expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


def simulate_trajectory(system: dict[str, Any], case: str,
                        rng: np.random.Generator,
                        n_steps: int = N_S, dt: float = DT
                        ) -> dict[str, np.ndarray]:
    A = system["A"]
    B_u = system["B_u"]
    N = system["N"]
    Ad, B_u_d = discretise(A, B_u, dt)
    x = np.zeros((2 * N, n_steps))

    w = np.zeros((N, n_steps))
    if case == "a":
        omegas = system["omegas"]
        q0 = (1.0 / omegas) * rng.choice([-1.0, 1.0], size=N)
        x0 = np.concatenate([q0, np.zeros(N)])
        x0 /= np.linalg.norm(x0)
        x[:, 0] = x0
        u = np.zeros((1, n_steps))
        for k in range(n_steps - 1):
            x[:, k + 1] = Ad @ x[:, k]
    elif case == "b":
        t = np.arange(n_steps) * dt
        # Random-phase swept-sine: each of the N_EXP_PER_CELL trajectories of
        # a cell draws an independent phase, so the three realisations differ
        # in the excitation itself, not only in the noise draw.
        phi_deg = float(rng.uniform(0.0, 360.0))
        u = signal.chirp(t, f0=CHIRP_F0, f1=CHIRP_F1,
                         t1=float(t[-1]), method="linear",
                         phi=phi_deg).reshape(1, -1)
        for k in range(n_steps - 1):
            x[:, k + 1] = Ad @ x[:, k] + B_u_d[:, 0] * u[0, k]
    elif case == "c":
        # Random unmeasured forces w drive every modal coordinate with
        # independent Gaussian white noise. Initial state zero, no measurable
        # input. The denoiser sees only the noisy measurement record.
        _, B_w_d = discretise(A, system["B_w"], dt)
        w = rng.standard_normal((N, n_steps))
        u = np.zeros((1, n_steps))
        for k in range(n_steps - 1):
            x[:, k + 1] = Ad @ x[:, k] + B_w_d @ w[:, k]
    else:
        raise ValueError(f"Unknown case '{case}'")

    q = x[:N, :]
    q_dot = x[N:, :]
    Omega = np.diag(system["omegas"])
    Z = np.diag(system["zetas"])
    q_ddot = (-Omega @ Omega) @ q + (-2.0 * Z @ Omega) @ q_dot
    if case == "b":
        q_ddot += system["S_u"] @ u
    elif case == "c":
        q_ddot += w
    return {"q": q, "q_dot": q_dot, "q_ddot": q_ddot, "u": u}


# ============================================================================
# Channel ordering (no load cell in v8)
# ============================================================================
def channel_order_for_case(case: str, N: int) -> list[tuple[str, int]]:
    """Nested-consistent ordered list. Index 0 is the focal channel.

    * Case (a) and (b): N accelerometers then N velocity sensors
      (max n_z = 2 N). No load cell channel; case (b) hides the force from
      the denoiser, matching the OMA-like regime.
    * Case (c): per DOF triplet [acc, vel, pos] in round-robin order, so
      n_z = 1 sees acc_0, n_z = 2 adds vel_0, n_z = 3 completes DOF 0
      with pos_0, n_z = 4 starts DOF 1 with acc_1, etc. Max n_z = 3 N.
    """
    if case == "c":
        order = []
        for k in range(N):
            order.append(("acc", k))
            order.append(("vel", k))
            order.append(("pos", k))
        return order
    # Case (a) and (b): N accelerometers then N velocity sensors. The focal
    # channel (index 0) is always acc_0. Case (b) appends the measurable
    # shaker force as the LAST channel, so it is only included at the largest
    # n_z value and never displaces the focal channel.
    order = [("acc", 0)]
    for k in range(1, N):
        order.append(("acc", k))
    for k in range(N):
        order.append(("vel", k))
    if case == "b":
        order.append(("force", 0))
    return order


def select_channels_from_traj(traj: dict[str, np.ndarray],
                              channels_spec: list[tuple[str, int]]
                              ) -> tuple[np.ndarray, list[str]]:
    rows, names = [], []
    for ctype, dof in channels_spec:
        if ctype == "acc":
            rows.append(traj["q_ddot"][dof])
            names.append(f"acc_dof{dof}")
        elif ctype == "vel":
            rows.append(traj["q_dot"][dof])
            names.append(f"vel_dof{dof}")
        elif ctype == "pos":
            rows.append(traj["q"][dof])
            names.append(f"pos_dof{dof}")
        elif ctype == "force":
            rows.append(traj["u"][0])
            names.append("force")
        else:
            raise ValueError(f"Unknown channel type {ctype!r}")
    return np.vstack(rows), names


def add_noise_per_channel(z_clean: np.ndarray, snr_db: float,
                          rng: np.random.Generator
                          ) -> tuple[np.ndarray, np.ndarray]:
    snr_linear = 10.0 ** (snr_db / 10.0)
    sigma_clean = z_clean.std(axis=1, keepdims=True)
    sigma_clean = np.maximum(sigma_clean, 1e-12)
    sigma_v = sigma_clean / np.sqrt(snr_linear)
    v = rng.standard_normal(z_clean.shape) * sigma_v
    return z_clean + v, v


# ============================================================================
# Anchor-architecture lag matrix and TWN2N model (identical to v7)
# ============================================================================
class StopAfterOverfitting(Callback):
    def __init__(self, monitor="val_loss", patience=PATIENCE,
                 restore_best_weights=True):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.stopped_epoch = 0
        self.best_loss = np.inf
        self.best_weights = None

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_loss = np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        cv = logs.get(self.monitor)
        cl = logs.get("loss")
        if cv is None or cl is None:
            return
        if cv < self.best_loss:
            self.best_loss = cv
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
        if self.wait >= self.patience and cv > cl:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)


def build_twn2n_model(num_in: int, num_out: int, latent_dim: int,
                      n_train_rows: int) -> Model:
    """Uniform-width MLP: Input(num_in) -> N_HIDDEN_LAYERS x Dense(HIDDEN_WIDTH, tanh)
    -> Dense(num_out, linear).  The latent_dim and n_train_rows parameters are
    retained for call-site compatibility but ignored.
    """
    del latent_dim, n_train_rows  # signature-only, not used
    H = HIDDEN_WIDTH
    inp = Input(shape=(num_in,))
    x = inp
    for _ in range(N_HIDDEN_LAYERS):
        x = Dense(H, activation="tanh")(x)
    out = Dense(num_out, activation="linear")(x)
    return Model(inp, out, name=f"TWN2N_v8_uniformH{H}_L{N_HIDDEN_LAYERS}")


def build_lag_anchor(channels: np.ndarray, p: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Anchor architecture lag matrix.
    channels[0] is the anchor: blind-spot (2p samples). Others: full window (2p+1)."""
    n_z, n_t = channels.shape
    pad = np.zeros((n_z, p), dtype=channels.dtype)
    padded = np.concatenate([pad, channels, pad], axis=1)
    cols = []
    for i in range(p, 0, -1):
        cols.append(padded[0, p - i: p - i + n_t])
    for i in range(1, p + 1):
        cols.append(padded[0, p + i: p + i + n_t])
    for j in range(1, n_z):
        for i in range(p, 0, -1):
            cols.append(padded[j, p - i: p - i + n_t])
        cols.append(padded[j, p: p + n_t])
        for i in range(1, p + 1):
            cols.append(padded[j, p + i: p + i + n_t])
    X = np.stack(cols, axis=1).astype(np.float32)
    Y = channels[0:1, :].T.astype(np.float32)
    return X, Y


def denoise_anchor_multi(channels_list: list[np.ndarray], p: int,
                         latent_dim: int, seed: int
                         ) -> tuple[list[np.ndarray], dict[str, Any]]:
    tf.keras.utils.set_random_seed(seed)
    Xs, Ys = [], []
    for ch in channels_list:
        X, Y = build_lag_anchor(ch, p)
        Xs.append(X)
        Ys.append(Y)
    X_all = np.concatenate(Xs, axis=0)
    Y_all = np.concatenate(Ys, axis=0)

    sx = StandardScaler()
    sy = StandardScaler()
    X_s = sx.fit_transform(X_all)
    Y_s = sy.fit_transform(Y_all)
    X_tr, X_va, Y_tr, Y_va = train_test_split(
        X_s, Y_s, test_size=VAL_FRACTION, random_state=seed)

    model = build_twn2n_model(X_all.shape[1], 1, latent_dim,
                              n_train_rows=X_tr.shape[0])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=LOSS)
    cb = StopAfterOverfitting(patience=PATIENCE, restore_best_weights=True)
    t0 = time.time()
    history = model.fit(X_tr, Y_tr, validation_data=(X_va, Y_va),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[cb], verbose=0)
    train_time = time.time() - t0

    den_list = []
    for X in Xs:
        Yp = sy.inverse_transform(
            model.predict(sx.transform(X), batch_size=BATCH_SIZE, verbose=0))
        den_list.append(Yp[:, 0].astype(np.float64))

    info = {
        "epochs_run": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "train_time_s": train_time,
        "n_params": int(model.count_params()),
        "input_dim": int(X_all.shape[1]),
        "n_rows_total": int(X_all.shape[0]),
        "n_exp_per_cell": len(channels_list),
    }
    tf.keras.backend.clear_session()
    return den_list, info


# ============================================================================
# Metrics
# ============================================================================
def gain_db(noisy: np.ndarray, denoised: np.ndarray,
            clean: np.ndarray) -> dict[str, float]:
    err_in = (noisy - clean) ** 2
    err_out = (denoised - clean) ** 2
    sig = (clean ** 2).sum()
    nmse_in = float(err_in.sum() / max(sig, 1e-30))
    nmse_out = float(err_out.sum() / max(sig, 1e-30))
    g = 10.0 * np.log10(max(nmse_in, 1e-30) / max(nmse_out, 1e-30))
    return {"NMSE_in": nmse_in, "NMSE_out": nmse_out, "G_NMSE_dB": float(g)}


# ============================================================================
# Single cell runner
# ============================================================================
def run_cell(case: str, snr_db: float, n_modes: int, n_z: int,
             p_samples: int, seed: int) -> dict[str, Any]:
    rng_sys = np.random.default_rng(
        seed * 100003 + 7 * n_modes + _CASE_SEED_OFFSET[case])
    system = sample_system(n_modes, rng_sys)

    full_order = channel_order_for_case(case, n_modes)
    if n_z > len(full_order):
        raise ValueError(
            f"n_z={n_z} > max channels {len(full_order)} for case={case}, N={n_modes}")
    spec = full_order[:n_z]

    clean_list, noisy_list, ch_names = [], [], []
    for exp_id in range(N_EXP_PER_CELL):
        rng_traj = np.random.default_rng(
            seed * 100003 + 7 * n_modes + 1000 + 11 * exp_id
            + _CASE_SEED_OFFSET[case])
        traj = simulate_trajectory(system, case, rng_traj)
        chs, names = select_channels_from_traj(traj, spec)
        chs_noisy, _ = add_noise_per_channel(chs, snr_db, rng_traj)
        clean_list.append(chs)
        noisy_list.append(chs_noisy)
        if not ch_names:
            ch_names = names

    latent_dim = max(2, int(round(LATENT_RHO * 2 * n_modes)))
    den_list, train_info = denoise_anchor_multi(
        noisy_list, p_samples, latent_dim, seed)

    per_traj: list[dict[str, float]] = []
    for noisy_mat, den_arr, clean_mat in zip(noisy_list, den_list, clean_list):
        per_traj.append(gain_db(noisy_mat[0], den_arr, clean_mat[0]))

    anchor_noisy_pool = np.concatenate([n[0] for n in noisy_list])
    anchor_clean_pool = np.concatenate([c[0] for c in clean_list])
    anchor_den_pool = np.concatenate(den_list)
    pooled = gain_db(anchor_noisy_pool, anchor_den_pool, anchor_clean_pool)

    row: dict[str, Any] = {
        "case": case,
        "snr_db": snr_db,
        "n_modes": n_modes,
        "n_z": int(n_z),
        "p_samples": int(p_samples),
        "P_grid_nominal": p_samples / R_OVERSAMPLE,
        "P_sys": float(p_samples * DT * system["f_max_sys"]),
        "latent_dim": int(latent_dim),
        "rho": float(latent_dim / (2 * n_modes)),
        "f_max_sys": system["f_max_sys"],
        "f_min_sys": system["f_min_sys"],
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
# Sweep, resume, persistence
# ============================================================================
def build_cells(smoke: bool = False,
                cases_filter: tuple = CASES,
                snr_filter: tuple = SNR_DB,
                n_modes_filter: tuple = N_MODES) -> list[tuple]:
    """Build the list of sweep cells.

    The three ``*_filter`` arguments let the caller slice the sweep along any
    of the three discrete axes: this is how we distribute the same script
    across multiple machines (cluster does case=a, GitHub Actions matrix does
    case=b chunked by (SNR, N), etc.) without forking the codebase.
    """
    if smoke:
        return [
            ("a", 20, 1, 1, 3),
            ("a", 20, 6, 6, 3),
            ("b", 30, 8, 8, 3),
            ("b", 35, 16, 16, 3),
        ]
    cells = []
    for case in CASES:
        if case not in cases_filter:
            continue
        for snr in SNR_DB:
            if snr not in snr_filter:
                continue
            for N in N_MODES:
                if N not in n_modes_filter:
                    continue
                for n_z in n_sensors_grid(N, case):
                    for p in P_SAMPLES:
                        cells.append((case, snr, N, n_z, p))
    return cells


def load_already_done(csv_path: Path) -> set[tuple]:
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(zip(
            df["case"].astype(str),
            df["snr_db"].astype(float),
            df["n_modes"].astype(int),
            df["n_z"].astype(int),
            df["p_samples"].astype(int),
            df["seed"].astype(int),
        ))
    except Exception as exc:
        print(f"Warning: could not parse existing CSV ({exc}); starting fresh.",
              flush=True)
        return set()


def sweep_cells(cells: list[tuple], n_seeds: int,
                already_done: set[tuple],
                seed_start: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = len(cells) * n_seeds
    done = 0
    skipped = 0
    t_start = time.time()
    for (case, snr, N, n_z, p) in cells:
        for seed in range(seed_start, seed_start + n_seeds):
            key = (str(case), float(snr), int(N), int(n_z), int(p), int(seed))
            if key in already_done:
                done += 1
                skipped += 1
                continue
            t0 = time.time()
            try:
                row = run_cell(case, snr, N, n_z, p, seed)
            except Exception as exc:
                row = {
                    "case": case, "snr_db": snr, "n_modes": N,
                    "n_z": n_z, "p_samples": p, "seed": seed,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            row["wall_time_s"] = time.time() - t0
            rows.append(row)
            done += 1
            pd.DataFrame([row]).to_csv(
                CSV_PATH, mode="a", header=not CSV_PATH.exists(), index=False)
            elapsed = time.time() - t_start
            eta = elapsed / max(done - skipped, 1) * (total - done)
            log_line = (f"[{done}/{total}] case={case} snr={snr} N={N} "
                        f"n_z={n_z} p={p} seed={seed} -> "
                        f"G={row.get('G_NMSE_dB', 'n/a')!s:>6.6} "
                        f"({row['wall_time_s']:.1f}s)  eta={eta/60:.1f}min  "
                        f"(skipped: {skipped})")
            print(log_line, flush=True)
            with LOG_PATH.open("a", encoding="utf-8") as fh:
                fh.write(log_line + "\n")
    return rows


def write_xlsx(rows: list[dict[str, Any]], xlsx_path: Path,
               run_config: dict[str, Any]) -> None:
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
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of consecutive seeds to run starting at "
                             "--seed-start (default 1 -> just one seed).")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="First seed index to run (default 0). Combined "
                             "with --seeds this runs seed-start..seed-start+seeds-1. "
                             "Use --seeds 1 --seed-start N to pin a single seed.")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--cases", default=",".join(CASES),
                        help="Comma-separated subset of cases (a, b, c). "
                             "Default: 'a,b,c'. Use e.g. --cases c to run only "
                             "case (c) on this machine.")
    parser.add_argument("--snr-only", default=",".join(str(s) for s in SNR_DB),
                        help="Comma-separated subset of SNR_DB values. "
                             "Default: all. Use e.g. --snr-only 5,10 to slice "
                             "the sweep on a matrix runner.")
    parser.add_argument("--n-modes-only",
                        default=",".join(str(n) for n in N_MODES),
                        help="Comma-separated subset of N_MODES values. "
                             "Default: all.")
    args = parser.parse_args()

    cases_filter = tuple(c.strip() for c in args.cases.split(",") if c.strip())
    snr_filter = _parse_csv_list(args.snr_only, int)
    n_modes_filter = _parse_csv_list(args.n_modes_only, int)

    cells = build_cells(args.smoke, cases_filter=cases_filter,
                        snr_filter=snr_filter, n_modes_filter=n_modes_filter)
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
        "tensorflow": tf.__version__,
        "ntfy_topic": NTFY_TOPIC,
        "variant": "v8-anchor (extended N and SNR axes, no load cell channel)",
        "smoke": bool(args.smoke),
        "restart": bool(args.restart),
        "n_seeds": n_seeds,
        "n_cells": len(cells),
        "n_trainings": total,
        "n_already_done": len(already_done),
        "n_exp_per_cell": N_EXP_PER_CELL,
        "fs": FS, "dt": DT, "N_s": N_S, "T_rec": T_REC,
        "f_max_bracket": F_MAX_BRACKET, "f_min_bracket": F_MIN_BRACKET,
        "R_oversample": R_OVERSAMPLE, "M_cycles": M_CYCLES, "nperseg": NPERSEG,
        "zeta_min": ZETA_MIN, "zeta_max": ZETA_MAX,
        "CASES": CASES, "SNR_DB": SNR_DB, "N_MODES": N_MODES,
        "P_SAMPLES": P_SAMPLES,
        "n_sensors_per_N": {N: n_sensors_grid(N) for N in N_MODES},
        "EPOCHS": EPOCHS, "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE, "PATIENCE": PATIENCE,
        "LATENT_RHO": LATENT_RHO,
        "out_dir": str(OUT_DIR),
    }
    print(json.dumps(run_config, indent=2, default=str), flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"== v8 anchor sweep started {run_config['started_at']} "
                 f"(resume: {len(already_done)} cells) ==\n")

    ntfy(
        message=(f"Starting v8 anchor sweep on {run_config['host']}: "
                 f"{total} trainings, {len(already_done)} already done."),
        title="TWN2N v8 — sweep started",
        priority="low", tags="hourglass_flowing_sand")

    t_start = time.time()
    try:
        rows = sweep_cells(cells, n_seeds, already_done,
                           seed_start=args.seed_start)
    except KeyboardInterrupt:
        ntfy(message="v8 sweep interrupted by user.",
             title="TWN2N v8 — interrupted",
             priority="high", tags="warning")
        return 130
    except Exception as exc:
        tb = traceback.format_exc()
        msg = f"v8 sweep crashed: {type(exc).__name__}: {exc}\n\n{tb[:1800]}"
        print(msg, flush=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
        ntfy(message=msg, title="TWN2N v8 — CRASH",
             priority="urgent", tags="rotating_light")
        return 1

    elapsed_min = (time.time() - t_start) / 60
    run_config["finished_at"] = datetime.now().isoformat(timespec="seconds")
    run_config["elapsed_min"] = round(elapsed_min, 2)

    try:
        write_xlsx(rows, XLSX_PATH, run_config)
    except Exception as exc:
        ntfy(message=f"v8 finished but Excel write failed: {exc}",
             title="TWN2N v8 — done w/ warning",
             priority="high", tags="warning")
        raise

    n_err = sum(1 for r in rows if r.get("error"))
    summary = (f"v8 anchor sweep finished on {run_config['host']}: "
               f"{len(rows)} trainings, {n_err} errors, "
               f"{elapsed_min:.1f} min. Output: {XLSX_PATH.name}")
    print(summary, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write("\n" + summary + "\n")
    ntfy(message=summary, title="TWN2N v8 — DONE",
         priority="high",
         tags="white_check_mark" if n_err == 0 else "warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
