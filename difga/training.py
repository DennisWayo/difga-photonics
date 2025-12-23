import matplotlib.pyplot as plt
from pennylane import numpy as np
import pennylane as qml
from difga.noise import (
    MC_SAMPLES,
    NG_SEED,
)

from difga.config import DEFAULT_INPUT_PARAMS, DEFAULT_PROC_PARAMS, EPS
from difga.cost import cost_mm, cost_sm, cost_mm_ng
from difga.io_utils import savefig, save_log

def train_mm(ip, pp, eta, steps=60, lr=0.06):
    opt = qml.GradientDescentOptimizer(lr)
    params = np.zeros(6, requires_grad=True)
    hist = []
    for _ in range(steps):
        params, loss = opt.step_and_cost(lambda p: cost_mm(p, ip, pp, eta), params)
        hist.append(loss)
    return params, np.array(hist)

def train_sm(r, phi, alpha, eta, steps=60, lr=0.06):
    opt = qml.GradientDescentOptimizer(lr)
    params = np.zeros(3, requires_grad=True)
    hist = []
    for _ in range(steps):
        params, loss = opt.step_and_cost(lambda p: cost_sm(p, r, phi, alpha, eta), params)
        hist.append(loss)
    return params, np.array(hist)

def train_mm_ng(ip, pp, eta, delta_sig, delta_anc, steps=60, lr=0.06, K=32):
    opt = qml.GradientDescentOptimizer(lr)
    params = np.zeros(6, requires_grad=True)
    hist = []
    for _ in range(steps):
        params, loss = opt.step_and_cost(
            lambda p: cost_mm_ng(p, ip, pp, eta, delta_sig, delta_anc, K),
            params
        )
        hist.append(loss)
    return params, np.array(hist)

# ============================================================
# ANALYSES
# ============================================================

def analysis1_noise_sweep():
    ip = np.array(DEFAULT_INPUT_PARAMS)
    pp = np.array(DEFAULT_PROC_PARAMS)

    etas = np.linspace(0.3, 0.95, 7)
    errs = []
    for eta in etas:
        _, hist = train_mm(ip, pp, eta)
        errs.append(hist[-1])

    plt.plot(etas, errs, marker="o")
    plt.xlabel("Loss transmissivity η")
    plt.ylabel("Final error")
    plt.title("Gaussian loss mitigation")
    savefig("analysis1_noise")
    plt.show()

    save_log("analysis1_noise.csv", np.column_stack([etas, errs]), "eta,error")


def analysis2_single_vs_multi():
    ip = np.array(DEFAULT_INPUT_PARAMS)
    pp = np.array(DEFAULT_PROC_PARAMS)
    eta = 0.55

    r, phi, alpha = ip[:3]
    p_sm, _ = train_sm(r, phi, alpha, eta)
    p_mm, _ = train_mm(ip, pp, eta)

    vals = [
        cost_sm(np.zeros(3), r, phi, alpha, eta),
        cost_sm(p_sm, r, phi, alpha, eta),
        cost_mm(np.zeros(6), ip, pp, eta),
        cost_mm(p_mm, ip, pp, eta),
    ]

    plt.bar(["SM base", "SM mit", "MM base", "MM mit"], vals)
    plt.ylabel("Error")
    plt.title("Single vs Multi-mode mitigation")
    savefig("analysis2_single_vs_multi")
    plt.show()


def analysis3_phase_diagram_ng():
    ip = np.array(DEFAULT_INPUT_PARAMS)
    pp = np.array(DEFAULT_PROC_PARAMS)

    etas = np.linspace(0.3, 0.95, 6)
    deltas = np.linspace(0.0, 0.7, 6)

    Z = np.zeros((len(etas), len(deltas)))
    for i, eta in enumerate(etas):
        for j, d in enumerate(deltas):
            _, hist = train_mm_ng(ip, pp, eta, d, 0.6 * d, steps=30)
            Z[i, j] = hist[-1]

    X, Y = np.meshgrid(deltas, etas)
    plt.contourf(X, Y, np.log10(Z + EPS), levels=20)
    plt.colorbar(label="log10(error)")
    plt.xlabel("Phase jitter δ")
    plt.ylabel("Loss transmissivity η")
    plt.title("Non-Gaussian phase noise")
    savefig("analysis3_phase_diagram_ng")
    plt.show()

# ============================================================
# ANALYSIS 4 — NG Robustness & Generalization (SEPARATE CELL)
# Heavy Monte-Carlo — run only when needed
# ============================================================

def analysis4_ng_generalization(
    eta=0.55,
    deltas=np.linspace(0.0, 0.7, 6),
    steps_gauss=60,
    steps_ng=40,
    lr=0.06,
    K=MC_SAMPLES,
    seed=NG_SEED
):
    """
    Compare:
      (1) Recovery trained on Gaussian loss only
      (2) Recovery trained with explicit non-Gaussian phase noise

    Evaluation: final NG error vs phase jitter δ
    """

    print("\n[Analysis 4] Training Gaussian-only recovery...")
    p_gauss, _ = train_mm(
        default_input_params,
        default_proc_params,
        eta,
        steps=steps_gauss,
        lr=lr
    )

    print("[Analysis 4] Training NG-aware recovery...")
    p_ng, _ = train_mm_ng(
        default_input_params,
        default_proc_params,
        eta,
        delta_sig=0.3,
        delta_anc=0.18,
        steps=steps_ng,
        lr=lr,
        K=K
    )

    err_gauss = []
    err_ng = []

    for d in deltas:
        e_g = cost_mm_ng(
            p_gauss,
            default_input_params,
            default_proc_params,
            eta,
            delta_sig=d,
            delta_anc=0.6 * d,
            K=K
        )
        e_n = cost_mm_ng(
            p_ng,
            default_input_params,
            default_proc_params,
            eta,
            delta_sig=d,
            delta_anc=0.6 * d,
            K=K
        )
        err_gauss.append(float(e_g))
        err_ng.append(float(e_n))

    err_gauss = np.array(err_gauss)
    err_ng = np.array(err_ng)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(deltas, err_gauss, "o-", label="Gaussian-trained")
    plt.plot(deltas, err_ng, "s-", label="NG-trained")
    plt.xlabel("Phase jitter δ (rad)")
    plt.ylabel("Final error")
    plt.title("Generalization under non-Gaussian phase noise")
    plt.legend()
    plt.tight_layout()
    savefig("analysis4_ng_generalization")
    plt.show()

    save_log(
        "analysis4_ng_generalization.csv",
        np.column_stack([deltas, err_gauss, err_ng]),
        header="delta,err_gauss_trained,err_ng_trained"
    )

    return deltas, err_gauss, err_ng


# ============================================================
# ANALYSIS 5  — Phase-noise critical threshold detection
# ============================================================

def find_critical_delta(
    eta=0.55,
    delta_grid=None,
    criterion="ratio",      # "ratio" or "absolute"
    ratio_thresh=0.90,      # mitigated must be <= ratio_thresh * baseline
    abs_thresh=None,        # mitigated must be <= abs_thresh
    steps_train=40,
    lr=0.06,
    K=MC_SAMPLES,
    train_delta=0.25,
):
    """
    Finds the smallest delta where mitigation 'fails' under NG noise.

    Baseline: params = zeros (no recovery)
    Mitigated: params trained under NG at train_delta (signal) and 0.6*train_delta (ancilla)

    Failure criterion:
      - ratio: mitigated_error > ratio_thresh * baseline_error
      - absolute: mitigated_error > abs_thresh
    """
    if delta_grid is None:
        delta_grid = np.linspace(0.0, 0.8, 17)

    ip, pp = default_input_params, default_proc_params

    # Train NG-aware recovery once
    params_star, _ = train_mm_ng(
        ip, pp, eta,
        delta_sig=train_delta,
        delta_anc=0.6 * train_delta,
        steps=steps_train,
        lr=lr,
        K=K
    )

    baseline = []
    mitigated = []
    fail_flags = []

    for d in delta_grid:
        e_base = cost_mm_ng(
            np.zeros(6), ip, pp, eta,
            delta_sig=d, delta_anc=0.6 * d,
            K=K
        )
        e_mit = cost_mm_ng(
            params_star, ip, pp, eta,
            delta_sig=d, delta_anc=0.6 * d,
            K=K
        )

        e_base_f = float(e_base)
        e_mit_f = float(e_mit)

        baseline.append(e_base_f)
        mitigated.append(e_mit_f)

        if criterion == "ratio":
            fail = e_mit_f > ratio_thresh * e_base_f
        elif criterion == "absolute":
            if abs_thresh is None:
                raise ValueError("abs_thresh must be provided when criterion='absolute'")
            fail = e_mit_f > abs_thresh
        else:
            raise ValueError("criterion must be 'ratio' or 'absolute'")

        fail_flags.append(fail)

    baseline = np.array(baseline)
    mitigated = np.array(mitigated)
    fail_flags = np.array(fail_flags)

    # Detect critical delta (first failure)
    idx = np.where(fail_flags)[0]
    dcrit = float(delta_grid[idx[0]]) if len(idx) > 0 else None

    # Plot
    plt.figure(figsize=(7,4))
    plt.plot(delta_grid, baseline, "o-", label="Baseline (no recovery)")
    plt.plot(delta_grid, mitigated, "s-", label="Mitigated (NG-trained)")
    if dcrit is not None:
        plt.axvline(dcrit, linestyle="--")
        plt.text(dcrit, max(baseline.max(), mitigated.max())*0.9, f" δ*={dcrit:.2f}", rotation=90)
    plt.xlabel("Phase jitter δ (rad)")
    plt.ylabel("Error")
    plt.title(f"Critical phase-noise threshold (criterion={criterion})")
    plt.legend()
    plt.tight_layout()
    savefig("analysisA_critical_delta")
    plt.show()

    save_log(
        "analysisA_critical_delta.csv",
        np.column_stack([delta_grid, baseline, mitigated, fail_flags.astype(int)]),
        header="delta,baseline,mitigated,fail_flag"
    )

    print("Critical delta δ* =", dcrit)
    return dcrit, delta_grid, baseline, mitigated, fail_flags



# ============================================================
# ANALYSIS 6 — Sensitivity scaling vs number of modes
# ============================================================

def build_devices_and_layers_for_modes(M):
    """
    Modes:
      0 = signal
      1..M-2 = ancillas (if any)
      M-1 = environment for signal loss

    Recovery: local phase+displacement on signal and all ancillas (excluding env)
    """
    dev = qml.device("default.gaussian", wires=M)

    def prepare_input(ip):
        # ip = [r_s, phi_s, alpha_s, r_a, phi_a] used only for signal+first ancilla
        r_s, phi_s, alpha_s, r_a, phi_a = ip
        qml.Squeezing(r_s, phi_s, wires=0)
        qml.Displacement(alpha_s, 0.0, wires=0)
        if M >= 3:
            qml.Squeezing(r_a, phi_a, wires=1)  # first ancilla
        # any extra ancillas beyond wire 1 remain vacuum

    def entangle(pp):
        # Entangle signal with each ancilla using same beamsplitter params
        theta, phi = pp
        for a in range(1, M-1):
            qml.Beamsplitter(theta, phi, wires=[0, a])

    def loss(eta):
        theta = np.arccos(np.sqrt(eta))
        qml.Beamsplitter(theta, 0.0, wires=[0, M-1])

    def recovery(params):
        # params length = 3*(M-1): [phi0, d0r, d0i, phi1, d1r, d1i, ...] for wires 0..M-2
        assert len(params) == 3*(M-1)
        for w in range(M-1):
            phi = params[3*w + 0]
            dr  = params[3*w + 1]
            di  = params[3*w + 2]
            qml.Rotation(phi, wires=w)
            qml.Displacement(dr + 1j*di, 0.0, wires=w)

    @qml.qnode(dev, interface="autograd")
    def noisy_x(ip, pp, eta, params):
        prepare_input(ip)
        entangle(pp)
        loss(eta)
        recovery(params)
        return qml.expval(qml.QuadX(0))

    @qml.qnode(dev, interface="autograd")
    def noisy_p(ip, pp, eta, params):
        prepare_input(ip)
        entangle(pp)
        loss(eta)
        recovery(params)
        return qml.expval(qml.QuadP(0))

    @qml.qnode(dev)
    def ideal_x(ip, pp):
        prepare_input(ip)
        entangle(pp)
        return qml.expval(qml.QuadX(0))

    @qml.qnode(dev)
    def ideal_p(ip, pp):
        prepare_input(ip)
        entangle(pp)
        return qml.expval(qml.QuadP(0))

    def cost(params, ip, pp, eta):
        return (ideal_x(ip, pp) - noisy_x(ip, pp, eta, params))**2 + (ideal_p(ip, pp) - noisy_p(ip, pp, eta, params))**2

    def train(ip, pp, eta, steps=40, lr=0.06):
        opt = qml.GradientDescentOptimizer(lr)
        params = np.zeros(3*(M-1), requires_grad=True)
        hist = []
        for _ in range(steps):
            params, loss_val = opt.step_and_cost(lambda p: cost(p, ip, pp, eta), params)
            hist.append(loss_val)
        return params, np.array(hist)

    return cost, train, M


def analysisB_sensitivity_vs_modes(
    modes_list=(2,3,4,5),
    eta=0.55,
    steps=40,
    lr=0.06
):
    """
    Measures how final mitigated error scales as we add more ancilla modes.
    M=2 corresponds to single-mode+env (signal=0, env=1) with no ancilla.
    """
    ip, pp = default_input_params, default_proc_params

    finals_base = []
    finals_mit  = []

    for M in modes_list:
        cost, train, _ = build_devices_and_layers_for_modes(M)

        p_star, hist = train(ip, pp, eta, steps=steps, lr=lr)

        e_base = float(cost(np.zeros(3*(M-1)), ip, pp, eta))
        e_mit  = float(cost(p_star, ip, pp, eta))

        finals_base.append(e_base)
        finals_mit.append(e_mit)

        print(f"M={M}: base={e_base:.4e}  mit={e_mit:.4e}")

    finals_base = np.array(finals_base)
    finals_mit  = np.array(finals_mit)

    plt.figure(figsize=(6,4))
    plt.plot(modes_list, finals_base, "o-", label="Baseline")
    plt.plot(modes_list, finals_mit, "s-", label="Mitigated")
    plt.xlabel("Total modes M (includes env)")
    plt.ylabel("Final error")
    plt.title("Sensitivity scaling vs number of modes")
    plt.legend()
    plt.tight_layout()
    savefig("analysisB_sensitivity_vs_modes")
    plt.show()

    save_log(
        "analysisB_sensitivity_vs_modes.csv",
        np.column_stack([np.array(modes_list), finals_base, finals_mit]),
        header="M,baseline,mitigated"
    )

    return np.array(modes_list), finals_base, finals_mit

#%%
# ============================================================
# Differentiable Error Mitigation in Quantum Photonic Circuits
# PennyLane 0.43.1
# ============================================================

import pennylane as qml
from pennylane import numpy as np
import numpy as onp
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# GLOBAL CONFIG
# ============================================================

plt.rcParams["figure.dpi"] = 140
plt.rcParams["axes.grid"] = True

FIG_DIR = Path("figures")
LOG_DIR = Path("logs")
FIG_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

def savefig(name, dpi=300):
    plt.savefig(FIG_DIR / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    print(f"[Saved] figures/{name}.png, figures/{name}.pdf")

def save_log(name, data, header=None):
    path = LOG_DIR / name
    np.savetxt(path, np.asarray(data, dtype=float), delimiter=",",
               header=(header or ""), comments="")
    print(f"[Saved log] {path}")

print("PennyLane version:", qml.__version__)

# ============================================================
# DEFAULT PARAMETERS
# ============================================================

# input params: [r_s, phi_s, alpha_s, r_a, phi_a]
default_input_params = np.array([0.6, 0.3, 0.8, 0.4, 0.1])
# proc params: [theta_bs, phi_bs]
default_proc_params  = np.array([0.7, 0.2])

# ============================================================
# DEVICES
# ============================================================

dev_mm = qml.device("default.gaussian", wires=3)  # signal, ancilla, environment
dev_sm = qml.device("default.gaussian", wires=2)  # signal, environment

# ============================================================
# NON-GAUSSIAN PHASE NOISE (Monte-Carlo)
# ============================================================

NG_SEED = 123
MC_SAMPLES = 32
DELTA_SIG_DEFAULT = 0.20
DELTA_ANC_DEFAULT = 0.12
EPS = 1e-12

def make_phase_samples(delta_sig, delta_anc, K=MC_SAMPLES, seed=NG_SEED):
    rng = onp.random.RandomState(seed)   # <-- FIX
    eps_sig = rng.normal(0.0, float(delta_sig), K)
    eps_anc = rng.normal(0.0, float(delta_anc), K)
    return eps_sig, eps_anc

# ============================================================
# MULTI-MODE BUILDING BLOCKS
# ============================================================

def prepare_input_state_mm(r_s, phi_s, alpha_s, r_a, phi_a):
    qml.Squeezing(r_s, phi_s, wires=0)
    qml.Displacement(alpha_s, 0.0, wires=0)
    qml.Squeezing(r_a, phi_a, wires=1)

def entangling_layer(theta, phi):
    qml.Beamsplitter(theta, phi, wires=[0, 1])

def loss_on_signal_mm(eta):
    theta = np.arccos(np.sqrt(eta))
    qml.Beamsplitter(theta, 0.0, wires=[0, 2])

def recovery_layer_mm(params):
    phi0, d0r, d0i, phi1, d1r, d1i = params
    qml.Rotation(phi0, wires=0)
    qml.Displacement(d0r + 1j*d0i, 0.0, wires=0)
    qml.Rotation(phi1, wires=1)
    qml.Displacement(d1r + 1j*d1i, 0.0, wires=1)

def phase_noise_layer_mm(eps_sig, eps_anc):
    qml.Rotation(eps_sig, wires=0)
    qml.Rotation(eps_anc, wires=1)

# ============================================================
# SINGLE-MODE BUILDING BLOCKS
# ============================================================

def prepare_input_state_sm(r, phi, alpha):
    qml.Squeezing(r, phi, wires=0)
    qml.Displacement(alpha, 0.0, wires=0)

def loss_on_signal_sm(eta):
    theta = np.arccos(np.sqrt(eta))
    qml.Beamsplitter(theta, 0.0, wires=[0, 1])

def recovery_layer_sm(params):
    phi, dr, di = params
    qml.Rotation(phi, wires=0)
    qml.Displacement(dr + 1j*di, 0.0, wires=0)

# ============================================================
# IDEAL QNODES
# ============================================================

@qml.qnode(dev_mm)
def ideal_x_mm(ip, pp):
    prepare_input_state_mm(*ip)
    entangling_layer(*pp)
    return qml.expval(qml.QuadX(0))

@qml.qnode(dev_mm)
def ideal_p_mm(ip, pp):
    prepare_input_state_mm(*ip)
    entangling_layer(*pp)
    return qml.expval(qml.QuadP(0))

@qml.qnode(dev_sm)
def ideal_x_sm(r, phi, alpha):
    prepare_input_state_sm(r, phi, alpha)
    return qml.expval(qml.QuadX(0))

@qml.qnode(dev_sm)
def ideal_p_sm(r, phi, alpha):
    prepare_input_state_sm(r, phi, alpha)
    return qml.expval(qml.QuadP(0))

# ============================================================
# NOISY QNODES
# ============================================================

@qml.qnode(dev_mm, interface="autograd")
def noisy_x_mm(ip, pp, eta, params):
    prepare_input_state_mm(*ip)
    entangling_layer(*pp)
    loss_on_signal_mm(eta)
    recovery_layer_mm(params)
    return qml.expval(qml.QuadX(0))

@qml.qnode(dev_mm, interface="autograd")
def noisy_p_mm(ip, pp, eta, params):
    prepare_input_state_mm(*ip)
    entangling_layer(*pp)
    loss_on_signal_mm(eta)
    recovery_layer_mm(params)
    return qml.expval(qml.QuadP(0))

@qml.qnode(dev_sm, interface="autograd")
def noisy_x_sm(r, phi, alpha, eta, params):
    prepare_input_state_sm(r, phi, alpha)
    loss_on_signal_sm(eta)
    recovery_layer_sm(params)
    return qml.expval(qml.QuadX(0))

@qml.qnode(dev_sm, interface="autograd")
def noisy_p_sm(r, phi, alpha, eta, params):
    prepare_input_state_sm(r, phi, alpha)
    loss_on_signal_sm(eta)
    recovery_layer_sm(params)
    return qml.expval(qml.QuadP(0))

@qml.qnode(dev_mm, interface="autograd")
def noisy_x_mm_ng(ip, pp, eta, params, eps_sig, eps_anc):
    prepare_input_state_mm(*ip)
    entangling_layer(*pp)
    loss_on_signal_mm(eta)
    phase_noise_layer_mm(eps_sig, eps_anc)
    recovery_layer_mm(params)
    return qml.expval(qml.QuadX(0))

@qml.qnode(dev_mm, interface="autograd")
def noisy_p_mm_ng(ip, pp, eta, params, eps_sig, eps_anc):
    prepare_input_state_mm(*ip)
    entangling_layer(*pp)
    loss_on_signal_mm(eta)
    phase_noise_layer_mm(eps_sig, eps_anc)
    recovery_layer_mm(params)
    return qml.expval(qml.QuadP(0))

# ============================================================
# MONTE-CARLO AVERAGING
# ============================================================

def mc_avg_noisy_quads_mm(
    ip, pp, eta, params,
    delta_sig, delta_anc,
    K=MC_SAMPLES, seed=NG_SEED
):
    eps_sig, eps_anc = make_phase_samples(delta_sig, delta_anc, K, seed)

    xs = []
    ps = []
    for k in range(K):
        xk = noisy_x_mm_ng(ip, pp, eta, params, eps_sig[k], eps_anc[k])
        pk = noisy_p_mm_ng(ip, pp, eta, params, eps_sig[k], eps_anc[k])
        xs.append(xk)
        ps.append(pk)

    xs = qml.math.stack(xs)   # ← KEY FIX
    ps = qml.math.stack(ps)

    return qml.math.mean(xs), qml.math.mean(ps)

# ============================================================
# COST FUNCTIONS
# ============================================================

def cost_mm(params, ip, pp, eta):
    return (ideal_x_mm(ip, pp) - noisy_x_mm(ip, pp, eta, params))**2 + \
           (ideal_p_mm(ip, pp) - noisy_p_mm(ip, pp, eta, params))**2

def cost_sm(params, r, phi, alpha, eta):
    return (ideal_x_sm(r, phi, alpha) - noisy_x_sm(r, phi, alpha, eta, params))**2 + \
           (ideal_p_sm(r, phi, alpha) - noisy_p_sm(r, phi, alpha, eta, params))**2

def cost_mm_ng(params, ip, pp, eta,
               delta_sig=DELTA_SIG_DEFAULT,
               delta_anc=DELTA_ANC_DEFAULT,
               K=MC_SAMPLES, seed=NG_SEED):
    x_id = ideal_x_mm(ip, pp)
    p_id = ideal_p_mm(ip, pp)
    x_no, p_no = mc_avg_noisy_quads_mm(ip, pp, eta, params,
                                      delta_sig, delta_anc, K, seed)
    return (x_id - x_no)**2 + (p_id - p_no)**2

# ============================================================
# TRAINING ROUTINES
# ============================================================

def train_mm(ip, pp, eta, steps=60, lr=0.06):
    opt = qml.GradientDescentOptimizer(lr)
    params = np.zeros(6, requires_grad=True)
    hist = []
    for _ in range(steps):
        params, loss = opt.step_and_cost(lambda p: cost_mm(p, ip, pp, eta), params)
        hist.append(loss)
    return params, np.array(hist)

def train_sm(r, phi, alpha, eta, steps=60, lr=0.06):
    opt = qml.GradientDescentOptimizer(lr)
    params = np.zeros(3, requires_grad=True)
    hist = []
    for _ in range(steps):
        params, loss = opt.step_and_cost(lambda p: cost_sm(p, r, phi, alpha, eta), params)
        hist.append(loss)
    return params, np.array(hist)

def train_mm_ng(ip, pp, eta, delta_sig, delta_anc,
                steps=60, lr=0.06, K=MC_SAMPLES):
    opt = qml.GradientDescentOptimizer(lr)
    params = np.zeros(6, requires_grad=True)
    hist = []
    for _ in range(steps):
        params, loss = opt.step_and_cost(
            lambda p: cost_mm_ng(p, ip, pp, eta, delta_sig, delta_anc, K),
            params
        )
        hist.append(loss)
    return params, np.array(hist)


# ============================================================
# APPENDIX A — CIRCUIT DIAGRAMS (qml.draw)
# ============================================================

def draw_circuits():
    ip, pp = default_input_params, default_proc_params
    eta = 0.55

    print("\n--- Ideal multi-mode circuit (QuadX) ---")
    print(qml.draw(ideal_x_mm)(ip, pp))

    print("\n--- Noisy multi-mode + recovery (QuadX) ---")
    print(qml.draw(noisy_x_mm)(ip, pp, eta, np.zeros(6)))

    r, phi, alpha = ip[:3]
    print("\n--- Ideal single-mode circuit (QuadX) ---")
    print(qml.draw(ideal_x_sm)(r, phi, alpha))

    print("\n--- Noisy single-mode + recovery (QuadX) ---")
    print(qml.draw(noisy_x_sm)(r, phi, alpha, eta, np.zeros(3)))

# ============================================================
# ANALYSES
# ============================================================

def analysis1_noise_sweep():
    etas = np.linspace(0.3, 0.95, 7)
    errs = []
    for eta in etas:
        _, hist = train_mm(default_input_params, default_proc_params, eta)
        errs.append(hist[-1])
    plt.plot(etas, errs, marker="o")
    plt.xlabel("Loss transmissivity η")
    plt.ylabel("Final error")
    plt.title("Gaussian loss mitigation")
    savefig("analysis1_noise")
    plt.show()
    save_log("analysis1_noise.csv", np.column_stack([etas, errs]), "eta,error")

def analysis2_single_vs_multi():
    eta = 0.55
    r, phi, alpha = default_input_params[:3]
    p_sm, _ = train_sm(r, phi, alpha, eta)
    p_mm, _ = train_mm(default_input_params, default_proc_params, eta)
    vals = [
        cost_sm(np.zeros(3), r, phi, alpha, eta),
        cost_sm(p_sm, r, phi, alpha, eta),
        cost_mm(np.zeros(6), default_input_params, default_proc_params, eta),
        cost_mm(p_mm, default_input_params, default_proc_params, eta),
    ]
    plt.bar(["SM base","SM mit","MM base","MM mit"], vals)
    plt.ylabel("Error")
    plt.title("Single vs Multi-mode mitigation")
    savefig("analysis2_single_vs_multi")
    plt.show()

def analysis3_phase_diagram_ng():
    etas = np.linspace(0.3, 0.95, 6)
    deltas = np.linspace(0.0, 0.7, 6)
    Z = np.zeros((len(etas), len(deltas)))
    for i, eta in enumerate(etas):
        for j, d in enumerate(deltas):
            _, hist = train_mm_ng(default_input_params, default_proc_params,
                                  eta, d, 0.6*d, steps=30)
            Z[i,j] = hist[-1]
    X,Y = np.meshgrid(deltas, etas)
    plt.contourf(X, Y, np.log10(Z+EPS), levels=20)
    plt.colorbar(label="log10(error)")
    plt.xlabel("Phase jitter δ")
    plt.ylabel("Loss transmissivity η")
    plt.title("Non-Gaussian phase noise")
    savefig("analysis3_phase_diagram_ng")
    plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    draw_circuits()
    analysis1_noise_sweep()
    analysis2_single_vs_multi()
    analysis3_phase_diagram_ng()
#%%
# ============================================================
# ANALYSIS 4 — NG Robustness & Generalization (SEPARATE CELL)
# Heavy Monte-Carlo — run only when needed
# ============================================================

def analysis4_ng_generalization(
    eta=0.55,
    deltas=np.linspace(0.0, 0.7, 6),
    steps_gauss=60,
    steps_ng=40,
    lr=0.06,
    K=MC_SAMPLES,
    seed=NG_SEED
):
    """
    Compare:
      (1) Recovery trained on Gaussian loss only
      (2) Recovery trained with explicit non-Gaussian phase noise

    Evaluation: final NG error vs phase jitter δ
    """

    print("\n[Analysis 4] Training Gaussian-only recovery...")
    p_gauss, _ = train_mm(
        default_input_params,
        default_proc_params,
        eta,
        steps=steps_gauss,
        lr=lr
    )

    print("[Analysis 4] Training NG-aware recovery...")
    p_ng, _ = train_mm_ng(
        default_input_params,
        default_proc_params,
        eta,
        delta_sig=0.3,
        delta_anc=0.18,
        steps=steps_ng,
        lr=lr,
        K=K
    )

    err_gauss = []
    err_ng = []

    for d in deltas:
        e_g = cost_mm_ng(
            p_gauss,
            default_input_params,
            default_proc_params,
            eta,
            delta_sig=d,
            delta_anc=0.6 * d,
            K=K
        )
        e_n = cost_mm_ng(
            p_ng,
            default_input_params,
            default_proc_params,
            eta,
            delta_sig=d,
            delta_anc=0.6 * d,
            K=K
        )
        err_gauss.append(float(e_g))
        err_ng.append(float(e_n))

    err_gauss = np.array(err_gauss)
    err_ng = np.array(err_ng)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(deltas, err_gauss, "o-", label="Gaussian-trained")
    plt.plot(deltas, err_ng, "s-", label="NG-trained")
    plt.xlabel("Phase jitter δ (rad)")
    plt.ylabel("Final error")
    plt.title("Generalization under non-Gaussian phase noise")
    plt.legend()
    plt.tight_layout()
    savefig("analysis4_ng_generalization")
    plt.show()

    save_log(
        "analysis4_ng_generalization.csv",
        np.column_stack([deltas, err_gauss, err_ng]),
        header="delta,err_gauss_trained,err_ng_trained"
    )

    return deltas, err_gauss, err_ng

analysis4_ng_generalization()
#%%
# ============================================================
# ANALYSIS 5  — Phase-noise critical threshold detection
# ============================================================

def find_critical_delta(
    eta=0.55,
    delta_grid=None,
    criterion="ratio",      # "ratio" or "absolute"
    ratio_thresh=0.90,      # mitigated must be <= ratio_thresh * baseline
    abs_thresh=None,        # mitigated must be <= abs_thresh
    steps_train=40,
    lr=0.06,
    K=MC_SAMPLES,
    train_delta=0.25,
):
    """
    Finds the smallest delta where mitigation 'fails' under NG noise.

    Baseline: params = zeros (no recovery)
    Mitigated: params trained under NG at train_delta (signal) and 0.6*train_delta (ancilla)

    Failure criterion:
      - ratio: mitigated_error > ratio_thresh * baseline_error
      - absolute: mitigated_error > abs_thresh
    """
    if delta_grid is None:
        delta_grid = np.linspace(0.0, 0.8, 17)

    ip, pp = default_input_params, default_proc_params

    # Train NG-aware recovery once
    params_star, _ = train_mm_ng(
        ip, pp, eta,
        delta_sig=train_delta,
        delta_anc=0.6 * train_delta,
        steps=steps_train,
        lr=lr,
        K=K
    )

    baseline = []
    mitigated = []
    fail_flags = []

    for d in delta_grid:
        e_base = cost_mm_ng(
            np.zeros(6), ip, pp, eta,
            delta_sig=d, delta_anc=0.6 * d,
            K=K
        )
        e_mit = cost_mm_ng(
            params_star, ip, pp, eta,
            delta_sig=d, delta_anc=0.6 * d,
            K=K
        )

        e_base_f = float(e_base)
        e_mit_f = float(e_mit)

        baseline.append(e_base_f)
        mitigated.append(e_mit_f)

        if criterion == "ratio":
            fail = e_mit_f > ratio_thresh * e_base_f
        elif criterion == "absolute":
            if abs_thresh is None:
                raise ValueError("abs_thresh must be provided when criterion='absolute'")
            fail = e_mit_f > abs_thresh
        else:
            raise ValueError("criterion must be 'ratio' or 'absolute'")

        fail_flags.append(fail)

    baseline = np.array(baseline)
    mitigated = np.array(mitigated)
    fail_flags = np.array(fail_flags)

    # Detect critical delta (first failure)
    idx = np.where(fail_flags)[0]
    dcrit = float(delta_grid[idx[0]]) if len(idx) > 0 else None

    # Plot
    plt.figure(figsize=(7,4))
    plt.plot(delta_grid, baseline, "o-", label="Baseline (no recovery)")
    plt.plot(delta_grid, mitigated, "s-", label="Mitigated (NG-trained)")
    if dcrit is not None:
        plt.axvline(dcrit, linestyle="--")
        plt.text(dcrit, max(baseline.max(), mitigated.max())*0.9, f" δ*={dcrit:.2f}", rotation=90)
    plt.xlabel("Phase jitter δ (rad)")
    plt.ylabel("Error")
    plt.title(f"Critical phase-noise threshold (criterion={criterion})")
    plt.legend()
    plt.tight_layout()
    savefig("analysisA_critical_delta")
    plt.show()

    save_log(
        "analysisA_critical_delta.csv",
        np.column_stack([delta_grid, baseline, mitigated, fail_flags.astype(int)]),
        header="delta,baseline,mitigated,fail_flag"
    )

    print("Critical delta δ* =", dcrit)
    return dcrit, delta_grid, baseline, mitigated, fail_flags

find_critical_delta()
#%%
# ============================================================
# ANALYSIS 6 — Sensitivity scaling vs number of modes
# ============================================================

def build_devices_and_layers_for_modes(M):
    """
    Modes:
      0 = signal
      1..M-2 = ancillas (if any)
      M-1 = environment for signal loss

    Recovery: local phase+displacement on signal and all ancillas (excluding env)
    """
    dev = qml.device("default.gaussian", wires=M)

    def prepare_input(ip):
        # ip = [r_s, phi_s, alpha_s, r_a, phi_a] used only for signal+first ancilla
        r_s, phi_s, alpha_s, r_a, phi_a = ip
        qml.Squeezing(r_s, phi_s, wires=0)
        qml.Displacement(alpha_s, 0.0, wires=0)
        if M >= 3:
            qml.Squeezing(r_a, phi_a, wires=1)  # first ancilla
        # any extra ancillas beyond wire 1 remain vacuum

    def entangle(pp):
        # Entangle signal with each ancilla using same beamsplitter params
        theta, phi = pp
        for a in range(1, M-1):
            qml.Beamsplitter(theta, phi, wires=[0, a])

    def loss(eta):
        theta = np.arccos(np.sqrt(eta))
        qml.Beamsplitter(theta, 0.0, wires=[0, M-1])

    def recovery(params):
        # params length = 3*(M-1): [phi0, d0r, d0i, phi1, d1r, d1i, ...] for wires 0..M-2
        assert len(params) == 3*(M-1)
        for w in range(M-1):
            phi = params[3*w + 0]
            dr  = params[3*w + 1]
            di  = params[3*w + 2]
            qml.Rotation(phi, wires=w)
            qml.Displacement(dr + 1j*di, 0.0, wires=w)

    @qml.qnode(dev, interface="autograd")
    def noisy_x(ip, pp, eta, params):
        prepare_input(ip)
        entangle(pp)
        loss(eta)
        recovery(params)
        return qml.expval(qml.QuadX(0))

    @qml.qnode(dev, interface="autograd")
    def noisy_p(ip, pp, eta, params):
        prepare_input(ip)
        entangle(pp)
        loss(eta)
        recovery(params)
        return qml.expval(qml.QuadP(0))

    @qml.qnode(dev)
    def ideal_x(ip, pp):
        prepare_input(ip)
        entangle(pp)
        return qml.expval(qml.QuadX(0))

    @qml.qnode(dev)
    def ideal_p(ip, pp):
        prepare_input(ip)
        entangle(pp)
        return qml.expval(qml.QuadP(0))

    def cost(params, ip, pp, eta):
        return (ideal_x(ip, pp) - noisy_x(ip, pp, eta, params))**2 + (ideal_p(ip, pp) - noisy_p(ip, pp, eta, params))**2

    def train(ip, pp, eta, steps=40, lr=0.06):
        opt = qml.GradientDescentOptimizer(lr)
        params = np.zeros(3*(M-1), requires_grad=True)
        hist = []
        for _ in range(steps):
            params, loss_val = opt.step_and_cost(lambda p: cost(p, ip, pp, eta), params)
            hist.append(loss_val)
        return params, np.array(hist)

    return cost, train, M


def analysisB_sensitivity_vs_modes(
    modes_list=(2,3,4,5),
    eta=0.55,
    steps=40,
    lr=0.06
):
    """
    Measures how final mitigated error scales as we add more ancilla modes.
    M=2 corresponds to single-mode+env (signal=0, env=1) with no ancilla.
    """
    ip, pp = default_input_params, default_proc_params

    finals_base = []
    finals_mit  = []

    for M in modes_list:
        cost, train, _ = build_devices_and_layers_for_modes(M)

        p_star, hist = train(ip, pp, eta, steps=steps, lr=lr)

        e_base = float(cost(np.zeros(3*(M-1)), ip, pp, eta))
        e_mit  = float(cost(p_star, ip, pp, eta))

        finals_base.append(e_base)
        finals_mit.append(e_mit)

        print(f"M={M}: base={e_base:.4e}  mit={e_mit:.4e}")

    finals_base = np.array(finals_base)
    finals_mit  = np.array(finals_mit)

    plt.figure(figsize=(6,4))
    plt.plot(modes_list, finals_base, "o-", label="Baseline")
    plt.plot(modes_list, finals_mit, "s-", label="Mitigated")
    plt.xlabel("Total modes M (includes env)")
    plt.ylabel("Final error")
    plt.title("Sensitivity scaling vs number of modes")
    plt.legend()
    plt.tight_layout()
    savefig("analysisB_sensitivity_vs_modes")
    plt.show()

    save_log(
        "analysisB_sensitivity_vs_modes.csv",
        np.column_stack([np.array(modes_list), finals_base, finals_mit]),
        header="M,baseline,mitigated"
    )

    return np.array(modes_list), finals_base, finals_mit

analysisB_sensitivity_vs_modes()
#%%
# ============================================================
# ANALYSIS 7 — Recovery parameter drift analysis
# ============================================================

def analysisC_parameter_drift(
    eta=0.55,
    delta=0.30,
    steps=60,
    lr=0.06,
    K=MC_SAMPLES,
    seed=NG_SEED
):
    """
    Tracks how recovery parameters evolve during NG training.
    Outputs:
      - L2 norm of params vs step
      - per-parameter trajectories (optional heavy plot)
    """
    ip, pp = default_input_params, default_proc_params
    opt = qml.GradientDescentOptimizer(lr)
    params = np.zeros(6, requires_grad=True)

    hist_loss = []
    hist_norm = []
    hist_params = []

    for _ in range(steps):
        params, loss = opt.step_and_cost(
            lambda p: cost_mm_ng(
                p, ip, pp, eta,
                delta_sig=delta,
                delta_anc=0.6*delta,
                K=K,
                seed=seed
            ),
            params
        )
        hist_loss.append(loss)
        hist_params.append(params)
        hist_norm.append(np.linalg.norm(params))

    hist_loss = np.array(hist_loss, dtype=float)
    hist_norm = np.array(hist_norm, dtype=float)
    hist_params = np.array(hist_params)

    # Plot loss + norm
    plt.figure(figsize=(7,4))
    plt.plot(hist_loss, label="Loss")
    plt.plot(hist_norm, label="||params||₂")
    plt.yscale("log")
    plt.xlabel("Step")
    plt.title("Parameter drift under NG training")
    plt.legend()
    plt.tight_layout()
    savefig("analysisC_param_drift_loss_norm")
    plt.show()

    save_log("analysisC_param_drift_loss.csv", hist_loss.reshape(-1,1), header="loss")
    save_log("analysisC_param_drift_norm.csv", hist_norm.reshape(-1,1), header="param_l2_norm")
    save_log("analysisC_param_drift_params.csv", hist_params, header="phi0,d0r,d0i,phi1,d1r,d1i")

    # Optional: plot each parameter trajectory (can be visually busy)
    plt.figure(figsize=(8,4))
    for k in range(6):
        plt.plot(hist_params[:, k], label=f"p{k}")
    plt.xlabel("Step")
    plt.title("Recovery parameter trajectories (NG training)")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    savefig("analysisC_param_drift_params")
    plt.show()

    return params, hist_loss, hist_norm, hist_params

analysisC_parameter_drift()
#%%
# ============================================================
# ANALYSIS 8 — Runtime / scaling benchmarks
# ============================================================

import time

def analysisD_runtime_benchmarks(
    eta=0.55,
    steps=20,
    lr=0.06,
    K_list=(4, 8, 16, 32),
    seed=NG_SEED
):
    """
    Measures wall-clock cost of:
      - one Gaussian training run (multi-mode)
      - one NG training run (multi-mode) as function of MC samples K

    NOTE: timings depend strongly on your CPU and notebook state.
    """
    ip, pp = default_input_params, default_proc_params

    # Gaussian baseline timing
    t0 = time.perf_counter()
    _ = train_mm(ip, pp, eta, steps=steps, lr=lr)
    t_gauss = time.perf_counter() - t0

    rows = []
    print(f"Gaussian train runtime (steps={steps}): {t_gauss:.3f} s")

    for K in K_list:
        t0 = time.perf_counter()
        _ = train_mm_ng(ip, pp, eta, delta_sig=0.3, delta_anc=0.18, steps=steps, lr=lr, K=K)
        t_ng = time.perf_counter() - t0
        rows.append([K, t_ng, t_ng / t_gauss if t_gauss > 0 else np.nan])
        print(f"NG train runtime K={K} (steps={steps}): {t_ng:.3f} s  (x{t_ng/t_gauss:.2f})")

    rows = np.array(rows, dtype=float)

    plt.figure(figsize=(6,4))
    plt.plot(rows[:,0], rows[:,1], "o-")
    plt.xlabel("MC samples K")
    plt.ylabel("NG training runtime (s)")
    plt.title(f"Runtime scaling vs K (steps={steps})")
    plt.tight_layout()
    savefig("analysisD_runtime_vs_K")
    plt.show()

    save_log(
        "analysisD_runtime_benchmarks.csv",
        rows,
        header="K,time_seconds,time_ratio_vs_gauss"
    )

    return t_gauss, rows

# ============================================================
# ANALYSIS 8 — Runtime / scaling benchmarks
# ============================================================

import time

def analysisD_runtime_benchmarks(
    eta=0.55,
    steps=20,
    lr=0.06,
    K_list=(4, 8, 16, 32),
    seed=NG_SEED
):
    """
    Measures wall-clock cost of:
      - one Gaussian training run (multi-mode)
      - one NG training run (multi-mode) as function of MC samples K

    NOTE: timings depend strongly on your CPU and notebook state.
    """
    ip, pp = default_input_params, default_proc_params

    # Gaussian baseline timing
    t0 = time.perf_counter()
    _ = train_mm(ip, pp, eta, steps=steps, lr=lr)
    t_gauss = time.perf_counter() - t0

    rows = []
    print(f"Gaussian train runtime (steps={steps}): {t_gauss:.3f} s")

    for K in K_list:
        t0 = time.perf_counter()
        _ = train_mm_ng(ip, pp, eta, delta_sig=0.3, delta_anc=0.18, steps=steps, lr=lr, K=K)
        t_ng = time.perf_counter() - t0
        rows.append([K, t_ng, t_ng / t_gauss if t_gauss > 0 else np.nan])
        print(f"NG train runtime K={K} (steps={steps}): {t_ng:.3f} s  (x{t_ng/t_gauss:.2f})")

    rows = np.array(rows, dtype=float)

    plt.figure(figsize=(6,4))
    plt.plot(rows[:,0], rows[:,1], "o-")
    plt.xlabel("MC samples K")
    plt.ylabel("NG training runtime (s)")
    plt.title(f"Runtime scaling vs K (steps={steps})")
    plt.tight_layout()
    savefig("analysisD_runtime_vs_K")
    plt.show()

    save_log(
        "analysisD_runtime_benchmarks.csv",
        rows,
        header="K,time_seconds,time_ratio_vs_gauss"
    )

    return t_gauss, rows

