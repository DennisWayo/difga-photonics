import pennylane as qml
from pennylane import numpy as np

from difga.config import DEFAULT_INPUT_PARAMS, DEFAULT_PROC_PARAMS
from difga.recovery import recovery_layer_mm, recovery_layer_sm
from difga.noise import phase_noise_layer_mm

# ============================================================
# DEVICES
# ============================================================

dev_mm = qml.device("default.gaussian", wires=3)  # signal, ancilla, environment
dev_sm = qml.device("default.gaussian", wires=2)  # signal, environment

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

# ============================================================
# SINGLE-MODE BUILDING BLOCKS
# ============================================================

def prepare_input_state_sm(r, phi, alpha):
    qml.Squeezing(r, phi, wires=0)
    qml.Displacement(alpha, 0.0, wires=0)

def loss_on_signal_sm(eta):
    theta = np.arccos(np.sqrt(eta))
    qml.Beamsplitter(theta, 0.0, wires=[0, 1])

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
# NOISY QNODES (Gaussian loss only)
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

# ============================================================
# NOISY QNODES (Gaussian loss + phase jitter sample)
# ============================================================

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
# DRAW (Appendix)
# ============================================================

def draw_circuits():
    ip = np.array(DEFAULT_INPUT_PARAMS)
    pp = np.array(DEFAULT_PROC_PARAMS)
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