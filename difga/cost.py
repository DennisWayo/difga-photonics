import pennylane as qml
from pennylane import numpy as np

from difga.circuits import (
    ideal_x_mm, ideal_p_mm, noisy_x_mm, noisy_p_mm,
    ideal_x_sm, ideal_p_sm, noisy_x_sm, noisy_p_sm,
    noisy_x_mm_ng, noisy_p_mm_ng,
)
from difga.noise import make_phase_samples
from difga.config import MC_SAMPLES, NG_SEED, DELTA_SIG_DEFAULT, DELTA_ANC_DEFAULT

def mc_avg_noisy_quads_mm(ip, pp, eta, params, delta_sig, delta_anc, K=MC_SAMPLES, seed=NG_SEED):
    eps_sig, eps_anc = make_phase_samples(delta_sig, delta_anc, K, seed)

    xs, ps = [], []
    for k in range(K):
        xs.append(noisy_x_mm_ng(ip, pp, eta, params, eps_sig[k], eps_anc[k]))
        ps.append(noisy_p_mm_ng(ip, pp, eta, params, eps_sig[k], eps_anc[k]))

    xs = qml.math.stack(xs)
    ps = qml.math.stack(ps)
    return qml.math.mean(xs), qml.math.mean(ps)

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
    x_no, p_no = mc_avg_noisy_quads_mm(ip, pp, eta, params, delta_sig, delta_anc, K, seed)
    return (x_id - x_no)**2 + (p_id - p_no)**2