import numpy as onp
from pennylane import numpy as np
import pennylane as qml

from difga.config import NG_SEED, MC_SAMPLES

def make_phase_samples(delta_sig, delta_anc, K: int = MC_SAMPLES, seed: int = NG_SEED):
    rng = onp.random.RandomState(seed)
    eps_sig = rng.normal(0.0, float(delta_sig), K)
    eps_anc = rng.normal(0.0, float(delta_anc), K)
    return eps_sig, eps_anc

def phase_noise_layer_mm(eps_sig, eps_anc):
    qml.Rotation(eps_sig, wires=0)
    qml.Rotation(eps_anc, wires=1)