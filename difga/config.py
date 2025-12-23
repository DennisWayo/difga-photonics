from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# GLOBAL CONFIG
# ============================================================

plt.rcParams["figure.dpi"] = 140
plt.rcParams["axes.grid"] = True

FIG_DIR = Path("figures")
LOG_DIR = Path("logs")
FIG_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# DEFAULT PARAMETERS
# ============================================================

# input params: [r_s, phi_s, alpha_s, r_a, phi_a]
DEFAULT_INPUT_PARAMS = [0.6, 0.3, 0.8, 0.4, 0.1]
# proc params: [theta_bs, phi_bs]
DEFAULT_PROC_PARAMS = [0.7, 0.2]

# ============================================================
# NON-GAUSSIAN PHASE NOISE (Monte-Carlo)
# ============================================================

NG_SEED = 123
MC_SAMPLES = 32
DELTA_SIG_DEFAULT = 0.20
DELTA_ANC_DEFAULT = 0.12
EPS = 1e-12