import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

from difga.config import FIG_DIR, LOG_DIR

def savefig(name: str, dpi: int = 300):
    plt.savefig(FIG_DIR / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    print(f"[Saved] figures/{name}.png, figures/{name}.pdf")

def save_log(name: str, data, header: str | None = None):
    path = LOG_DIR / name
    np.savetxt(
        path,
        np.asarray(data, dtype=float),
        delimiter=",",
        header=(header or ""),
        comments="",
    )
    print(f"[Saved log] {path}")

def print_versions():
    print("PennyLane version:", qml.__version__)