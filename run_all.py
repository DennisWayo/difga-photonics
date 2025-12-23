from difga.circuits import *
from difga.noise import *
from difga.recovery import *
from difga.cost import *
from difga.training import *
# run_all.py

from difga.circuits import draw_circuits
from difga.training import (
    analysis1_noise_sweep,
    analysis2_single_vs_multi,
    analysis3_phase_diagram_ng,
)

if __name__ == "__main__":
    draw_circuits()
    analysis1_noise_sweep()
    analysis2_single_vs_multi()
    analysis3_phase_diagram_ng()