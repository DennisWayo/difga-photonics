from difga.io_utils import print_versions
from difga.circuits import draw_circuits
from difga.training import (
    analysis1_noise_sweep,
    analysis2_single_vs_multi,
    analysis3_phase_diagram_ng,
    analysis4_ng_generalization,
    find_critical_delta,
    analysisB_sensitivity_vs_modes,
    analysisD_runtime_benchmarks
)

if __name__ == "__main__":
    print_versions()
    draw_circuits()
    analysis1_noise_sweep()
    analysis2_single_vs_multi()
    analysis3_phase_diagram_ng()
    analysis4_ng_generalization()
    find_critical_delta()
    analysisB_sensitivity_vs_modes()
    analysisD_runtime_benchmarks()
