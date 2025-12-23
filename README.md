# DifGa

**DifGa** is a fully differentiable, Gaussian-only error mitigation framework for continuous-variable photonic quantum circuits.

Unlike bosonic quantum error correction schemes based on non-Gaussian
code states (e.g., GKP), DifGa operates entirely at the level of
quadrature observables and uses only Gaussian operations, vacuum ancillas,
and gradient-based optimization.

## Key Features
- Multi-mode Gaussian architectures
- Trainable Gaussian recovery layers
- Gaussian loss + weak non-Gaussian phase noise
- End-to-end differentiability via PennyLane
- Hardware-compatible with near-term photonic platforms

## Installation
```bash
pip install -r requirements.txt
