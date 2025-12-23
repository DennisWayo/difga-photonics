![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PennyLane](https://img.shields.io/badge/Powered%20by-PennyLane-8A2BE2)
![Quantum](https://img.shields.io/badge/Quantum-Continuous--Variable-green)
![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen)
![Last Commit](https://img.shields.io/github/last-commit/DennisWayo/difga-photonics)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

## DifGa–photonics

**DifGa** is a fully differentiable, Gaussian-only error mitigation framework for
continuous-variable (CV) photonic quantum circuits.

Unlike bosonic quantum error correction schemes based on non-Gaussian code
states (e.g., GKP), DifGa operates entirely at the level of *quadrature
observables* and uses only Gaussian operations, vacuum ancillas, and
gradient-based optimization. The framework is designed to be compatible with
near-term integrated photonic hardware and modern differentiable quantum
software stacks.



### Motivation

Photonic quantum computers based on Gaussian continuous-variable operations
offer scalability, room-temperature operation, and high-bandwidth optical
interconnects. However, their performance is fundamentally limited by optical
loss, phase drift, and other noise sources.

Conventional bosonic quantum error correction circumvents Gaussian no-go
theorems using highly non-Gaussian resources such as Gottesman–Kitaev–Preskill
(GKP) states, which require large squeezing, modular measurements, and complex
state preparation. These requirements are beyond the reach of current
large-scale photonic platforms.

DifGa addresses a *different problem*: **hardware-compatible error mitigation for expectation values**, rather than
fault-tolerant logical quantum memory. By remaining entirely within the
Gaussian regime, DifGa enables efficient simulation, differentiable training,
and immediate experimental relevance.


### Core Features

- **Gaussian-only framework**  
  No non-Gaussian states, measurements, or resources required.

- **Multi-mode Gaussian architectures**  
  Exploits ancillary modes to redistribute noise and suppress quadrature errors.

- **Trainable Gaussian recovery layers**  
  Physically realizable displacements and phase-space rotations optimized
  end-to-end.

- **Gaussian loss + weak non-Gaussian phase noise**  
  Supports realistic optical loss and differentiable Monte Carlo modeling of
  phase jitter.

- **End-to-end differentiability**  
  Implemented using PennyLane’s `default.gaussian` backend with automatic
  differentiation.

- **Hardware-compatible**  
  Requires only linear optics, squeezing, vacuum ancillas, and homodyne
  detection.

---

### What DifGa Is — and Is Not

#### DifGa **is**
- An **observable-level error mitigation** method
- A **software-native**, differentiable approach
- Designed for **near-term photonic hardware**
- Compatible with **Gaussian simulators and experiments**

#### DifGa **is not**
-  Fault-tolerant quantum error correction  
-  A logical encoding scheme  
-  A replacement for GKP or bosonic QEC  
-  A non-Gaussian protocol  

DifGa and GKP-based methods address **complementary regimes** rather than
competing objectives.

---

### Architecture Overview

At a high level, DifGa consists of:

1. **Ideal Gaussian encoding**  
   Multi-mode Gaussian resource preparation (signal + ancilla).

2. **Noise model**
   - Gaussian loss modeled by beam-splitter coupling to an environment mode
   - Weak non-Gaussian phase noise modeled via differentiable Monte Carlo
     mixtures

3. **Trainable Gaussian recovery**
   - Local phase rotations and displacements
   - Parameters optimized using gradient-based methods

4. **Cost function**
   - Mean-squared error between ideal and noisy quadrature expectations

The full architecture and its contrast with GKP-based logical recovery are
illustrated in the accompanying manuscript figures.

<img width="2672" height="1204" alt="image3" src="https://github.com/user-attachments/assets/b54aa3b4-81e3-48d1-b8da-44ae8b2f2dd5" />

---

### Installation

Clone the repository:
```bash
git clone https://github.com/DennisWayo/difga-photonics.git
cd difga-photonics
```

### Acknowledgments

DifGa makes extensive use of the open-source PennyLane framework developed
by Xanadu for differentiable quantum programming. We gratefully acknowledge
the Xanadu team for providing robust continuous-variable simulation tools and
automatic differentiation support.
