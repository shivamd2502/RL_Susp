# Comparative Deep Reinforcement Learning Control of Active Suspension Systems Using DDPG and SAC



## Overview

This repository contains the implementation and evaluation code for a comparative study of two deep reinforcement learning algorithms — **Deep Deterministic Policy Gradient (DDPG)** and **Soft Actor-Critic (SAC)** — applied to physics-guided active vehicle suspension control.

The study builds on the physics-guided DDPG framework of Nhu et al. (IEEE ICMLA 2023) and makes the following contributions:

- Implements SAC as a novel algorithmic alternative in the physics-guided suspension framework
- Conducts the **first multi-class generalisation study** across ISO 8608 road roughness classes B, C, D, and E
- Demonstrates that SAC's entropy-regularised stochastic policy consistently and significantly outperforms DDPG's deterministic policy across all road classes and metrics

---

## Results Summary

All metrics are percentage change relative to the passive suspension baseline (negative = improvement), averaged over 10 independent test runs per road class.

| Road Class | Algorithm | Mean Vel.% | Q3 Vel.% | Mean Acc.% | Q3 Acc.% |
|---|---|---|---|---|---|
| B — Good highway | DDPG | −7.17 | −5.31 | **+14.16** | +15.81 |
| | SAC | −17.17 | −11.18 | −8.27 | −7.95 |
| C — Average road | DDPG | −12.32 | −10.82 | **+5.34** | +8.18 |
| | SAC | −25.02 | −25.45 | −12.83 | −13.09 |
| D — Poor road | DDPG | −16.77 | −15.39 | **+1.62** | +2.00 |
| | SAC | −38.92 | −39.80 | −16.53 | −13.69 |
| E — Very rough (training) | DDPG | −5.74 | −3.01 | **+12.17** | +12.82 |
| | SAC | −26.08 | −25.76 | −9.76 | −8.95 |

**Key finding:** DDPG worsens body acceleration on every road class it was evaluated on (positive Acc.% values). SAC achieves improvement on both velocity and acceleration simultaneously across all classes.

---

## System Architecture

### Physical Model

A **2-DOF quarter-car model** simulates one corner of a vehicle:

- **Sprung mass** (car body): $m_b = 450\ \text{kg}$
- **Unsprung mass** (wheel): $m_w = 45\ \text{kg}$
- Passive spring: $k_b = 15{,}000\ \text{N/m}$, damper: $c_b = 1{,}500\ \text{Ns/m}$
- Tyre stiffness: $k_w = 150{,}000\ \text{N/m}$

Equations of motion:

$$m_b \ddot{x}_b = -k_b(x_b - x_w) - c_b(\dot{x}_b - \dot{x}_w) + f_a$$

$$m_w \ddot{x}_w = k_b(x_b - x_w) + c_b(\dot{x}_b - \dot{x}_w) - k_w(x_w - x_r) - f_a$$

### Physics-Guided Action Space

The RL agent controls two physically meaningful actuator variables rather than an abstract force:

$$f_a = (k_b + k_a)(x_b - x_w) + (c_b + c_a)(\dot{x}_b - \dot{x}_w)$$

- **Active stiffness:** $k_a \in [-2500,\ 5000]\ \text{N/m}$ → maps to hydraulic pressure commands
- **Active damping:** $c_a \in [-600,\ 600]\ \text{Ns/m}$ → maps to current signals for magnetorheological dampers

The ODE system is integrated at each timestep ($\Delta t = 0.01\ \text{s}$) using `scipy.integrate.solve_ivp`.

### State Space

$$s_t = [\dot{x}_b(t),\ \dot{x}_w(t),\ \dot{x}_r(t),\ \dot{x}_b(t{-}1),\ \dot{x}_w(t{-}1),\ \dot{x}_r(t{-}1)] \in \mathbb{R}^6$$

Absolute positions are excluded as their removal empirically improves convergence and generalisation.

### Reward Function

$$r_t = -\frac{1}{10}|\dot{x}_b(t+1)| - \frac{1}{100}|\ddot{x}_b(t+1)|$$

Penalises both body velocity and acceleration, directly targeting comfort metrics evaluated quantitatively.

---

## Network Architectures

| Network | Architecture | Activation |
|---|---|---|
| DDPG Actor | 6 → 16 → 16 → 2 | ReLU + tanh (output) |
| DDPG Critic | 8 → 32 → 32 → 1 | ReLU (linear output) |
| SAC Actor | 6 → 256 → 256 → [2, 2] | ReLU + tanh/clamp |
| SAC Critic ×2 | 8 → 256 → 256 → 1 | ReLU (linear output) |

---

## Training Hyperparameters

| Parameter | DDPG | SAC |
|---|---|---|
| Episodes | 1500 | 1500 |
| Buffer size | 10⁵ | 10⁵ |
| Batch size | 512 | 512 |
| Discount γ | 0.95 | 0.95 |
| Soft update τ | 0.99 | 0.99 |
| Critic lr | 10⁻³ | 3×10⁻⁴ |
| Actor lr | 10⁻⁴ | 3×10⁻⁴ |
| Exploration | OU noise (σ: 0.5→0.05) | Stochastic policy (automatic) |
| Initial α | — | 0.2 |
| Target entropy H* | — | −2 |

Both algorithms are trained **exclusively on Class-E** and evaluated on all four road classes without retraining.

---

## Road Profile Generation

Road profiles are generated per ISO 8608 as a sum of cosine waves with random phases:

$$x_r(x_i) = \sum_{j=1}^{N} \sqrt{G_d(n_j)\Delta n} \cos(2\pi n_j x_i + \phi_j)$$

| Class | $G_d(n_0)$ (m³) | Description |
|---|---|---|
| B | 128 × 10⁻⁶ | Good highway |
| C | 512 × 10⁻⁶ | Average road |
| D | 2048 × 10⁻⁶ | Poor road |
| E | 8192 × 10⁻⁶ | Very rough road (training class) |

Class E is 64× rougher than Class B.

---



## Installation

```bash
git clone https://github.com/<your-username>/drl-active-suspension.git
cd drl-active-suspension
pip install -r requirements.txt
```

**Dependencies:**

```
numpy
scipy
torch
matplotlib
```

---

## Usage

**Train DDPG:**
```bash
python train.py --algo ddpg --episodes 1500 --road-class E --seed 99
```

**Train SAC:**
```bash
python train.py --algo sac --episodes 1500 --road-class E --seed 42
```

**Evaluate across all road classes:**
```bash
python evaluate.py --algo sac --checkpoint checkpoints/sac_best.pt --runs 10
```

---

## Why SAC Outperforms DDPG

DDPG's deterministic policy maps each state to a single action. Once it finds a locally good policy, update gradients become small and learning stagnates — visible as a flat reward curve plateauing near −17.5 from episode 1.

SAC augments the reward with an entropy term:

$$J(\pi) = \mathbb{E}\left[\sum_{t=0}^{T} \gamma^t \left(r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))\right)\right]$$

The entropy bonus prevents policy collapse — the agent is simultaneously rewarded for minimising suspension vibration **and** for maintaining action diversity. On stiff ODE environments like the quarter-car system, where physics create sharp reward landscapes, this sustained exploration is critical for escaping suboptimal equilibria.

Additionally, SAC's smooth, proportional damping adjustments are more compatible with the response bandwidth of real magnetorheological dampers and hydraulic actuators, supporting real-world deployment.

---

## Citation

If you use this code or find this work useful, please cite:

```bibtex
@inproceedings{.......,
  title     = {Comparative Deep Reinforcement Learning Control of Active Suspension Systems Using DDPG and SAC},
  author    = {Mundada, Kapil and Dapkekar, Shivam and Deshpande, Raghav},
  booktitle = {Proceedings of [Conference Name]},
  year      = {........},
  institution = {Vishwakarma Institute of Technology, Pune}
}
```

This work builds on:

> A. N. Nhu et al., "Physics-Guided Reinforcement Learning System for Realistic Vehicle Active Suspension Control," IEEE ICMLA 2023. DOI: 10.1109/ICMLA58977.2023.00065

---
