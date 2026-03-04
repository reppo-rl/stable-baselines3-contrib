# REPPO Effectiveness Report

**Algorithm:** REPPO (Relative Entropy Pathwise Policy Optimization)
**Baseline:** PPO (Proximal Policy Optimization, SB3 default)
**Hardware:** MacBook (Apple Silicon), CPU only
**Date:** 2026-02-23

---

## Summary

| Environment | REPPO | PPO | Steps | Verdict |
|---|---|---|---|---|
| Pendulum-v1 | **-840 ± 101** | -1222 ± 507 | 200k | REPPO wins, neither solved |
| MountainCarContinuous-v0 | -0.0 ± 0.0 | -51.9 ± 73.5 | 300k | Both fail; REPPO collapses |
| BipedalWalker-v3 | **241.6 ± 2.0** | -110.8 ± 1.3 | 1M | REPPO wins clearly |

---

## Environment 1: Pendulum-v1 (easy)

**Spec:** 3D obs · 1D action · dense reward · max ~-120/episode
**Budget:** 200k steps · 4 envs · ~3 min

| | Mean return | Std |
|---|---|---|
| REPPO | **-840** | 101 |
| PPO | -1222 | 507 |

REPPO learns faster and is much more stable (std 101 vs 507). Neither agent is
fully converged at 200k steps — the PPO zoo result for Pendulum uses ~1M steps
with a tuned hyperparameter set. The lower std for REPPO reflects the entropy
regularization keeping the policy from collapsing into bad local optima, which
is what drives PPO's high variance here.

**Key observation:** The distributional critic (`vmin=-1600, vmax=0`) fits
Pendulum's per-episode return range, and entropy regularization helps REPPO
explore more systematically than PPO's clip-based updates.

---

## Environment 2: MountainCarContinuous-v0 (sparse reward)

**Spec:** 2D obs · 1D action · very sparse reward (+100 only at goal) · hard exploration
**Budget:** 300k steps · 4 envs · ~8 min total

| | Mean return | Std |
|---|---|---|
| REPPO | -0.0 | 0.0 |
| PPO | -51.9 | 73.5 |

**REPPO fails here.** The -0.0 return sounds deceptively good — it is not.
The agent learned to output near-zero actions (mean action ≈ -0.01, std ≈ 0.001),
incurring almost no step penalty (`-0.1 · a²`) but never moving the car or
reaching the goal.

This is an **entropy trap**: REPPO's temperature parameter finds a local
optimum where "do nothing" maximises the entropy-regularised objective.
The sparse reward gives no gradient signal to escape it.

PPO at -51.9 is also failing to solve the task, but it at least explores
randomly and occasionally climbs partway up the hill — some episodes end at
~+90 (the high variance reflects this).

**Root cause:** Entropy regularisation + sparse rewards is a known failure
mode. The algorithm needs either reward shaping, intrinsic motivation (e.g.
count-based or curiosity), or a much longer training budget so the temperature
anneals enough for the policy to commit to directed actions.

---

## Environment 3: BipedalWalker-v3 (hard locomotion)

**Spec:** 24D obs · 4D action · dense reward · solve threshold 300
**Budget:** 1M steps · 4 envs · ~19 min

| | Mean return | Std |
|---|---|---|
| REPPO | **241.6** | 2.0 |
| PPO | -110.8 | 1.3 |

REPPO reaches **241.6/300** — the robot walks the full course consistently but
loses points on gait efficiency. PPO falls over immediately throughout training
with this hyperparameter configuration (PPO *can* solve BipedalWalker but
typically needs careful reward normalisation, learning rate scheduling, and
~5M steps in the SB3 zoo).

The near-zero std (2.0) shows REPPO has converged to a stable walking policy,
not a lucky average. The KL-divergence constraint and separate actor/critic
optimisers appear to produce a smooth, reliable gait.

**Sample efficiency:** The SB3 PPO zoo solves BipedalWalker in ~5M steps with
a tuned config. REPPO reaches 80% of the solve threshold in 1M steps with a
default config — suggesting meaningfully better sample efficiency on dense
reward locomotion tasks.

---

## Where REPPO Excels

- **Dense reward, multi-dimensional action spaces** (BipedalWalker, and likely
  MuJoCo locomotion tasks like HalfCheetah, Hopper, Walker2d)
- **Stable convergence** — the separate actor/critic updates and KL constraint
  prevent the policy from collapsing during training
- **Sample efficiency** — distributional value estimation (HL-Gauss) + entropy
  regularisation gives more useful gradients early in training

## Where REPPO Struggles

- **Sparse reward environments** — entropy regularisation creates a local
  optimum of "high-entropy, low-magnitude actions" that the sparse reward
  cannot dislodge
- **Slow wall-clock time** — REPPO is ~2–3× slower per step than PPO (789s vs
  367s for 1M BipedalWalker steps) due to the distributional critic forward
  pass, auxiliary self-prediction loss, and Monte Carlo KL estimation

## Recommendations

| Situation | Recommendation |
|---|---|
| Dense reward, continuous control | Use REPPO |
| Sparse reward | Add reward shaping or intrinsic motivation before REPPO |
| Time-constrained / simple env | PPO is faster per step and simpler to tune |
| Need > 300 on BipedalWalker | Continue training to ~1.5–2M steps |
