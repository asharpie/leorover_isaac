# Leo Rover Hybrid Residual RL — Complete Project Reference

**Project:** Autonomous waypoint navigation for the Leo Rover on Mars-analog terrain, using hybrid residual reinforcement learning (a model-based LQR baseline plus a bounded PPO residual), migrated from a PyBullet (CPU) simulation to Isaac Sim 4.5 / Isaac Lab (GPU-vectorized).

**Maintainer:** Aaron ("Sharpie"), URCA undergraduate research.

**Status (as of this document):** The Isaac migration is functionally complete and validated. Training, monitoring, and evaluation tooling all work. The current substantive finding is that the bounded PPO residual does not improve on the pure LQR baseline under the settings tried, and the dominant performance limiter is a terrain-independent "stall" failure mode rather than terrain difficulty.

**How to use this document:** It is intentionally exhaustive. Read the Executive Summary (Section 2) for the 5-minute picture. Use the Operational Runbook (Section 20) to actually run things. Use the Configuration Reference (Section 19) and CLI Reference (Section 13) as lookup tables. The Debugging Chronicle (Section 16) explains *why* the code looks the way it does — every non-obvious design decision traces back to a bug documented there. Where a specific numeric value is quoted, treat it as accurate at time of writing but verify against the live `config.py` before relying on it, several values have changed over the project's life and those changes are flagged inline.

---

## Table of Contents

1. [Document Purpose & How to Use It](#1-document-purpose--how-to-use-it)
2. [Executive Summary](#2-executive-summary)
3. [Project Background & Research Goal](#3-project-background--research-goal)
4. [The Two Codebases & Migration Mapping](#4-the-two-codebases--migration-mapping)
5. [System Architecture Overview](#5-system-architecture-overview)
6. [The Simulation Environment (DirectRLEnv)](#6-the-simulation-environment-directrlenv)
7. [The Reward Function](#7-the-reward-function)
8. [The Controllers (LQR Baseline + Residual Composition)](#8-the-controllers-lqr-baseline--residual-composition)
9. [The Rover Asset (URDF to USD, Actuators)](#9-the-rover-asset-urdf-to-usd-actuators)
10. [The Terrain System & ADR Curriculum](#10-the-terrain-system--adr-curriculum)
11. [The RL Algorithm (PPO via rsl_rl)](#11-the-rl-algorithm-ppo-via-rsl_rl)
12. [The Lab Workstation](#12-the-lab-workstation)
13. [The `leo` CLI — Complete Command Reference](#13-the-leo-cli--complete-command-reference)
14. [The Tooling Scripts](#14-the-tooling-scripts)
15. [Metrics, Logging & the CSV Schema](#15-metrics-logging--the-csv-schema)
16. [The Debugging Chronicle](#16-the-debugging-chronicle)
17. [Key Findings & Results](#17-key-findings--results)
18. [Current State of the Project](#18-current-state-of-the-project)
19. [Configuration Reference](#19-configuration-reference)
20. [Operational Runbook](#20-operational-runbook)
21. [Known Issues, Gotchas & Footguns](#21-known-issues-gotchas--footguns)
22. [Open Questions & Recommended Next Steps](#22-open-questions--recommended-next-steps)
23. [Glossary](#23-glossary)
24. [File & Directory Map](#24-file--directory-map)
25. [Appendices](#25-appendices)

---

## 1. Document Purpose & How to Use It

This is the single-source reference for the Leo Rover hybrid residual RL project. It exists because the project accumulated a large amount of hard-won, non-obvious knowledge, much of it bug-driven, that is not discoverable from the code alone. The intent is that a newcomer (or the maintainer six months from now) can pick this up and understand not just *what* the system does but *why* every design choice was made.

Conventions used throughout:

- **Paths** are given relative to the Isaac repo root (`~/leorover_work/leorover_isaac` on the lab box) unless stated otherwise.
- **`leo ...`** refers to the wrapper CLI (`scripts/leo.sh`); prefer it over raw commands.
- **Env vars** of the form `LEOROVER_*` are runtime overrides read by the environment and agent configs.
- **"Per-rover" vs "per-episode" success** is a recurring and important distinction; see Section 17 and Appendix B.
- Values that have changed during the project are written as `OLD -> NEW` with a date where known.

---

## 2. Executive Summary

The Leo Rover is a small four-wheeled differential/skid-drive rover. The research goal is autonomous waypoint-following navigation across procedurally generated Mars-like terrain, learned with **hybrid residual reinforcement learning**: a classical, model-based **LQR controller** produces a baseline steering/velocity command from the path geometry, and a **PPO neural policy** outputs a small, bounded **residual** that is added on top. The premise is that the LQR handles the bulk of the control and the learned residual corrects for what the LQR cannot model (rough terrain, slip, dynamics), giving better sample efficiency and safety than learning from scratch.

The project began as a working **PyBullet** (CPU, single/few environments) simulation and was **migrated to Isaac Sim 4.5 / Isaac Lab** to get GPU-vectorized training across thousands of parallel environments. The migration is complete: the environment, controller, reward, curriculum, and asset were all ported, and a full tooling layer (`leo` CLI, trace/eval/report scripts) was built.

The migration surfaced a long chain of subtle bugs, each of which is documented in Section 16. The most consequential were: a metrics recorder reading state *after* Isaac's automatic per-episode reset (making every CSV look like 4% progress / 0% success regardless of real performance); a reward imbalance where a residual-effort penalty dominated the tiny forward rewards at the rover's crawl speed, making "park and do nothing" the reward-optimal policy; an actuator effort limit that starved the wheel motors; and a series of measurement bugs in a later deterministic-evaluation system.

After those fixes, the substantive results are:

- The reward rebalance took from-scratch training from ~0-4% to ~59% training-time success.
- A clean held-out comparison shows the **hybrid (LQR + residual) is statistically identical to pure LQR at every terrain difficulty**. The bounded PPO residual, as configured, does **not** add value over the baseline.
- Success is **flat across terrain difficulty** (≈57% per-rover from 10% up to 79% intensity), so the limiter is **not** terrain difficulty.
- Failures are dominated by **terrain-independent stalls**: ~15% of rovers park near the start and never get going, and another ~23% stall partway. Once a rover reaches 90% of the path it succeeds ~95% of the time, so goal-capture is not the problem either.
- The reward function is **roughly aligned** with the desired behavior (success earns +282 on average, failure −406, correlation 0.56), but has a real "give-up-early" shape problem: a rover that drives far and fails accrues more total negative reward (−548) than one that parks immediately (−244), which can reinforce parking.

The open question, and the recommended next experiment, is a **reward ablation** to test whether reshaping the reward changes the stall rate, paired with a **direct visualization of the stalls** to see what the parked rovers physically do. Whether the PPO residual can be made to beat the LQR remains unresolved; the current honest reading is that the LQR is near the achievable ceiling for this task under the present formulation.

---

## 3. Project Background & Research Goal

### 3.1 The Leo Rover

The Leo Rover is a compact four-wheel rover (a real, commercially available research platform) modeled here for Mars-analog autonomous navigation. In simulation it is driven as a differential/skid-steer platform: left wheels (front-left, rear-left) and right wheels (front-right, rear-right) are commanded by wheel angular velocities, and turning is achieved by left/right speed differences. The physical wheel radius in the model is **0.0625 m**.

### 3.2 The Research Question — Hybrid Residual RL

The central idea is **residual reinforcement learning** layered on a model-based controller:

- A **trajectory-profiled LQR controller** computes a baseline command `(v_ref-corrected, omega_ref-corrected)` at every control step from the rover's cross-track error, heading error, and velocity error relative to the current path segment. This is a faithful reproduction of the PyBullet project's hand-tuned "Controller2."
- A **PPO policy** observes the rover state (plus the LQR's baseline command and a short camera-style terrain lookahead) and outputs a **2-D residual** in `[-1, 1]`, which is scaled by configured bounds (`max_residual_velocity`, `max_residual_omega`) and **added** to the LQR baseline.
- An **effort penalty** on the residual's magnitude pulls it toward zero unless acting demonstrably improves the reward — so in easy conditions the system behaves like pure LQR, and the learned correction only "switches on" where it helps.

The scientific question is whether this bounded residual measurably improves navigation success/robustness over the LQR baseline alone, especially as terrain difficulty rises. (As of this writing, the answer under the settings tried is "no"; see Sections 17-18.)

### 3.3 Why Migrate PyBullet to Isaac

The original PyBullet simulation worked but was CPU-bound and effectively single-environment, making RL training slow. **Isaac Sim 4.5 / Isaac Lab** provides GPU-accelerated PhysX with thousands of environments stepped in parallel, which is the standard modern stack for this kind of locomotion/navigation RL. The migration goal was a faithful port: same control law, same reward, same curriculum, same observation/action semantics, but vectorized on the GPU so that 4096 environments can train at once. Preserving fidelity was a hard requirement so that results would be comparable to the PyBullet baseline.

---

## 4. The Two Codebases & Migration Mapping

There are two related codebases:

1. **The PyBullet stack (original):** a CPU simulation built around an environment class (`MyEnv2`), a hand-tuned controller (`Controller2`), a Stable-Baselines3 (SB3) training loop with an ADR curriculum callback, and analysis tooling (`evaluate_training.py`, `analyze_training.py`). It produced an `episode_metrics.csv` with a specific column schema.

2. **The Isaac Lab stack (this repo):** a GPU-vectorized reimplementation. The engine-agnostic logic (LQR math, ADR curriculum, trajectory profiling, terrain intensity mapping) was carried over into a `common/` subpackage and reused; the engine-specific parts (environment stepping, terrain generation, asset loading, training loop) were rewritten against Isaac Lab's `DirectRLEnv` and `rsl_rl` PPO.

A deliberate design constraint: the Isaac CSV recorder writes the **identical column schema** as the PyBullet `MetricsCallback`, so the old analysis/plotting tools read Isaac runs unchanged.

High-level mapping (PyBullet → Isaac):

| PyBullet concept | Isaac Lab equivalent |
|---|---|
| `MyEnv2.reset()` | `_reset_idx(env_ids)` (per-env, batched) |
| `Controller2.forward()` | `_pre_physics_step` + `_apply_action` + `VectorizedLQR` |
| 10× `p.stepSimulation()` | physics `decimation` substeps (10 per control step) |
| `_build_observation()` | `_get_observations()` |
| SB3 `ADRCallback` | in-env ADR reporting + (now) a deterministic-eval driver in `train.py` |
| SB3 PPO | `rsl_rl` `OnPolicyRunner` PPO |
| `MetricsCallback` → `episode_metrics.csv` | `EpisodeMetricsRecorder` → same-schema `episode_metrics.csv` |
| VecNormalize | rsl-rl empirical observation normalization |

Known, accepted divergences from PyBullet are documented in the agent-config comments: chiefly the much larger effective batch (4096 envs × 32-step rollout vs PyBullet's `n_steps=4096`), GPU advantage computation, and obs-normalization mechanism.

---

## 5. System Architecture Overview

The end-to-end control/training pipeline, per control step, across all parallel environments:

1. **Observation assembly** (`_get_observations`): the env builds the observation vector for each rover (state + LQR baseline command + terrain lookahead).
2. **Policy forward**: the PPO actor maps observation → a 2-D residual in `[-1, 1]` (during training, sampled stochastically with the policy's action-noise std; during evaluation, the deterministic mean).
3. **LQR baseline** (`VectorizedLQR`): computes the model-based `(v, omega)` baseline from the path geometry and current state.
4. **Composition** (`_pre_physics_step` / `_apply_action`): `total command = LQR baseline + (residual × residual bounds)`, then the differential-drive kinematic mapping converts `(v, omega)` to left/right wheel angular velocities, with acceleration limiting and hardware clipping.
5. **Physics**: PhysX steps the articulation `decimation` (10) substeps at 0.02 s each (0.2 s control step).
6. **Reward** (`_get_rewards`): dense shaping terms + terminal bonus/penalty.
7. **Termination/Done** (`_get_dones`): goal, flip, cross-track-error, stagnation, time-out; this is where the per-episode metric snapshots are taken **before** the auto-reset.
8. **Auto-reset**: Isaac `DirectRLEnv` automatically resets any environment that is `done`, **inside** `step()`. (This auto-reset is the single most important behavioral quirk of the whole stack — see Section 16.)
9. **Curriculum**: terrain difficulty is adjusted by the ADR system; in the current design a periodic deterministic evaluation drives advancement (Section 10.3).
10. **Logging**: `EpisodeMetricsRecorder` appends one CSV row per finished episode.

The training loop (`train.py`) wraps this with `rsl_rl`'s `OnPolicyRunner`, which collects fixed-length rollouts (`num_steps_per_env`) and runs PPO updates.

---

## 6. The Simulation Environment (DirectRLEnv)

### 6.1 Environment Hierarchy

The environment is implemented as a small class hierarchy on Isaac Lab's `DirectRLEnv`:

- **`leo_rover_base_env.py`** — the base environment with all shared logic: waypoint buffers, the LQR controller instance, observation/reward/termination machinery, the ADR curriculum, terrain resampling, the metric snapshots, and the runtime env-var override hooks. This is the largest and most important file.
- **`leo_rover_flat_env.py`** — flat-ground variant (`Isaac-LeoRover-Flat-v0`), used for smoke tests.
- **`leo_rover_mars_env.py`** — Mars terrain, **pure PPO** (no LQR baseline; the policy output *is* the command). Task id `Isaac-LeoRover-Mars-v0`.
- **`leo_rover_mars_hybrid_env.py`** — Mars terrain, **hybrid** (LQR baseline + PPO residual). Task id `Isaac-LeoRover-Mars-Hybrid-v0`. This is the primary research environment.

The hybrid env adds the two LQR baseline commands to the observation and applies the residual-effort penalty (gated by `use_lqr_baseline`).

### 6.2 Observation Space

The hybrid observation is **17-dimensional**: an 11-D base (rover kinematic state, path-relative errors, and the two LQR baseline commands) plus a 6-D forward "camera lookahead" of upcoming terrain slope features. The actor and critic are both 17-input MLPs. (Pure-PPO Mars uses the same shape minus the explicit baseline commands, depending on variant; the network printed at load is `Linear(17 -> 256) -> ReLU -> Linear(256 -> 256) -> ReLU -> Linear(256 -> 2)` for the actor and `... -> 1` for the critic.)

Observations are normalized by rsl-rl's empirical normalization (the modern equivalent of SB3's VecNormalize). This matters for evaluation: a policy must be evaluated through the same normalization it trained with, which is why the inference policy obtained from the runner (`get_inference_policy`) is used rather than calling the raw network.

### 6.3 Action Space

The action is a **2-D residual** in `[-1, 1]` (velocity residual, omega residual). It is scaled by `max_residual_velocity` and `max_residual_omega` (config defaults 0.15 and 0.30 respectively) and then multiplied by a runtime `LEOROVER_RES_SCALE` factor (default **0.33** as of 2026-06-24; see Sections 16-17 for why). At `RES_SCALE = 0`, the residual is forced to zero and the system is pure LQR. For pure-PPO tasks the action is the full command, not a residual.

### 6.4 Episode Lifecycle (Reset, Step, Auto-Reset)

- **Reset** (`_reset_idx`): assigns a fresh random path from a precomputed path "bank", places the rover at the path start at a small spawn clearance (`SPAWN_CLEARANCE`) above rest height (PyBullet used a +0.5 m drop and a 2000-step low-gravity settle; that per-env settle is unaffordable across thousands of envs, so the Isaac port spawns at small clearance and lets PhysX settle within the first step), assigns a terrain patch (random difficulty row up to the ADR ceiling, random column variant), and resets all per-env runtime buffers and the controller's internal state.
- **Step**: assemble obs → policy → LQR → compose → physics decimation → reward → dones → auto-reset.
- **Auto-reset**: `DirectRLEnv` resets any `done` environment **inside** `step()`. The consequence is that *any* state read *after* `step()` returns reflects the *respawned* episode, not the one that just finished. All per-episode metric reads must therefore happen on a snapshot taken inside `_get_dones`, before the reset. This is the root cause of multiple bugs (Section 16).

### 6.5 Termination Conditions

An episode ends (`_get_dones`) when any of:

- **Goal reached** — within `goal_tol = 0.2 m` of the final waypoint. (Success.)
- **Flip** — body-up gravity component below threshold (`flip_threshold_gz = 0.5`).
- **Cross-track error too large** — `|cte|` exceeds `ppo_max_cte_termination` (drove off the path).
- **Stagnation** — forward speed below `STAGNATION_VEL_THRESHOLD = 0.02 m/s` for `STAGNATION_TERMINATION_STEPS = 600` consecutive steps. This is the mechanism that kills "parked" rovers around step ~600-700.
- **Out of time / truncation** — `MAX_EPISODE_STEPS = 2000` steps (= 400 s at the 0.2 s control step).

`_get_dones` also snapshots `self._log_progress = self._path_progress()` and `self._log_goal = goal` *before* the auto-reset; the recorder reads these snapshots, not live state.

---

## 7. The Reward Function

The per-step reward (in `_get_rewards`) is a sum of dense shaping terms plus terminal events. The terms (names follow the `_ppo` weight dictionary):

| Term | Formula (per step) | Weight key | Config default |
|---|---|---|---|
| Cross-track error | `r_cte = -w_cte * cte²` | `ppo_w_cte` | `PPO_W_CTE = 5` |
| Heading | `r_head = w_head * (heading alignment)` | `ppo_w_heading` | `PPO_W_HEADING = 0.5` |
| Forward velocity | `r_vel = w_vel * vel_scale * fwd_vel` | `ppo_w_velocity` | `PPO_W_VELOCITY = 0.5` |
| Smoothness | `r_smooth = -w_smooth * (command change)` | `ppo_w_smoothness` | `PPO_W_SMOOTHNESS = 0.5` |
| Progress | `r_prog = w_prog * (Δprogress / 100)` | `ppo_w_progress` | `PPO_W_PROGRESS = 10` |
| Residual effort (hybrid only) | `r_eff = -ppo_w_effort * (rvn² + rwn²)` | `ppo_w_effort` | `PPO_W_EFFORT = 0.5` |
| Alive | `r_alive = w_alive` | `ppo_w_alive` | `PPO_W_ALIVE = 0` |
| Terminal success | `+PPO_SUCCESS_BONUS` | — | `200` |
| Terminal failure | `-PPO_FAILURE_PENALTY` | — | `50` |

where `rvn`, `rwn` are the residual velocity/omega norms (residual magnitude as a fraction of its bound), and `cte` is the cross-track error in metres.

**The validated reward fix (`FIX_ENV`).** The `config.py` defaults above are the *stock* values. The `leo` CLI applies a validated override set on every non-`--raw` run, because the stock values cause a training collapse at the rover's crawl speed (see Section 16.4). The override is:

```
LEOROVER_W_EFFORT=0.05    # was 0.5 — the dominant cause of the park-and-die collapse
LEOROVER_W_PROGRESS=150   # was 10  — make forward progress the dominant signal
LEOROVER_W_SMOOTH=0.1     # was 0.5
LEOROVER_ENT_COEF=0.001   # entropy coefficient
```

These are exposed as runtime env-var overrides (`LEOROVER_W_PROGRESS`, `LEOROVER_W_CTE`, `LEOROVER_W_VELOCITY`, `LEOROVER_W_ALIVE`, `LEOROVER_W_EFFORT`, `LEOROVER_W_SMOOTH`, `LEOROVER_W_HEADING`, `LEOROVER_CTE_OK`) read by an override loop in `leo_rover_base_env.py`, so the reward can be retuned without editing code.

**Reward alignment (measured).** On a real run's data, total episode reward separates success from failure cleanly (success mean ≈ +282, failure mean ≈ −406; correlation of total reward with success ≈ 0.56; with progress ≈ 0.35). Per step, parking is penalized (≈ −0.39/step) and reaching the goal pays (≈ +0.19/step). So the reward is *broadly* aligned with the intended behavior. The one genuine shape problem (Appendix C): because the dense reward is negative-per-step when not progressing, a rover that drives far and fails accumulates *more* total negative reward (≈ −548 over ~1600 steps) than one that gives up and parks early (≈ −244 over ~700 steps). Combined with the stagnation cutoff letting a stalled rover end its episode early, this creates a "cut losses by quitting" incentive that can reinforce parking. This is the leading reward-side hypothesis for the stall behavior and is the subject of the recommended reward-ablation experiment (Section 22).

---

## 8. The Controllers (LQR Baseline + Residual Composition)

### 8.1 VectorizedLQR (`controllers/lqr.py`)

`VectorizedLQR` is a torch, GPU-batched reimplementation of the PyBullet `Controller2`. It computes a baseline `(v, omega)` command for every environment in parallel. Its control law, per the docstring:

- State error vector `e = [lateral_error, heading_error, velocity_error]`.
- Command `v = v_ref + (-K e)[0]`, `omega = omega_ref + (-K e)[1]`, where `K` is a velocity-scheduled LQR gain matrix.
- A yaw-reference blending near a waypoint (within `0.3 m`) is reproduced from the original controller, so the heading reference smoothly transitions between the path-tangent direction and the stored waypoint yaw as the rover approaches each waypoint.

### 8.2 Gain Table & Trajectory Profiling

Because the optimal LQR gain depends on the reference velocity, the controller precomputes a **gain lookup table `K(v_ref)`** over a grid of reference velocities at construction, using the reference scipy Riccati solver. Key constructor parameters (defaults):

- `wheel_base = 0.34` m
- `wheel_radius = 0.3` m (the *kinematic* radius — note this is **not** the physical 0.0625 m wheel; see Section 8.4)
- `max_wheel_speed = 1.333`, `max_wheel_accel = 1.333` rad/s
- `sim_timestep = 1/50`
- `max_residual_velocity = 0.15`, `max_residual_omega = 0.30`
- `max_velocity_clip = 0.4` m/s, `max_omega_clip = 1.047` rad/s (Leo hardware limits)
- gain-table grid `v_table_min = 0.05`, `v_table_max = 1.0`, `v_table_steps = 401` (≈0.0025 m/s resolution); `v_ref` is floored at 0.15 by the trajectory profiler and lookups clamp to the table range (at `v_ref = 0` the lateral-error mode is uncontrollable and the Riccati solver fails, hence the floor).
- `yaw_smoothing_alpha = 0.85`

The trajectory profiler (`common/trajectory_profile.py`) computes the per-waypoint reference velocity/omega profile subject to the wheel-speed limits, using the controller's `wheel_base`, `wheel_radius`, and `max_wheel_speed`.

### 8.3 Differential-Drive Mapping & Wheel Limits

The `(v, omega)` total command is mapped to left/right wheel angular velocities via the standard differential-drive kinematics (`wheel = (v ± omega·L/2) / r`), then per-step acceleration-limited (`|Δwheel| ≤ max_wheel_accel · sim_timestep · 50`) and clipped to `[-max_wheel_speed, max_wheel_speed]`. With the defaults, `v = 0.4` maps exactly to `wheel = 0.4/0.3 = 1.333` rad/s, which is the wheel-speed clip — i.e., the clip and the velocity ceiling are tuned together.

### 8.4 The Wheel-Radius / Crawl-Speed Issue

There is a **kinematic/physical wheel-radius mismatch**. The controller's kinematic radius is 0.3 m, but the rover's physical wheel radius is 0.0625 m. So commanding the maximum `v = 0.4 m/s` produces a wheel speed of 1.333 rad/s, which on the physical 0.0625 m wheel yields a ground speed of only `1.333 × 0.0625 ≈ 0.083 m/s` — about 20% of the commanded velocity. In practice the rover cruises at roughly **0.035 m/s** (less than the 0.083 max because the reference is not always at 0.4 and there is slip). This "crawl" has wide consequences:

- A ~10 m path takes on the order of ~1300-1400 control steps, near the episode budget.
- The cruise speed (~0.035) sits only ~1.75× above the stagnation kill threshold (0.02 m/s), so any slowdown can trip the stagnation timer.

**Important:** this mismatch existed identically in the PyBullet stack, which still reached ~70% success there, so the crawl by itself is not what separates Isaac from PyBullet. A runtime lever, `LEOROVER_SPEED_SCALE` (and `leo train --speed S`), was added to scale the rover's effective speed by shrinking the kinematic radius and raising the wheel-speed clip together (so the commanded ceiling stays 0.4 m/s but the rover actually achieves more of it). Default is 1.0 (unchanged). Empirically, scaling speed did **not** rescue the parked rovers (they remain stuck at a fixed *fraction* of cruise, so they stay below the kill line at any multiplier), which is itself an informative negative result.

### 8.5 The Hybrid Residual Composition

In the hybrid env, `total = LQR_baseline + residual_scaled`, where `residual_scaled = action × [max_residual_velocity, max_residual_omega] × LEOROVER_RES_SCALE`. The residual-effort penalty `r_eff = -ppo_w_effort·(rvn² + rwn²)` discourages large residuals. The recorder logs the residual norms (`mean_residual_v_norm`, `mean_residual_w_norm`) so you can see how hard the PPO is pushing; near-zero means the policy has learned to leave the LQR alone. Note the recorder normalizes the residual norms by the *config* `MAX_RESIDUAL_*` (0.15/0.30), not the scaled bound, so at `RES_SCALE = 0.33` a logged norm of ~0.10 corresponds to the residual using a larger fraction of its actual (scaled) authority.

---

## 9. The Rover Asset (URDF to USD, Actuators)

The Leo Rover is loaded from URDF and converted to USD for Isaac. The articulation has four wheels grouped left (FL, RL) and right (FR, RR). Key actuator configuration (`assets/leo_rover/__init__.py`):

- Wheels are **velocity-controlled**: `stiffness = 0`, `damping > 0` — the Isaac equivalent of PyBullet's `VELOCITY_CONTROL`. The code sets both the 5.x (`*_sim`) and 4.5 field names for version tolerance.
- `effort_limit = effort_limit_sim = 1000` N·m, `damping = 1000`. **This was the fix for a 0%-success bug**: the original effort limit of 2.0 N·m starved the motors so the wheels could not move the rover; raising it to ~1000 (matching PyBullet's effective default) fixed it (Section 16.2).
- `velocity_limit = velocity_limit_sim = 100` rad/s — high enough that even full-speed (6.4 rad/s at `SPEED_SCALE = 4.8`) is within limits.
- The rockers are fixed joints.

The terrain is generated as a heightfield-derived mesh (Section 10).

---

## 10. The Terrain System & ADR Curriculum

### 10.1 Terrain Bank

At startup the env bakes a bank of distinct Mars-like terrain patches: a grid of **rows × columns** (logged as `20 × 10 = 200 patches @ 12.0 m, hscale 0.1 m, ~28,800 tris/patch, ~5.8M tris total`). Rows index **difficulty** (0 = flat, increasing roughness/slope); columns are random **variants** at that difficulty. Per-episode, a resetting env is assigned a random difficulty row in `[0, ADR ceiling]` and a random column. The logged `terrain_intensity` for an episode is `(row / (num_rows − 1)) × ADR_TERRAIN_MAX_LIMIT`, i.e., a 0-100% scale; with 20 rows the achievable intensities are discrete (≈0, 5.3, 10.5, ... %).

This heavy mesh terrain (≈5.8M triangles) is a significant contributor to the slow per-iteration time (Section 12.5).

### 10.2 ADR Curriculum (`adr_curriculum.py`)

Automatic Domain Randomization: terrain difficulty starts low and rises only when performance clears a threshold over a rolling window; if performance drops, difficulty decreases (bidirectional). Key config (`config.py`):

- `ADR_TERRAIN_MAX_START = 10.0` (starting ceiling, %), `ADR_TERRAIN_MAX_LIMIT = 100.0`
- `ADR_SUCCESS_THRESHOLD = 0.70 -> 0.60` (2026-06-24): advance when rolling success ≥ this **and** mean CTE ≤ threshold
- `ADR_CTE_THRESHOLD = 0.10` m
- `ADR_REGRESSION_SUCCESS_THRESHOLD = 0.50`: regress below this
- `ADR_EVAL_WINDOW = 200`, `ADR_MIN_EPISODES_PER_LEVEL = 50`, `ADR_COOLDOWN_EPISODES = 30`
- `ADR_STEP_UP = 3.0`, `ADR_STEP_DOWN = 3.0` (% per change)

`ADRCurriculum` exposes `report_episode(success, mean_cte)`, `should_update_difficulty()`, `update_difficulty()`, and (added in this project) `force_eval(success_rate, mean_cte)` for direct rate-driven advancement.

### 10.3 The Deterministic-Eval ADR (the fix)

A subtle, important failure mode: the **stochastic** training success rate badly understates the policy's true competence because exploration noise (action std) throws good rovers off course. Measured: ~57% stochastic vs ~92% deterministic on easy terrain at one point. That noisy ~57% sits in the dead zone between the advance (70%) and regress (50%) thresholds, so the curriculum **freezes** at the starting terrain and never ramps — fatal for any from-scratch learner, and a silent cap on the hybrid.

The fix (`config.ADR_DETERMINISTIC_EVAL = True`, default on): instead of letting the noisy rollout success drive the curriculum, `train.py` periodically runs the policy **deterministically** over the current `[0, ceiling]` terrain band and advances/regresses the curriculum on that noise-free measurement. Mechanics:

- `train.py` runs `runner.learn()` in **chunks** of `ADR_EVAL_EVERY_ITERS` (default 100) iterations; between chunks it runs a deterministic eval of `ADR_EVAL_STEPS` (default 2050, > the 2000 episode cap so every rover finishes one episode) and calls `env.apply_adr_eval(success, cte)` → `ADRCurriculum.force_eval()`.
- `env._adr_external = True` gates off the stochastic in-step ADR updates so only the deterministic eval moves the curriculum.
- `env._skip_record = True` during the eval so its episodes do not pollute the training `episode_metrics.csv`.
- Env-var overrides: `LEOROVER_ADR_EVAL` (0/1), `LEOROVER_ADR_EVAL_EVERY`, `LEOROVER_ADR_EVAL_STEPS`.

This works (terrain ramped 10% → 32% in a real run), but it was the source of several bugs and one architectural wart (the chunked `learn()` makes `leo watch` print a per-chunk "pseudo total" iteration count). See Sections 16.7-16.11. The deterministic-eval success measurement must be computed **per-rover** (first episode after reset), not per-episode over the window, or fast-failing rovers are over-counted and the rate is understated (Section 16.10, Appendix B).

---

## 11. The RL Algorithm (PPO via rsl_rl)

### 11.1 Network Architecture

Actor and critic are separate MLPs: `Linear(17 → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → 2)` (actor; critic ends `→ 1`). Small by modern standards, which is appropriate — the LQR does the heavy lifting and the residual is a low-dimensional correction.

### 11.2 Hyperparameters (`tasks/leo_rover_agents.py`)

- `num_steps_per_env = 32` (rollout length per env per update; **overridable** via `LEOROVER_NUM_STEPS`). With 4096 envs this is a 131,072-transition batch per iteration.
- `max_iterations = 30000` (note: this is far more than is practical at the current ~50 s/iter — always pass `--iters` for a sane target).
- `save_interval = 200`.
- `entropy_coef` — config `PPO_ENT_COEF = 0.001` (overridable via `LEOROVER_ENT_COEF`).
- Observation normalization: `empirical_normalization = True` (5.x runner flag) / `actor_obs_normalization = critic_obs_normalization = True`.
- `PPO_LOG_STD_INIT = -1.0` (std ≈ 0.37), `PPO_LOG_STD_MIN = -4.0` (std floor ≈ 0.018). A `SafeMlpPolicy` clamp on log_std was used in the PyBullet pure-PPO work to prevent a negative-std crash / log_std runaway.

### 11.3 Rollout Length & Credit Assignment

The 32-step rollout is **short** relative to the ~1300-step episodes. The terminal success reward (+200) lands ~1300 steps in, far beyond a 32-step horizon, so it cannot propagate back within a single rollout; the value-function loss stays high (~130) and the critic struggles to anticipate the goal reward. This is a leading suspect for why the policy plateaus (Section 17). PyBullet used effectively full-episode rollouts. `--rollout 64`/`128` raises this at the cost of ~proportionally longer iterations.

### 11.4 Isaac vs PyBullet Differences (accepted)

Documented in the agent-config comments: much larger effective batch, GPU advantage computation, empirical (vs VecNormalize) obs normalization, and the short rollout vs PyBullet's long one. These are accepted divergences, noted so results are interpreted with them in mind.

---

## 12. The Lab Workstation

### 12.1 Hardware

- GPU: **NVIDIA GeForce RTX 4090, 24 GB** (≈24,564 MiB usable).
- CPU: Intel Core i9-7900X (10 cores / 20 threads) @ 3.30 GHz.
- RAM: 128 GB.
- OS: Ubuntu 22.04.5 LTS, kernel 6.8.

### 12.2 Software Stack

- **Isaac Sim 4.5** standalone (`isaac-sim-standalone@4.5.0-rc.36`), under `~/Desktop/Core_libraries/NVIDIA_GPU/`.
- **Isaac Lab** (the `isaaclab` / `omni.isaac.lab` namespace; code is written version-tolerantly to support both 4.5 and 5.x import paths).
- **rsl-rl-lib** (rsl_rl PPO; `handle_deprecated_rsl_rl_cfg` is used to translate the agent cfg into the installed lib's schema).
- Launched via `scripts/run_lab.sh`, which wraps `isaaclab.sh -p` / the Isaac python and sets `DISPLAY`, libcuda, etc.

### 12.3 The Driver-595 Headless Constraint

The installed NVIDIA driver is **595.71.05**, and the RTX renderer **segfaults** under it (in `rtx.scenedb`). Therefore everything runs **headless physics only** — PhysX on the GPU works fine, but no rendering/GUI and **no video recording** on the box. This is why "watching" an episode is done by logging the rover's physics state (positions over time) and replaying it as a top-down plot, rather than rendering video (the `trace_episode.py` approach, Section 14).

### 12.4 Access, venv, GPU Etiquette

- **Access:** `ssh irl@10.115.102.210`, over the UA VPN. (The login password is shared separately and is deliberately not recorded in this document or in memory.)
- **Environments:** Python venvs live under `~/Desktop/environments`. The standing rule is **venv-only** changes — no driver reinstalls, no system-level changes without asking first.
- **Shared GPU etiquette:** the box is shared. A labmate ("Dang") runs an `aurora` neuro-symbolic training process that must **not** be killed. Before starting a run, check `nvidia-smi` / `leo gpu`; if a process is using more than a few GB, someone else is training — wait or coordinate. Several early failures (PhysX kernel-launch cascades) were GPU contention, not bugs.

### 12.5 Performance / Throughput

Training runs at roughly **~3,300-3,800 environment-steps/second**, i.e., **~35-50 s per iteration** (4096 envs × 32 steps ≈ 131k transitions/iter, plus the periodic deterministic-eval overhead). That is **slow for a 4090** — a clean Isaac Lab task does 50k-200k+ steps/s. The bottleneck is **not** GPU compute (the PPO update is ~0.1 s; collection is ~35 s). The suspects, in order: the ~5.8M-triangle mesh-terrain collision; Python-side per-env loops in `_reset_idx` / ADR resample (host-device copies when many envs reset at once); per-step recorder + LQR overhead; and the 10-substep decimation (each control step is 10 physics steps). Practical consequence: `max_iterations = 30000` would take ~2 weeks; always set `--iters` (the hybrid converges by ~600-800). Throughput drops further if a labmate's job shares the GPU.

---

## 13. The `leo` CLI — Complete Command Reference

`scripts/leo.sh` (alias `leo`) is the one-stop wrapper. Setup (once): `chmod +x scripts/leo.sh` and add an alias to `~/.bashrc` pointing at the repo's `scripts/leo.sh`. It auto-locates the repo, sets up the Isaac launcher path, and writes logs to `~/leo_logs/`.

**TRAIN**

- `leo train <hybrid|ppo|flat> [flags]` — start a background, logged training run (4096 envs, headless, reward fix baked in unless `--raw`, `nohup`'d).
  - `--envs N` (default 4096)
  - `--iters N` (max iterations; **always set this** given the 30000 default)
  - `--ent E` (PPO entropy coef; default 0.001)
  - `--rollout N` (steps per update; default 32; higher = better long-horizon credit)
  - `--speed S` (rover speed multiplier; default 1; 4.8 ≈ full 0.4 m/s)
  - `--residual F` (PPO residual authority; **default 0.33** = stable; 1 = full/unstable; 0 = pure LQR)
  - `--raw` (stock config, no reward fix)
  - `--fg` (foreground, do not detach)

**MONITOR**

- `leo watch` — live-tail the most recent training log (Ctrl-C stops watching, not training). Note: the chunked-eval design makes rsl-rl print a per-chunk "iteration N/total" where `total` is the chunk boundary, not the run end — it climbs, appears to finish, then the next chunk bumps it. Track real progress by the `[ADR-eval] iter N` lines.
- `leo curve` — success% + reward + std trends over recent iterations.
- `leo gpu` — `nvidia-smi` plus which training PIDs are yours.
- `leo checkpoints` — list saved checkpoints of the latest run (first saved at iteration 200).
- `leo report` — full copy-pasteable performance report (success, progress, reward, CTE trends, terrain breakdown) from the latest run's CSV (via `leo_report.py`).
- `leo csv` — print a timestamped `scp` command to pull the latest run's CSV to your laptop's `Downloads\leo_csvs\` as `episode_matrix_<timestamp>.csv`.

**EVALUATE**

- `leo trace [N] [--envs E --steps S] [--lqr|--zero-residual]` — population eval of the newest (or Nth) hybrid checkpoint; reports a POPULATION block (judge by it, not env 0) and saves a top-down plot. `--lqr` forces the residual to zero (pure-LQR baseline).
- `leo eval <hybrid|lqr|ppo> [--levels 10,20,..,80 --envs N --steps N]` — **deterministic** held-out eval over a pinned terrain sweep; writes `evals/<algo>_<ts>.csv` in the `episode_metrics` schema. `lqr` = hybrid task with residual forced to 0; `ppo` = the `Mars-v0` task.
- `leo compare` — side-by-side success/progress/CTE-by-terrain table of the latest hybrid/lqr/ppo eval CSVs (via `eval_report.py`).
- `leo evalcsv <algo>` — print the `scp` line to pull an eval CSV to your laptop.
- `leo tb` — launch TensorBoard for the latest run (prints the SSH tunnel command).

**CONTROL**

- `leo stop` — safely stop **your** training. This is non-trivial: Isaac/kit python **traps SIGTERM**, so killing the wrapper PID leaves the trainer orphaned. `leo stop` uses `pkill -9 -u $USER -f scripts/train.py` and confirms via `nvidia-smi` (Section 16.12).

Internal helpers: `task_id`/`task_exp` (alias → gym id / log folder), `latest_run`, `latest_ckpt`, `FIX_ENV` (the validated reward override), `DEFAULT_ENT`, `BOX_HOST`.

---

## 14. The Tooling Scripts

- **`scripts/train.py`** — the training entry point. Builds the env, attaches `EpisodeMetricsRecorder` (a non-invasive `env.step` wrapper), wraps for rsl_rl, and runs PPO. When the deterministic-eval ADR is on, it runs `learn()` in chunks with a periodic deterministic eval (`_adr_eval`) between chunks, saving a checkpoint each chunk. Does **not** do a bare `import config` (that collides with Isaac's bundled `cv2/config.py` once Isaac is imported — Section 16.7).
- **`scripts/trace_episode.py`** — headless, no-renderer episode tracer. Records one env's full trajectory + the reference path and writes `trace.csv`, `path.csv`, and a top-down `trace.png` (rover path vs reference + speed/progress/waypoint panels). Also prints a POPULATION block over all envs (best progress reached, fraction that ever reached goal, parked fraction). Has a `--zero-residual` flag for the pure-LQR baseline. This is how you "watch" an episode given the broken renderer.
- **`scripts/evaluate_policy.py`** — deterministic, held-out evaluation across a terrain-difficulty sweep with random paths; logs one row per finished episode to a named CSV in the `episode_metrics` schema, via `EpisodeMetricsRecorder`. Pins terrain through `env._eval_levels` (disabling the ADR), runs the deterministic inference policy, handles hybrid / lqr (`--zero-residual`) / ppo modes. (Note: as written it counts per-episode success, which under-reports per-rover; see Appendix B.)
- **`scripts/eval_report.py`** — stdlib-only comparison tool. Takes one or more labeled eval CSVs and prints overall + success/progress/CTE broken down by terrain level, side by side. Runs anywhere (laptop or box), no pandas/numpy.
- **`scripts/leo_report.py`** — stdlib-only performance report on a single run's `episode_metrics.csv`: OVERALL, RECENT, TREND (12 bins), PROGRESS DISTRIBUTION, BY TERRAIN. Auto-finds the latest hybrid run.
- **`leorover_isaac/utils/recorder.py`** — `EpisodeMetricsRecorder`. Writes `episode_metrics.csv` in the exact PyBullet schema. Reads the pre-auto-reset snapshots (`_log_progress`, `_log_goal`) rather than live state. Honors `env._skip_record`. Accepts a `filename=` argument so evals can write named CSVs without clobbering the training one.
- **`scripts/run_lab.sh`** — the Isaac launcher wrapper (sets `DISPLAY`, env, runs the Isaac python on the target script).
- **`adr_curriculum.py`** (repo root) — engine-agnostic `ADRCurriculum` + `ADRConfig` (+ an unused SB3 `ADRCallback` shim).

---

## 15. Metrics, Logging & the CSV Schema

Every finished episode appends one row to `episode_metrics.csv` (training: `logs/<exp>/<run>/csv/`; evals: `evals/`). The schema (identical to PyBullet's, so old tools read it):

```
episode, mean_cte, max_cte, total_reward, mean_reward_per_step, mean_slip, steps,
success, terrain_intensity, friction_intensity, terrain_max_slope_deg,
terrain_avg_slope_deg, mean_local_slope_deg, path_progress, roll_max, pitch_max,
mean_residual_v_norm, mean_residual_w_norm
```

Notes and caveats:

- `mean_slip` and the `terrain_max/avg_slope_deg` columns are placeholders (hardcoded ~0); they are not used by the primary plots.
- `path_progress` and `success` are read from the **pre-auto-reset snapshots** (`_log_progress`/`_log_goal`). Before that fix they were read after the reset and pinned at ~3.8%/0% in every run (Section 16.3).
- `total_reward` is accumulated per step and was always valid even when progress/success were broken — it was the real discriminator during the bug period (collapsed run ≈ −200 flat; healthy run climbing to +250+).
- Two distinct success notions matter: the CSV's **per-episode** success (what fraction of *episodes* reached goal) over-weights fast failures, vs **per-rover** success (fraction of *rovers* that succeed on a fresh attempt). See Appendix B.
- The trace's "ever reached goal: 0/1024" population counter has historically read 0 even when real success was high — it is the same read-after-reset artifact and should be ignored in favor of the ADR-eval success and best-progress numbers.
- Episode `steps` exceeding the 2000 cap is impossible physically and is a recorder-corruption signature from the deterministic-eval boundary (Section 16.9); if you see it, the run is using pre-fix code or the recorder accumulators were not reset after an eval.

---

## 16. The Debugging Chronicle

This section is the heart of the project's institutional knowledge. Nearly every odd-looking line of code traces to one of these. They are roughly chronological.

### 16.1 PhysX GPU Kernel-Launch Cascade (GPU contention)

Early runs produced a cascade of PhysX GPU "kernel launch failure" errors. Root cause: **GPU contention** — a labmate's `aurora` process was using the GPU at the same time. Not a code bug. Resolved when the other job ended. **Lesson:** always check `leo gpu` / `nvidia-smi` before launching; a process using more than a few GB means someone else is training.

### 16.2 Actuator Effort-Limit Starvation (0% success)

The rover would not move and every run showed 0% success. Root cause: the wheel actuator `effort_limit` was **2.0 N·m**, which starved the motors — not enough torque to drive the rover on terrain. **Fix:** raise `effort_limit` (and `effort_limit_sim`) to ~**1000 N·m** with `damping = 1000` (PyBullet's velocity controller had an effectively high/default torque). The rover drove normally afterward.

### 16.3 The Recorder Read-After-Auto-Reset Bug (the big one)

For a long time, **every** `episode_metrics.csv` showed `path_progress` pinned at ~3.8% and `success = 0`, regardless of how well the policy was actually doing. This made it impossible to tell good runs from bad. Root cause: `EpisodeMetricsRecorder._flush` called `env._path_progress()` / `env._is_goal_reached()` **after** Isaac's `DirectRLEnv` auto-reset, so it read the *respawned* state (back at the path start = ~3.8% progress, not at goal) for every just-finished episode. **Fix:** `_get_dones` snapshots `self._log_progress = self._path_progress()` and `self._log_goal = goal` **before** the auto-reset, and the recorder reads those snapshots. **Crucial corollary:** the whole "4%/0%" pattern that looked like a learning failure was largely this logging artifact; `total_reward` (accumulated per step) was valid all along and was the real signal during the bug period. This same "read before the auto-reset" rule recurs in every later measurement bug.

### 16.4 The Training Stall — Effort Penalty Dominating at Crawl Speed

From-scratch training stalled at ~0-4% success: the policy would zero its residual, park, and the action-std would collapse. This looked like a physics or scale problem but was a **reward-shaping** problem, confirmed by reading `_get_rewards`. At the ~0.035 m/s crawl, the forward rewards are tiny (`r_vel ≈ 0.018`, `r_prog ≈ 0.007`/step) while the hybrid residual **effort penalty** `r_eff = -0.5·(rvn² + rwn²)` dominates (≈ −0.14/step for a random residual). Net ≈ −0.18/step ≈ −225/episode — so the reward-maximizing policy is to **zero the residual and park**, std collapses to ~0.02, 0% success, stuck near waypoint 3 (~3.8%). **Fix (env-vars, no code edit), validated:** `LEOROVER_W_EFFORT=0.05 W_PROGRESS=150 W_SMOOTH=0.1` (later `ENT_COEF=0.001`). By ~iter 220, ADR success ~50% (from 0%), mean reward climbing, std held, value loss fitting. This is the `FIX_ENV` baked into `leo`. The env was learnable all along; the blocker was reward shaping — exactly the surface the research wanted to study.

### 16.5 Pure-LQR vs Hybrid — the Residual Was Hurting

A paired `leo trace` on the same checkpoint, one with the residual on and one with `--zero-residual` (pure LQR), revealed that on easy terrain the **pure LQR reached ~94%** while the **full-authority hybrid reached ~43%** — the trained residual was actively *dragging the LQR down*. The noisy residual (std ~0.74) shoved a good LQR trajectory off-path and into stalls (pure-LQR parked ~1.2% of envs vs the hybrid's ~15%). This is what motivated bounding the residual.

### 16.6 Constrained Residual (0.33) — Stable

Shrinking the residual authority to **0.33** of its bound (`LEOROVER_RES_SCALE = 0.33`, later made the default everywhere) made the hybrid stable and brought its deterministic success up to ~88-92% on easy terrain — close to pure LQR and far above the full-authority 43%. The residual could no longer over-steer. (Whether it actually *helps* over LQR is the later, decisive question — Section 17.)

### 16.7 The `cv2/config.py` Import Collision (training crashed at startup)

A newly added `import config as cfg_mod` in `train.py` caused an instant crash: `NameError: name 'LOADER_DIR' is not defined`, originating in `.../pip_prebundle/cv2/config.py`. Root cause: the import ran **after** Isaac was loaded, and Isaac puts its bundled OpenCV on `sys.path` ahead of the repo, so a bare `import config` resolved to OpenCV's `cv2/config.py` (which crashes on import) instead of the repo's `config.py`. **Fix:** `train.py` does not bare-`import config` — it uses literal/env-var defaults for the two values it needed; `evaluate_policy.py` likewise avoids it (uses a literal terrain-max constant). **Lesson:** never bare-`import config` in a script that runs after `AppLauncher`.

### 16.8 The `_log_goal` Bug in the ADR Eval (curriculum frozen at 10%)

After adding the deterministic-eval ADR, terrain still would not advance — a 2300-iteration / 230k-episode run stayed pinned at ≤10% terrain the whole time. Root cause: the *same* read-after-auto-reset bug, reincarnated. The eval measured success with `raw._is_goal_reached()` **after** `wrapped.step()`, so a rover that reached the goal had already auto-respawned and read as "not at goal" — the eval measured ~0% success every cycle, decided the policy was failing, and pinned the curriculum at the floor. **Fix:** read the pre-reset `raw._log_goal` snapshot (the same fix as 16.3). The trace's "ever reached goal: 0/1024" had been showing this artifact in plain sight.

### 16.9 The Recorder Step-Counting Corruption (episodes > 2000 steps)

The "violent oscillation" plots showed episode lengths up to ~3,945 steps — impossible given the 2000 cap. Root cause: during the deterministic eval, `env._skip_record = True` freezes the recorder (no accumulation, no done-handling), while the eval itself resets the envs and runs ~2050 steps. At eval end the recorder's per-env step counters were **stale**, so episodes straddling the eval boundary double-counted steps, producing impossible lengths and spiky CTE/reward curves. **Fix:** after each eval, `recorder._reset_accum(slice(None))` zeros the accumulators so the next logged episodes start clean. Any `steps > 2000` in a CSV means this fix is not active in the running process.

### 16.10 Per-Episode vs Per-Rover Success Bias

The deterministic eval reported ~66% success, which felt low given ~97% median progress. Root cause: the eval counted **every** completed episode over its window, and fast-failing rovers cycle through far more episodes than finishers (a parker stalls and respawns ~2.5× in the time a finisher does one ~1300-step run). So failures were over-counted and the **per-episode** rate badly understated the **per-rover** rate. The math: a true ~81% per-rover success reads as ~66% per-episode (Appendix B). **Fix (ADR eval):** score **one episode per rover** — the first after reset — via `(first_succ & ~ever_done)` bookkeeping over a window long enough (`2050` steps) that every rover finishes one episode. **Note:** `evaluate_policy.py` (the `leo eval` sweep) still counts per-episode, so its absolute numbers under-report per-rover; the *comparison* between algorithms is unaffected because both are measured the same way.

### 16.11 The Chunked-`learn()` Display Artifact (the "pseudo last iteration")

`leo watch` appeared to show training "finishing" then continuing past the supposed last iteration. Root cause: the deterministic-eval design runs `runner.learn()` in chunks; rsl-rl prints `iteration N / chunk-target` for each chunk, where the target is just that chunk's endpoint, not the run's end. It climbs, looks done, then the next chunk bumps the target. Cosmetic only — track real progress by the cumulative `[ADR-eval] iter N` lines. (A clearer per-chunk banner is an easy future improvement.)

### 16.12 The SIGTERM-Trap Orphan Gotcha

Killing the `run_lab.sh` / `nohup` wrapper PID leaves the actual trainer (the kit python, ~11 GB) **orphaned and still running**, silently halving GPU throughput for the next run, because Isaac/kit python **traps SIGTERM**. **Fix:** kill the kit-python PID with `kill -9`, or `pkill -f scripts/train.py`, and confirm with `nvidia-smi`. This is exactly what `leo stop` does.

### 16.13 Operational / Tooling Gotchas

- **"Code is pulled" ≠ "the running process uses it."** A long-running training process keeps executing the source it loaded at launch; `git pull` updates files on disk but not a running process. The only way the trainer picks up a change is `leo stop` + relaunch. A run still showing pre-fix behavior after a pull almost always means it was started before the pull. (Confirm with `git log --oneline` plus the behavioral tells, e.g., no `steps > 2000`, `[ADR-eval]` reading the expected number.)
- **PowerShell quirks (laptop side):** no `&&` separator (run git commands one per line); `~` does not expand for `scp` (use `$HOME` or a full Windows path). The `leo csv` helper emits PowerShell-correct lines.
- **Shell-script executable bit lost on pull:** `git pull` can land `leo.sh` without its `+x` bit; `chmod +x scripts/*.sh` after pulling, or mark it executable in the git index once.
- **`.gitattributes`** pins `*.sh` to LF endings so the scripts run after Windows commits.

---

## 17. Key Findings & Results

### 17.1 The Reward Rebalance Works

Cutting the effort penalty and raising the progress weight (the `FIX_ENV`) took from-scratch hybrid training from ~0-4% to ~**59%** training-time success. This validated that the environment is learnable and that the original stall was reward shaping, not physics or scale.

### 17.2 The Curriculum Freeze and Its Fix

The stochastic training success (~57%) sits in the ADR dead zone (50-70%), so the success-gated curriculum froze at the start terrain. Driving the ADR from a periodic **deterministic** evaluation (Section 10.3) fixed it: in a real run the terrain ceiling ramped **10% → 13% → 16% → 19% → 26% → 32%**, the first time it ever advanced off the floor. With the per-rover success fix and a lowered advance bar (0.60), the curriculum advances correctly and holds when the policy's deterministic success drops below the bar.

### 17.3 The Decisive Comparison — the Residual Adds No Value

A clean held-out `leo eval` sweep (deterministic, terrain 10-79%, both measured identically) on the same checkpoint:

| terrain % | hybrid success | lqr success |
|---|---|---|
| 10 | 38% | 36% |
| 21 | 36% | 37% |
| 32 | 36% | 41% |
| 42 | 35% | 36% |
| 53 | 39% | 37% |
| 58 | 36% | 41% |
| 68 | 38% | 39% |
| 79 | 35% | 40% |
| **overall** | **36.7%** | **38.1%** |

(These are per-episode numbers — true per-rover ≈ 57% — but the comparison is what matters.) **Hybrid and pure LQR are statistically identical at every difficulty; LQR is marginally ahead overall.** Under the settings tried (residual 0.33, the fixed reward, short rollout), the bounded PPO residual does **not** improve on the LQR baseline.

### 17.4 The Bottleneck Is Not Terrain, and Not Goal-Capture

- **Not terrain:** success is essentially **flat across difficulty** (≈57-58% per-rover from 0-9% up to 79% terrain). Harder ground is not what is failing the rover.
- **Not goal-capture:** once a rover reaches 90% of the path, it succeeds ~95% of the time. The strict 0.2 m goal tolerance is not the gate.
- **It is stalls:** the failures are rovers getting **stuck**, and they are **terrain-independent**: ~15% park near the start (die at ~700 steps to the stagnation cutoff with <5% progress), another ~9% stall at 5-50%, and ~15% at 50-90%. That ~38% failure is roughly constant at every difficulty.

### 17.5 Reward Alignment

Measured on real data, the reward is broadly aligned (success +282 vs failure −406; corr 0.56; parking penalized per step). The one real shape problem is the "give-up-early" incentive (Section 7, Appendix C): a long failed run accrues more total negative reward than an early park, which can reinforce parking. This is the leading reward-side hypothesis and the subject of the recommended ablation.

### 17.6 Speed Is a Red Herring (for the stalls)

The crawl speed (~0.035 m/s) is close to the stagnation kill line, but a 3× speed trace showed the parked fraction essentially unchanged (~14%): stuck rovers move at a fixed *fraction* of cruise, so they stay below the kill line at any speed multiplier. Episode length is also not the limiter — only ~1-2% of episodes hit the 2000-step time cap.

---

## 18. Current State of the Project

- **Migration:** complete and validated. Env, controller, reward, curriculum, asset all ported and faithful. Tooling (CLI, trace, eval, compare, report) all working.
- **Curriculum:** the deterministic-eval ADR works; terrain ramps and holds appropriately. Per-rover success measurement is fixed in the ADR eval.
- **Performance:** deterministic per-rover success ≈ **57%**, roughly flat across terrain 10-79%. Hybrid ≈ pure LQR. The policy converges (low surrogate loss, action std ~0.13) to this plateau.
- **The open scientific question:** can the PPO residual be made to beat the LQR? Not yet, under any setting tried. The honest current reading is that the LQR is near the achievable ceiling for this task as currently formulated, and the real performance limiter is the terrain-independent stall mode.
- **Leading suspects for the plateau:** (a) the short 32-step rollout starves the value function (loss ~130) so the terminal goal reward never propagates; (b) the reward's give-up-early shape may reinforce parking; (c) the stalls may be physically un-fixable by bounded `(v, omega)` residuals (a stuck rover may need a maneuver the residual cannot express); (d) the LQR itself stalls ~13% reward-free, so part of the ceiling is the controller/physics, not the learner.
- **Outstanding experiments:** a reward ablation (test the give-up hypothesis), a longer-rollout run (`--rollout 128`), and a direct visualization of the stalls (trajectory viewer) to see what parked rovers physically do.

---

## 19. Configuration Reference

### 19.1 `config.py` — Key Values

(Verify against the live file; values that changed are shown `OLD -> NEW`.)

Reward weights:

- `PPO_W_CTE = 5`, `PPO_W_PROGRESS = 10`, `PPO_W_VELOCITY = 0.5`, `PPO_W_ALIVE = 0`, `PPO_W_SMOOTHNESS = 0.5`, `PPO_W_HEADING = 0.5`, `PPO_W_EFFORT = 0.5`
- `PPO_SUCCESS_BONUS = 200`, `PPO_FAILURE_PENALTY = 50`

PPO / policy:

- `PPO_ENT_COEF = 0.001`
- `PPO_LOG_STD_INIT = -1.0` (std ≈ 0.37), `PPO_LOG_STD_MIN = -4.0` (std floor ≈ 0.018)

Controller / dynamics:

- `MAX_RESIDUAL_VELOCITY = 0.15`, `MAX_RESIDUAL_OMEGA = 0.30`
- `MAX_VELOCITY_CLIP = 0.4`, `MAX_OMEGA_CLIP = 1.047`

Episode / termination:

- `MAX_EPISODE_STEPS = 2000` (= 400 s at 0.2 s/step)
- `STAGNATION_VEL_THRESHOLD = 0.02`, `STAGNATION_TERMINATION_STEPS = 600`
- goal tolerance `0.2 m`, waypoint tolerance `0.2 m`, flip threshold `g_z < 0.5`

ADR curriculum:

- `ADR_TERRAIN_MAX_START = 10.0`, `ADR_TERRAIN_MAX_LIMIT = 100.0`
- `ADR_SUCCESS_THRESHOLD = 0.70 -> 0.60` (2026-06-24)
- `ADR_REGRESSION_SUCCESS_THRESHOLD = 0.50`, `ADR_CTE_THRESHOLD = 0.10`
- `ADR_EVAL_WINDOW = 200`, `ADR_MIN_EPISODES_PER_LEVEL = 50`, `ADR_COOLDOWN_EPISODES = 30`
- `ADR_STEP_UP = 3.0`, `ADR_STEP_DOWN = 3.0`
- `ADR_DETERMINISTIC_EVAL = True`, `ADR_EVAL_EVERY_ITERS = 100`, `ADR_EVAL_STEPS = 1500` (train.py uses a 2050 literal so all first episodes finish)

Agent (`tasks/leo_rover_agents.py`):

- `num_steps_per_env = 32`, `max_iterations = 30000`, `save_interval = 200`, `empirical_normalization = True`

### 19.2 Runtime Env-Var Overrides

| Env var | Effect | Default |
|---|---|---|
| `LEOROVER_W_PROGRESS` | progress reward weight | 10 (fix: 150) |
| `LEOROVER_W_EFFORT` | residual effort penalty weight | 0.5 (fix: 0.05) |
| `LEOROVER_W_SMOOTH` | smoothness penalty weight | 0.5 (fix: 0.1) |
| `LEOROVER_W_CTE` / `_VELOCITY` / `_ALIVE` / `_HEADING` / `_CTE_OK` | corresponding reward terms | config |
| `LEOROVER_ENT_COEF` | PPO entropy coefficient | 0.001 |
| `LEOROVER_NUM_STEPS` | rollout length per env | 32 |
| `LEOROVER_SPEED_SCALE` | rover speed multiplier (shrinks kinematic radius + raises wheel clip) | 1.0 |
| `LEOROVER_RES_SCALE` | PPO residual authority multiplier (0 = pure LQR) | 0.33 |
| `LEOROVER_ADR_EVAL` | enable deterministic-eval ADR (0/1) | 1 |
| `LEOROVER_ADR_EVAL_EVERY` | iterations between deterministic evals | 100 |
| `LEOROVER_ADR_EVAL_STEPS` | steps per deterministic eval | 2050 |
| `LEOROVER_DEBUG` | print recorder exceptions | unset |

---

## 20. Operational Runbook

### 20.1 First-Time Setup (on the box)

```bash
# connect (over UA VPN)
ssh irl@10.115.102.210
cd ~/leorover_work/leorover_isaac
chmod +x scripts/*.sh
echo "alias leo='$HOME/leorover_work/leorover_isaac/scripts/leo.sh'" >> ~/.bashrc
source ~/.bashrc
```

### 20.2 Edit → Deploy Cycle (the safe way)

Code is edited on Windows and run on the box:

```
# Windows (in the repo)
git add <files>
git commit -m "..."
git push
```
```bash
# box
cd ~/leorover_work/leorover_isaac && git pull && chmod +x scripts/*.sh
git log --oneline -1     # confirm your commit is at HEAD
```

Then **relaunch** any training so the new code is actually loaded (a running process does not hot-reload — Section 16.13).

### 20.3 Train

```bash
leo gpu                          # confirm the GPU is free (~225 MiB idle)
leo train hybrid --iters 800     # background, logged, reward fix on, residual 0.33
leo watch                        # confirm it boots + learns, then Ctrl-C
```

For a quick sanity check that the deterministic-eval path works before an overnight run:

```bash
LEOROVER_ADR_EVAL_EVERY=10 leo train hybrid --iters 25 --fg
# expect two [ADR-eval] lines, non-zero det success, terrain_max climbing, no >2000-step episodes
```

### 20.4 Monitor

```bash
leo watch     # live log
leo curve     # success/reward/std trends
leo report    # full stats from the latest CSV
grep "ADR-eval" "$(ls -t ~/leo_logs/*.log | head -1)"   # curriculum progress + true det success
```

### 20.5 Evaluate

```bash
leo trace                  # population eval of the newest checkpoint (+ top-down plot)
leo trace --lqr            # pure-LQR baseline
leo eval hybrid ; leo eval lqr ; leo compare    # held-out terrain-sweep comparison
```

`leo eval` can run alongside training (it shares the GPU like `leo trace` does); it just slows both for a few minutes. Do **not** stop-and-restart training to evaluate — that resets the curriculum.

### 20.6 Pull Data to the Laptop

```bash
leo csv            # prints the scp line for the latest training CSV
leo evalcsv hybrid # prints the scp line for the latest hybrid eval CSV
```

### 20.7 Stop

```bash
leo stop      # SIGTERM-trap-safe; confirm with leo gpu afterward
```

---

## 21. Known Issues, Gotchas & Footguns

- **Auto-reset timing.** Any per-episode state must be read from the pre-reset snapshot (`_log_progress`/`_log_goal`), never live after `step()`. This bug has appeared three times (Sections 16.3, 16.8, and the trace's goal counter).
- **Bare `import config` after Isaac is loaded** resolves to OpenCV's `cv2/config.py` and crashes. Don't.
- **A running process ignores `git pull`.** Relaunch to load new code.
- **Isaac/kit traps SIGTERM.** Use `leo stop` / `kill -9` / `pkill -f scripts/train.py`, then verify GPU is freed.
- **`max_iterations = 30000`** is ~2 weeks at current speed; always pass `--iters`.
- **`leo watch` "iteration N/total"** is a per-chunk pseudo-total, not the run end.
- **Per-episode vs per-rover success.** The `leo eval` sweep reports per-episode (under-reports per-rover ~15-20 points); use it for comparisons, not absolute claims, until `evaluate_policy.py` is switched to per-rover.
- **Shared GPU.** Never kill Dang's `aurora`; check `leo gpu` before launching.
- **Sandbox mount truncation (development only).** When editing from the Claude tooling, the Linux sandbox sometimes sees a truncated copy of a just-edited file, causing spurious `bash -n`/`py_compile` "unexpected EOF" errors at lines past the edit; the file-API view is authoritative. Not a runtime issue on the box.
- **Full residual (`--residual 1`) is unstable** and degrades success; keep the 0.33 default unless deliberately testing.

---

## 22. Open Questions & Recommended Next Steps

1. **Reward ablation (highest value).** Test whether the "give-up-early" reward shape is reinforcing parking. Two clean variants: (a) floor the dense per-step reward at ≥0 so a long failed run is not punished more than an early park; (b) strip to progress + success only (drop CTE/velocity/smoothness shaping). Retrain and compare parked-% and success. If parking drops, reshape the reward; if not, the reward is exonerated and the cause is the controller/physics.
2. **Direct stall visualization.** Build a trajectory viewer that replays the logged positions of the parked/stalled rovers (renderer-free, since video is impossible on driver 595) to see *physically* what they do at spawn and where they get stuck. This is the fastest way to learn whether it is spawn pose, a sharp first waypoint, a terrain feature, or the controller giving up.
3. **Longer rollout.** `leo train hybrid --rollout 128`. The value-function loss (~130) and the +200 terminal reward landing far beyond a 32-step horizon point at starved credit assignment; a longer rollout may let the policy push past the plateau and will calm the per-iteration reward noise.
4. **Make `evaluate_policy.py` per-rover.** Port the first-episode-per-rover counting from the ADR eval into the `leo eval` sweep so its absolute numbers are honest.
5. **Residual formulation.** If bounded `(v, omega)` residuals fundamentally cannot un-stick rovers, consider a different residual action space or a state-dependent residual gate that activates only where the LQR struggles.
6. **Write-up.** Even the negative result is publishable: "a bounded PPO residual does not improve on a trajectory-profiled LQR for Leo-Rover Mars navigation; failures are dominated by terrain-independent stalls, not difficulty," with the curriculum and measurement methodology as contributions.

---

## 23. Glossary

- **ADR** — Automatic Domain Randomization; the terrain-difficulty curriculum.
- **CTE** — Cross-Track Error; perpendicular distance from the rover to the reference path.
- **DirectRLEnv** — Isaac Lab's direct (non-manager) RL environment base class; auto-resets done envs inside `step()`.
- **Hybrid** — LQR baseline + bounded PPO residual.
- **LQR** — Linear-Quadratic Regulator; the model-based baseline controller.
- **Per-episode vs per-rover success** — fraction of *episodes* that reach goal (over-weights fast failures) vs fraction of *rovers* that succeed on a fresh attempt.
- **Residual** — the PPO policy's 2-D `[-1,1]` output, scaled and added to the LQR command.
- **Rollout (`num_steps_per_env`)** — steps collected per env per PPO update.
- **Stagnation** — forward speed < 0.02 m/s for 600 steps → termination; the mechanism that kills parked rovers.
- **`FIX_ENV`** — the validated reward-override set baked into the `leo` CLI.
- **Trace / population block** — `trace_episode.py` output; judge by the population aggregate, not env 0.

---

## 24. File & Directory Map

```
leorover_isaac/
  config.py                         # all tunable constants (reward, ADR, dynamics, episode)
  adr_curriculum.py                 # ADRCurriculum + ADRConfig (+ unused SB3 callback shim)
  PROJECT_REFERENCE.md              # this document
  RUNBOOK.md                        # plain-English quickstart
  .gitattributes                    # *.sh eol=lf
  leorover_isaac/
    envs/
      leo_rover_base_env.py         # base DirectRLEnv: obs/reward/dones, LQR wiring, ADR, snapshots, overrides
      leo_rover_flat_env.py         # flat-ground task
      leo_rover_mars_env.py         # Mars, pure PPO
      leo_rover_mars_hybrid_env.py  # Mars, hybrid (primary)
    controllers/
      lqr.py                        # VectorizedLQR (baseline controller, gain table, diff-drive mapping)
    assets/leo_rover/__init__.py    # articulation cfg (actuators: effort 1000, vel 100, damping 1000)
    tasks/leo_rover_agents.py       # rsl_rl runner cfgs (rollout, entropy, normalization)
    utils/recorder.py               # EpisodeMetricsRecorder (PyBullet-schema CSV; pre-reset snapshots)
    common/                         # engine-agnostic carried-over logic (LQR baseline, trajectory profile, terrain intensity)
  scripts/
    leo.sh                          # the leo CLI wrapper
    run_lab.sh                      # Isaac launcher
    train.py                        # training entry (chunked learn + deterministic ADR eval + recorder)
    trace_episode.py                # headless trace + top-down plot (+ --zero-residual)
    evaluate_policy.py              # deterministic terrain-sweep eval -> named CSV
    eval_report.py                  # side-by-side comparison of eval CSVs
    leo_report.py                   # single-run performance report
  logs/<experiment>/<run>/          # checkpoints (model_*.pt) + csv/episode_metrics.csv + eval_trace/
  evals/<algo>_<timestamp>.csv      # held-out eval outputs

# on the box:  ~/leo_logs/*.log     # training stdout logs (leo watch tails these)
```

Experiment folder names: hybrid = `leo_rover_mars_hybrid`, pure PPO = `leo_rover_mars`, flat = `leo_rover_flat`.

---

## 25. Appendices

### Appendix A — Failure-Mode Breakdown (representative run)

From a hybrid run's training CSV (per-episode):

| Outcome (by path_progress) | Share | Mean steps |
|---|---|---|
| Parked (<5%) | 15.5% | ~700 (stagnation kill) |
| Stalled (5-50%) | 8.8% | ~900 |
| Partial (50-90%) | 14.6% | ~1600 |
| Near/goal (≥90%) | 60.5% | ~1400 |
| Reached ~100% flag | 0.6% | ~1810 |

Only ~8% of episodes hit the 2000-step time cap, and ~53% of those still succeeded — time is not the limiter. Once a rover reaches ≥90% progress it succeeds ~95% of the time. Parked-% is ~15-16% at every terrain band (terrain-independent).

### Appendix B — Per-Episode vs Per-Rover Success (the math)

If a succeeding episode takes `Ts ≈ 1300` steps and a failing one `Tf ≈ 600`, then over a fixed window failers complete more episodes per unit time, so the per-episode success rate understates per-rover success `p`:

```
per_episode = (p/Ts) / ( p/Ts + (1-p)/Tf )
p (per-rover) = e·Ts / ( (1-e)·Tf + e·Ts )     # inverse, e = per_episode
```

Worked values (Ts=1300, Tf=600):

| true per-rover | reads as per-episode |
|---|---|
| 60% | 41% |
| 70% | 52% |
| 81% | 66% |
| 90% | 81% |

So the eval's per-episode ~38% corresponds to roughly ~57% per-rover. The fix is to count one episode per rover (first attempt after reset), implemented in the ADR eval; `evaluate_policy.py` still needs the same treatment.

### Appendix C — Reward-Alignment Analysis (representative run)

- total reward vs success: success mean +281.8 (median +311.6); failure mean −405.8 (median −106.4); corr(reward, success) = 0.56; corr(reward, progress) = 0.35.
- per-step reward by outcome: parked −0.386, stalled −0.394, partial −0.360, near/goal +0.194.
- per-episode reward by outcome: parked −244.5 (≈700 steps); stalled −321.2; partial **−548.4** (≈1600 steps); near/goal +230.5.

The misalignment: per-episode, driving far and failing (−548) is punished more than parking early (−244), purely because negative-per-step reward accumulates over the longer episode and the stagnation cutoff lets a stalled rover quit early. This is the "give-up-early" incentive targeted by the proposed reward ablation.

### Appendix D — Run History & Key Milestones

- Reward fix validated (effort 0.5→0.05, progress 10→150): 0-4% → ~59% training success.
- Pure-LQR vs full-residual hybrid (early): ~94% vs ~43% on easy terrain → residual hurting.
- Constrained residual (0.33): ~88-92% on easy terrain, stable.
- Deterministic-eval ADR + per-rover fix: curriculum ramps 10% → 32%.
- Held-out sweep (10-79%): hybrid ≈ LQR (≈37% per-episode / ≈57% per-rover), flat across terrain.

### Appendix E — Sensitive Information

The lab login credential is shared out-of-band and is deliberately **not** recorded in this document, in memory, or in the repo. Access requires the UA VPN. Do not commit credentials.

---

*End of document.*

