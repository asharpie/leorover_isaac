# Porting Roadmap: leoroverpybullet → leorover_isaac

Phase-by-phase plan for porting the PyBullet Leo Rover RL stack to Isaac Lab.
The PyBullet repo stays intact throughout — this is purely additive work in
a separate codebase. You can revert to PyBullet at any point by switching
git remotes.

The roadmap is ordered so each phase produces something runnable end-to-end
with stub fallbacks for the parts not yet ported. You should never have a
"big bang" port where nothing runs for weeks at a time.

---

## PyBullet → Isaac Lab concept mapping

Use this as a quick-reference table while porting. The "Notes" column flags
the gotchas that bit other porting projects.

| PyBullet concept | Isaac Lab equivalent | Notes |
|------------------|----------------------|-------|
| `p.loadURDF()` | `ArticulationCfg` + USD asset | URDFs must be converted to USD; Isaac Lab ships `convert_urdf.py` |
| `p.stepSimulation()` | implicit — sim steps for all envs in parallel each `env.step()` | No more "for env in envs: env.step()"; everything is batched |
| `getNumJoints` / per-joint state | `articulation.data.joint_pos` (tensor of shape `[num_envs, num_joints]`) | All state lives on GPU as PyTorch tensors |
| `SubprocVecEnv(12)` | `num_envs=4096` in Isaac Lab task config | Native parallelism, no subprocesses |
| `MyEnv2.step(action)` | Manager-based env: `_apply_action()`, `_get_observations()`, reward `RewardTerm`s | Action application and obs computation become separate hooks |
| Per-env terrain heightmap (numpy) | `TerrainImporter` + `HfRandomUniformTerrainCfg` or custom heightfield | Heightfield colliders are GPU-resident; generation runs once at startup |
| Custom contact / friction | Material properties on USD prims | Use `PhysicsMaterialCfg` per asset |
| `np.array` observations | `torch.Tensor` of shape `[num_envs, obs_dim]` on cuda:0 | All math vectorized in PyTorch |
| Reward function (per-step scalar) | `RewardTerm` callbacks returning `[num_envs]` tensors | Each term is registered separately; weights live in the task config |
| ADR via `adr_curriculum.py` | `CurriculumTerm` or `EventTerm` with conditional triggers | Isaac Lab has built-in curriculum infra |
| LQR controller (`Controller2`) | PyTorch-vectorized LQR (matrix mul on GPU) | Don't sync to CPU per step — keep gains as `torch.Tensor` |
| `MyEnv2` `_use_lqr_baseline` flag | Hybrid task subclasses base task; LQR runs as a torch op inside `_apply_action` | Action = LQR(obs) + clip(residual_policy(obs), bound) |
| `make_env(...)` factory | Task entry point registered with `gymnasium.register` | Standard gym registration; Isaac Lab discovers tasks via entry points |
| SB3 `PPO` | `rsl_rl.runners.OnPolicyRunner` or `skrl.agents.PPO` | rsl_rl is the de facto standard in Isaac Lab |
| `VecNormalize` | `RunningMeanStd` from rsl_rl | Same idea, GPU-resident |
| `episode_metrics.csv` callback | Custom `Recorder` or wandb integration | wandb just works; raw CSV needs a few lines of glue |
| Random per-episode terrain seed | `EventTerm` with `mode="reset"` randomizing the terrain prim | Built-in randomization mode |

---

## Phase 0 — Environment ready (you are here)

**Goal:** Isaac Sim + Isaac Lab installed, smoke tests pass, leorover_isaac
package importable.

**Deliverable:** can run the cartpole demo end-to-end.

**Checklist:**
- [ ] WSL2 + Ubuntu 22.04 working with `nvidia-smi`
- [ ] Isaac Sim 4.5 installed in `isaaclab` conda env
- [ ] Isaac Lab cloned + `./isaaclab.sh --install` succeeded
- [ ] Cartpole training runs and reward increases
- [ ] `pip install -e .` succeeded in leorover_isaac
- [ ] git repo initialized and pushed to GitHub

Do not move to Phase 1 until all of these are green.

---

## Phase 1 — Asset port (URDF → USD)

**Goal:** the Leo Rover loads in Isaac Sim as a USD asset with correct
joints, inertias, and visuals.

**Deliverable:** a script that spawns one rover in an empty Isaac Sim scene,
applies a constant wheel velocity, and shows the rover driving in a straight
line.

**Steps:**

1. Copy `leo_robot_1_ros2_shared.urdf` and `macros.xacro` from the PyBullet
   repo into `leorover_isaac/assets/leo_rover/source/`. These are the
   ground-truth source files; the converted USD will be generated from them
   and is `.gitignore`d.
2. Resolve any xacro includes if needed (process xacro → URDF if the URDF
   isn't standalone).
3. Run Isaac Lab's URDF converter:
   ```bash
   ./isaaclab.sh -p source/standalone/tools/convert_urdf.py \
       leorover_isaac/assets/leo_rover/source/leo_robot.urdf \
       leorover_isaac/assets/leo_rover/leo_robot.usd \
       --merge-joints
   ```
4. Open the result in Isaac Sim GUI (just once, manually) and verify:
   - Wheel meshes look correct
   - All 4 wheel joints are present and continuous-rotation
   - Inertias are reasonable (no near-zero or absurdly large diagonals)
5. Write `leorover_isaac/assets/leo_rover/__init__.py` exporting a
   `LeoRoverArticulationCfg` for use by tasks. Pattern this after one of
   Isaac Lab's existing wheeled-robot configs (Carter, Husky).

**Gotcha:** URDF velocity controllers don't translate to USD cleanly. You
control wheel velocities by writing directly to `articulation.write_joint_velocity_target_to_sim()`,
not via the URDF's `<transmission>` blocks.

---

## Phase 2 — Bare terrain task

**Goal:** a minimal Isaac Lab task with the rover on a flat ground plane,
gymnasium-registered, runnable through the train script. No reward yet,
just movement.

**Deliverable:** `Isaac-LeoRover-Flat-v0` task that you can run with
`./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py
--task Isaac-LeoRover-Flat-v0 --num_envs 64 --headless` and see iteration
output (even if rewards are nonsense).

**Steps:**

1. Implement `leorover_isaac/envs/leo_rover_flat_env.py` extending
   `DirectRLEnv`. Override `_setup_scene`, `_pre_physics_step`,
   `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`.
2. Action: `Box(-1, 1, shape=(2,))` matching the PyBullet (linear_v, omega) interface.
   Inside `_apply_action`, convert to per-wheel velocities using the unicycle
   model already in `leoroverpybullet/.../controller2.py` — port the math
   to PyTorch.
3. Observation: start with a minimal 6D vector (cte, heading_error, vx, vy, ωz, distance_to_goal).
4. Reward: placeholder — just `-cte`. The point is to validate the data path,
   not produce good behavior yet.
5. Register the task in `leorover_isaac/__init__.py` via `gymnasium.register`.
6. Add the task config to a YAML at `leorover_isaac/tasks/leo_rover_flat_cfg.yaml`.

**Validation:** run for 100 iterations at num_envs=64. The reward graph
should at least be coherent (no NaNs, monotonic over very long runs).

---

## Phase 3 — Mars terrain (heightfield port)

**Goal:** rover trains on randomized Martian terrain, matching the PyBullet
terrain difficulty curriculum.

**Deliverable:** `Isaac-LeoRover-Mars-v0` task with heightfield terrain
parameterized by terrain_intensity.

**Steps:**

1. Port the heightmap generation from the PyBullet repo
   (`leoroverpybullet/.../environment2.py` terrain functions) to a function
   that returns a numpy array of shape `[H, W]` floats.
2. Construct a `TerrainImporterCfg` that consumes the heightmap as a
   custom terrain type. Each parallel env gets its own randomized terrain
   patch — Isaac Lab's `SubTerrainBaseCfg` interface supports this.
3. Wire terrain_intensity into an `EventTerm` so it can be queried/modified
   by the ADR curriculum.
4. Port the camera lookahead observation (slope_near, slope_mid, slope_far)
   to a GPU-side terrain query — Isaac Lab has `RayCaster` sensors that
   work for this and run on GPU.

**Gotcha:** terrain colliders in Isaac Lab are static after init. To get
per-episode randomization, build a *bank* of heightfields at startup and
let the `EventTerm` switch the rover's spawn position to a different patch
each reset. This is how Isaac Lab does it for locomotion benchmarks.

---

## Phase 4 — LQR controller (vectorized)

**Goal:** the Phase 2/3 task runs with hybrid action (LQR + residual)
matching the PyBullet hybrid configuration.

**Deliverable:** `Isaac-LeoRover-Mars-Hybrid-v0` task — same as Phase 3 but
with a zero-init residual on top of GPU-side LQR.

**Steps:**

1. Port `Controller2._compute_lqr_action()` to a PyTorch function operating
   on `[num_envs, state_dim]` tensors. The LQR gains `K` are constant
   `[2, 3]` matrices — pre-compute once at task init and reuse on every
   step.
2. Inside `_apply_action`, split incoming policy action into baseline_gain +
   residual_scale. Compute LQR action from state, scale residual by
   `MAX_RESIDUAL_VELOCITY` / `MAX_RESIDUAL_OMEGA`, clip, add.
3. Add the v33.9 `r_effort` reward term — straight port of the L2 magnitude
   penalty.
4. Mirror the rest of the v33.9 reward weights: PPO_W_CTE=5.0,
   PPO_W_PROGRESS=10.0, etc. Each becomes a `RewardTerm` with that weight.

**Gotcha:** LQR's reference state is the waypoint frame. Make sure you
compute the body-frame error using torch ops, not converting back and forth
to numpy — every numpy round-trip kills throughput.

---

## Phase 5 — Curriculum + ADR

**Goal:** ADR difficulty progression matches the behavior of the PyBullet
`adr_curriculum.py`.

**Deliverable:** training run shows terrain_intensity ceiling rising as
success rate clears the threshold, identical curriculum shape to PyBullet.

**Steps:**

1. Port the ADR controller from `adr_curriculum.py`. Most of the logic is
   pure Python and doesn't need vectorizing — it operates on aggregate
   success/CTE statistics, not per-env tensors. Plug it in as a
   `CurriculumTerm`.
2. Mirror the v33.9 mode-conditional thresholds (the hybrid-specific
   ADR_TERRAIN_MAX_START=30, ADR_SUCCESS_THRESHOLD=0.75 block).
3. Wire ADR into the `EventTerm` controlling terrain_intensity so the
   curriculum actually affects the simulation.

---

## Phase 6 — Logging + GUI parity

**Goal:** the new system produces the same per-episode CSV columns as the
PyBullet `episode_metrics.csv`, so `evaluate_training.py` from the
PyBullet repo can analyze Isaac Lab runs.

**Deliverable:** drop an Isaac Lab CSV into the PyBullet GUI and see all
the existing plots work — including the v33.9 residual analysis.

**Steps:**

1. Implement a `Recorder` callback that accumulates per-episode metrics
   into the same column schema as `EpisodeMetricsCallback` in
   `run_experiment.py`.
2. The residual columns (`mean_residual_v_norm`, `mean_residual_w_norm`)
   should come for free since you're computing them anyway for the
   r_effort term.
3. Set up wandb for live training graphs in parallel — Isaac Lab makes this
   trivial.

---

## Phase 7 — Comparison evaluation

**Goal:** equivalent of the PyBullet `run_comparison` mode — head-to-head
LQR vs Hybrid on paired terrain seeds.

**Deliverable:** a script that loads a trained Hybrid checkpoint and runs
the same N episodes on both Hybrid and pure-LQR (zero-residual) policies,
producing a paired-comparison CSV.

**Steps:**

1. Write `scripts/compare_hybrid_vs_lqr.py` that builds two env instances
   with the same terrain seeds, one running the trained policy, one
   forcing residual=0 (pure LQR).
2. Output to the same CSV schema so the existing PyBullet comparison
   analysis pipeline works unchanged.

---

## Phase 8 — Sim2Real prep (optional, much later)

If the URCA paper evolves into "deploy on the actual Leo Rover hardware,"
Isaac Lab has tooling for this that PyBullet doesn't:

- **Domain randomization** at GPU speed (orders of magnitude more
  diversity per training run)
- **Real2Sim** for matching observed real-world friction/inertia to sim
- **Direct policy export** to ONNX for deployment on edge hardware
  (Jetson, the rover's onboard compute)

This is a research direction more than a port phase — note it for the
future, don't plan against it now.

---

## What to NOT port

Some pieces of the PyBullet stack don't have meaningful Isaac Lab
equivalents and shouldn't be ported:

- **`comparison_workers.py`** — used `multiprocessing.Pool` to parallelize
  Compare mode across CPU cores. In Isaac Lab, parallelism is built into
  the env itself; you don't need worker pools.
- **`experiment_gui.py`** — Isaac Lab is CLI-first and wandb/tensorboard
  handle the live-monitoring use case. Building a Tk GUI on top would be
  fighting the framework.
- **`SafeMlpPolicy` clamp** — Isaac Lab's PPO implementations have proper
  log_std bounds in their config; the v31 fix doesn't need to migrate.
- **`PerformanceRollbackCallback`** — Isaac Lab's training is more stable
  by default (proper observation normalization, GPU-side advantage
  computation), and rsl_rl has built-in checkpoint rotation.

---

## Estimated effort

This is a side project, not a sprint. Realistic part-time pace:

| phase | scope | calendar time |
|-------|-------|---------------|
| 0 — install | small | 1 weekend |
| 1 — asset port | medium | 1–2 weeks |
| 2 — flat task | medium | 1–2 weeks |
| 3 — Mars terrain | large | 2–4 weeks |
| 4 — LQR + hybrid | medium | 1–2 weeks |
| 5 — curriculum | small | 1 week |
| 6 — logging | small | 1 week |
| 7 — comparison | small | 1 week |

Total: roughly **2–3 months of part-time work** for a complete port. Don't
let the calendar pressure you — there's no deadline here, and rushing the
asset port (Phase 1) almost always backfires.

---

## How to know the port is "done"

The port is complete when this single test passes:

> Train a Hybrid PPO model in Isaac Lab to the same success-rate ceiling
> as the best PyBullet run. Run Compare-Hybrid-vs-LQR on the same fixed
> terrain seeds. The aggregate success rates should match within 2 pp.

If the Isaac Lab version diverges from PyBullet by more than 2 pp, the
contact physics or terrain randomization is different in a way that
matters — track it down before claiming parity. Some divergence is
inevitable (PyBullet uses an explicit-stepping LCP solver, Isaac uses
PhysX TGS), so 2pp is the realistic ceiling for "same."

After that, the value proposition of Isaac Lab over PyBullet is clear:
roughly 100x faster iteration, GPU-resident randomization, and a cleaner
path to either sim2real or extension to multi-robot scenarios.
