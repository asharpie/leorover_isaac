# Simulation-Phase Checklist (updated 2026-07-13)

## DONE — proven and in the paper
- [x] Terramechanics-lite soil layer (sinkage drag + slip-thrust decay + lateral shear, seeded zones) + per-wheel slip/sinkage observation. SCM-style force injection, cited (Krenn & Hirzinger).
- [x] Hybrid trained IN the soil model (model_25400, 3.3B steps, 2.2 days): training world **-47% CTE, +4.1 pts success, both metrics at every level**.
- [x] Rigid-contact control condition: LQR near-optimal there (+2.2 pts, d=0.05, hairpins only) — the fidelity-gates-residual finding.
- [x] Scenario-locked paired protocol: path + terrain + spawn + **per-wheel friction** in one shared scenarios.npz (friction pairing verified 100.00% match across legs).
- [x] Sixteen-world generalization suite (192 fresh patches, 2000-path bank, mu 0.47-2.0): **CTE lower 16/16 worlds (-16%), success higher 15/16**, n=185,410 pairs.
- [x] Mechanism identified: **traction dose-response** — zero gap at high grip, -9% mid, -21% low mu; slip tertiles +11.9 pts / -59% CTE; residual = steering-dominant corrector.
- [x] Training-stability recipe: std clamp [0.05,0.6], effort 0.01 + credit 15 + slip-gated v-exemption, uniform difficulty for the protected residual. All are zero-flag defaults now.
- [x] Paper v2: all real results, honest confounds (episode cap, world variance), 5 data figures generated (paper/figs/).

## REMAINING — simulation side, in order
- [ ] **1. Retrain hybrid with the velocity channel freed** (champion predates the slip-gated
      exemption; its residual is steering-only, res_v ~0.03). Plain `leo train hybrid` now
      carries the full recipe. Health tells: res_v climbs off 0.03 while res_w holds ~0.13;
      det-eval success > 81%. ~2 days.
- [ ] 2. Re-evaluate the new checkpoint: `leo multieval --worlds 16 --n 12000` +
      `leo quickeval hybrid` / `leo quickeval lqr` / `leo compare`. If better, refresh paper
      numbers + figures.
- [ ] 3. **Pure PPO on sand**: `leo train ppo` (obs 23, ADR ramp retained). Fills the PPO
      columns + the RQ2 sample-efficiency ratio. Run after #2 (one GPU).
- [ ] 4. Optional multi-seed fine-tune (attacks the -47% vs -16% familiarity gap):
      `LEOROVER_TERRAIN_SEED=301 leo train hybrid --resume` style sequential fine-tuning,
      a few thousand iters per seed, then re-run multieval.
- [ ] 5. 800-s cap rerun of the discrete geometry suite (de-confound completion):
      `LEOROVER_EPISODE_S=800 leo eval --n 90000`.
- [ ] 6. Close the paper's \verify{} items (10 min on box):
      gravity setting (`grep -rn "gravity" leorover_isaac/envs/leo_rover_base_env.py config.py`),
      remaining reward weights (`python3 -c "import config as c; print({k:getattr(c,k) for k in dir(c) if k.startswith('PPO_W') or 'BONUS' in k or 'PENALTY' in k})"`),
      lr / target-KL (`grep -n "learning_rate\|desired_kl" leorover_isaac/tasks/leo_rover_agents.py`).
- [ ] 7. Isaac-rendered figures: sim screenshot with paths overlaid, soil-zone map render,
      representative trajectories (2x2 grid, slip color-coded).
- [ ] 8. Optional metrics/model polish: time-normalized completion metric; spatial Coulomb-mu
      via GeomSubsets; load transfer / rut memory (deliberately deferred to post-calibration).

## HARDWARE PHASE (next)
- [ ] Sand-pit system identification (constant-throttle slip ramps, drawbar pull) ->
      calibrate LEOROVER_SOIL_* coefficients -> retrain -> deploy (Marvelmind + ROS 2).
