# envs/ — Isaac Lab environment classes

Each module here corresponds to one of the PyBullet `MyEnv2` configurations.
They subclass `isaaclab.envs.DirectRLEnv` and define the env-specific scene
setup, action application, observation construction, reward computation,
and termination conditions.

## What lives here (by port phase)

| file | port phase | corresponds to in PyBullet |
|------|------------|----------------------------|
| `leo_rover_flat_env.py` | Phase 2 | `MyEnv2` configured with flat ground |
| `leo_rover_mars_env.py` | Phase 3 | `MyEnv2` with Mars heightfield + ADR |
| `leo_rover_mars_hybrid_env.py` | Phase 4 | `MyEnv2(use_lqr_baseline=True, use_pure_ppo_reward=True)` |

## Key API differences from PyBullet

In PyBullet, `MyEnv2.step(action)` ran one env in one Python loop. In Isaac
Lab, the framework calls these hooks once per simulation step, and each
returns a **tensor of shape `[num_envs, ...]`** covering all parallel envs
simultaneously:

| PyBullet hook | Isaac Lab hook(s) |
|---------------|-------------------|
| `MyEnv2._compute_action(action)` (controller integration) | `_pre_physics_step(action)` then `_apply_action()` |
| `MyEnv2._get_observation()` | `_get_observations()` |
| `MyEnv2._compute_reward(...)` | `_get_rewards()` |
| `MyEnv2._check_done(...)` | `_get_dones()` returning `(terminated, truncated)` |
| `MyEnv2.reset(...)` | `_reset_idx(env_ids)` — resets only the envs in `env_ids` |

The single biggest mental shift: **never loop over envs**. Every quantity
is a torch tensor with a leading num_envs dimension. CTE, heading error,
reward, residual magnitude — all `[num_envs]`-shaped tensors on `cuda:0`.

## Mapping the v33.9 reward terms

Each PyBullet reward term becomes a separately-registered `RewardTerm` in
the Isaac Lab task config:

```python
# leorover_isaac/tasks/leo_rover_mars_hybrid_cfg.py
rewards = RewardManagerCfg(
    cte         = RewTerm(func=mdp.cte_reward,     weight=5.0),    # PPO_W_CTE
    heading     = RewTerm(func=mdp.heading_reward, weight=0.5),
    velocity    = RewTerm(func=mdp.velocity_reward,weight=0.5),
    smoothness  = RewTerm(func=mdp.smoothness_rew, weight=0.5),
    progress    = RewTerm(func=mdp.progress_rew,   weight=10.0),
    effort      = RewTerm(func=mdp.l2_residual,    weight=0.5),    # v33.9 NEW
    success_bonus  = RewTerm(func=mdp.success_bonus,  weight=200.0),
    failure_penalty= RewTerm(func=mdp.failure_penalty,weight=50.0),
)
```

Keep the weights identical to v33.9. If the port produces different
behavior at parity weights, the divergence is in physics/observation, not
reward design — investigate physics before tuning weights.
