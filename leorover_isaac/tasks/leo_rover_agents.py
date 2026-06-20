# leo_rover_agents.py
"""
rsl_rl PPO agent configs for the Leo Rover tasks.

These mirror the PyBullet v33.9 PPO hyperparameters from config.py as closely as
the rsl_rl framework allows. Where the two frameworks differ structurally, the
difference is called out inline:

  * num_steps_per_env: rsl_rl collects this many steps PER ENV per update. With
    ~4096 envs that is a far larger batch than PyBullet's n_steps=4096 over 12
    envs, so we use 32 (32*4096 ~= 131 k transitions/update). The PyBullet
    n_steps does not translate directly because the parallelism model differs.
  * LayerNorm: the v30 anti-plasticity LayerNorm extractor isn't a drop-in for
    rsl_rl's MLP actor-critic; rsl_rl is far more stable at this scale (proper
    obs normalization + GPU advantage). Noted as a known, accepted divergence
    (PORTING_ROADMAP.md "What to NOT port").
  * SafeMlpPolicy log_std clamp: reproduced via noise_std_type="log". rsl-rl-lib
    defaults to "scalar" (std is a raw learnable parameter that a gradient step
    CAN push below zero -> `RuntimeError: normal expects all elements of std >=
    0.0`, exactly the v31 log_std-runaway failure). "log" parameterizes
    std = exp(log_std), so it is strictly positive and can never crash — the
    direct rsl_rl analogue of the SafeMlpPolicy clamp.

Everything that DOES translate (gamma, gae lambda, clip, entropy, lr, epochs,
value loss coef, max grad norm, net arch) is copied from config.py verbatim.
"""

from __future__ import annotations

import config as cfg_mod
import os as _os

# Shell-tunable training knobs (no code edit) for fixing the from-scratch stall:
#   LEOROVER_NUM_STEPS  rollout length per env (default 32). The +200 success lands
#                       ~1400 steps in, so a 32-step rollout never contains it and the
#                       terminal reward can't propagate; PyBullet used full-episode
#                       rollouts. Raise this (e.g. 96-256) so the value fn sees further.
#   LEOROVER_ENT_COEF   PPO entropy coef (default PPO_ENT_COEF=0.001). Raise (e.g.
#                       0.005-0.01) to hold exploration so std doesn't collapse to
#                       ~0.02 before the policy learns to drive.
_NUM_STEPS_PER_ENV = int(_os.environ.get("LEOROVER_NUM_STEPS", 32))
_ENT_COEF = float(_os.environ.get("LEOROVER_ENT_COEF", cfg_mod.PPO_ENT_COEF))

# rsl_rl config dataclasses moved namespaces across Isaac Lab versions.
try:
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
    )
    _RSL = True
except Exception:
    try:  # older Isaac Lab
        from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (  # type: ignore
            RslRlOnPolicyRunnerCfg,
            RslRlPpoActorCriticCfg,
            RslRlPpoAlgorithmCfg,
        )
        _RSL = True
    except Exception:
        _RSL = False

# configclass moved namespaces too (isaaclab.utils vs omni.isaac.lab.utils).
try:
    from isaaclab.utils import configclass
except Exception:
    from omni.isaac.lab.utils import configclass  # Isaac Sim 4.5 / Isaac Lab 1.x


def _supported(cls, **kw):
    """Keep only the kwargs that `cls` (a config dataclass) actually defines, so one
    call site works across rsl-rl versions whose cfg schemas differ (4.5's
    RslRlPpoActorCriticCfg has no noise_std_type / *_obs_normalization fields)."""
    names = set()
    try:
        import dataclasses
        names = {f.name for f in dataclasses.fields(cls)}
    except Exception:
        for base in getattr(cls, "__mro__", [cls]):
            names |= set(getattr(base, "__annotations__", {}).keys())
    return {k: v for k, v in kw.items() if k in names}


if _RSL:
    _RUNNER_FIELDS = set()
    try:
        import dataclasses as _dc
        _RUNNER_FIELDS = {f.name for f in _dc.fields(RslRlOnPolicyRunnerCfg)}
    except Exception:
        pass

    # Built with field-filtering so the same definition loads on both the new
    # (Isaac Sim 5.x / rsl-rl 5.x) and old (Isaac Sim 4.5 / rsl-rl 2.x) schemas.
    _POLICY = RslRlPpoActorCriticCfg(**_supported(
        RslRlPpoActorCriticCfg,
        init_noise_std=0.37,                  # = exp(PPO_LOG_STD_INIT=-1.0), the v31 init
        noise_std_type="log",                 # 5.x: std=exp(log_std), strictly >0 (v31 clamp)
        actor_obs_normalization=True,         # 5.x obs normalization (== VecNormalize)
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256],         # PPO_POLICY_KWARGS net_arch
        critic_hidden_dims=[256, 256],
        activation="relu",                    # PURE_PPO_POLICY_KWARGS ReLU
    ))
    _ALGO = RslRlPpoAlgorithmCfg(**_supported(
        RslRlPpoAlgorithmCfg,
        value_loss_coef=cfg_mod.PPO_VF_COEF,           # 0.5
        use_clipped_value_loss=True,
        clip_param=cfg_mod.PPO_CLIP_RANGE,             # 0.2
        entropy_coef=_ENT_COEF,                        # PPO_ENT_COEF=0.001; LEOROVER_ENT_COEF overrides
        num_learning_epochs=cfg_mod.PPO_N_EPOCHS,      # 5
        num_mini_batches=4,
        learning_rate=cfg_mod.PPO_LEARNING_RATE,       # 1.5e-4
        schedule="adaptive",                            # KL-adaptive
        gamma=cfg_mod.PPO_GAMMA,                        # 0.99
        lam=cfg_mod.PPO_GAE_LAMBDA,                     # 0.95
        desired_kl=cfg_mod.PPO_TARGET_KL,               # 0.02
        max_grad_norm=cfg_mod.PPO_MAX_GRAD_NORM,        # 0.5
    ))

    @configclass
    class LeoRoverPPORunnerCfg(RslRlOnPolicyRunnerCfg):
        num_steps_per_env = _NUM_STEPS_PER_ENV   # default 32; LEOROVER_NUM_STEPS overrides
        max_iterations = 30000          # ~ matches the multi-million-step PyBullet runs
        save_interval = 200
        experiment_name = "leo_rover"
        policy = _POLICY
        algorithm = _ALGO
        # On Isaac Sim 4.5 / rsl-rl 2.x obs normalization is a runner-level flag
        # (the 5.x path sets it on the policy above). Only define it when present.
        if "empirical_normalization" in _RUNNER_FIELDS:
            empirical_normalization = True

    @configclass
    class LeoRoverFlatPPORunnerCfg(LeoRoverPPORunnerCfg):
        experiment_name = "leo_rover_flat"

    @configclass
    class LeoRoverMarsPPORunnerCfg(LeoRoverPPORunnerCfg):
        experiment_name = "leo_rover_mars"

    @configclass
    class LeoRoverMarsHybridPPORunnerCfg(LeoRoverPPORunnerCfg):
        experiment_name = "leo_rover_mars_hybrid"
else:  # pragma: no cover
    LeoRoverPPORunnerCfg = None
    LeoRoverFlatPPORunnerCfg = None
    LeoRoverMarsPPORunnerCfg = None
    LeoRoverMarsHybridPPORunnerCfg = None
