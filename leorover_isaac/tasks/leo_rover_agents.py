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

from isaaclab.utils import configclass


if _RSL:
    @configclass
    class LeoRoverPPORunnerCfg(RslRlOnPolicyRunnerCfg):
        num_steps_per_env = 32
        max_iterations = 30000          # ~ matches the multi-million-step PyBullet runs
        save_interval = 200
        experiment_name = "leo_rover"

        policy = RslRlPpoActorCriticCfg(
            init_noise_std=0.37,                  # = exp(PPO_LOG_STD_INIT=-1.0), the v31 init
            noise_std_type="log",                 # std=exp(log_std): strictly >0, cannot crash.
                                                  # (default "scalar" let std go negative ->
                                                  #  RuntimeError at iter 4247. This is the v31
                                                  #  SafeMlpPolicy log_std clamp, rsl_rl-style.)
            actor_obs_normalization=True,         # == PyBullet VecNormalize(norm_obs=True)
            critic_obs_normalization=True,
            actor_hidden_dims=[256, 256],         # PPO_POLICY_KWARGS net_arch
            critic_hidden_dims=[256, 256],
            activation="relu",                    # PURE_PPO_POLICY_KWARGS ReLU
        )
        algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=cfg_mod.PPO_VF_COEF,           # 0.5
            use_clipped_value_loss=True,
            clip_param=cfg_mod.PPO_CLIP_RANGE,             # 0.2
            entropy_coef=cfg_mod.PPO_ENT_COEF,             # 0.001
            num_learning_epochs=cfg_mod.PPO_N_EPOCHS,      # 5
            num_mini_batches=4,
            learning_rate=cfg_mod.PPO_LEARNING_RATE,       # 1.5e-4
            schedule="adaptive",                            # KL-adaptive (replaces SafeMlpPolicy)
            gamma=cfg_mod.PPO_GAMMA,                        # 0.99
            lam=cfg_mod.PPO_GAE_LAMBDA,                     # 0.95
            desired_kl=cfg_mod.PPO_TARGET_KL,               # 0.02
            max_grad_norm=cfg_mod.PPO_MAX_GRAD_NORM,        # 0.5
        )

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
