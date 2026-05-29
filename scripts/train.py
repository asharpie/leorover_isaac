"""train.py — Phase 2 stub.

Thin wrapper around Isaac Lab's rsl_rl train script with leorover-specific
defaults. Once Phase 2 is implemented, this will:

  1. Parse a --task argument (Isaac-LeoRover-Flat-v0, etc.)
  2. Build the matching env config
  3. Hand off to rsl_rl's OnPolicyRunner with PPO defaults that mirror
     the PyBullet v33.9 hyperparameters
  4. Set up wandb logging
  5. Dump the per-episode CSV in the PyBullet-compatible schema

Until then, use Isaac Lab's bundled train script directly:

    ~/IsaacLab/isaaclab.sh -p ~/IsaacLab/source/standalone/workflows/rsl_rl/train.py \\
        --task Isaac-Cartpole-v0 --num_envs 64 --headless

(This is for verifying the install — Phase 0 smoke test in INSTALL.md.)
"""

if __name__ == "__main__":
    raise SystemExit(
        "scripts/train.py is a Phase 2 stub. See INSTALL.md for the temporary "
        "Isaac Lab smoke-test command and PORTING_ROADMAP.md for what's next."
    )
