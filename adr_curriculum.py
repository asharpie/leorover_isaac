"""
Automatic Domain Randomization (ADR) Curriculum for Hybrid Residual RL.

Based on OpenAI's ADR (Akkaya et al., 2019) adapted for terrain difficulty.

The key idea: terrain difficulty (slope intensity) starts low and increases
ONLY when the agent achieves a performance threshold over a rolling window.
If performance drops, difficulty can also decrease (bidirectional ADR).

This replaces fixed curriculum stages with adaptive progression.

References:
  - Akkaya et al. (2019) "Solving Rubik's Cube with a Robot Hand" (ADR)
  - Cobbe et al. (2020) "Leveraging Procedural Generation..."
  - Xie et al. (2020) "Iterative Residual Policy Learning"
"""

import os
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
try:
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:  # SB3 not installed in the Isaac env — ADRCurriculum still works.
    class BaseCallback:  # minimal shim; the SB3 ADRCallback below is unused under rsl_rl
        def __init__(self, *a, **k):
            pass


@dataclass
class ADRConfig:
    """Configuration for Automatic Domain Randomization curriculum."""

    # --- Terrain slope progression ---
    terrain_intensity_min: float = 0.0       # Starting lower bound (always samples from here)
    terrain_intensity_max_start: float = 15.0  # Starting upper bound (gentle hills)
    terrain_intensity_max_limit: float = 100.0  # Maximum possible upper bound

    # --- Friction progression (co-progresses with terrain) ---
    friction_intensity_min: float = 30.0
    friction_intensity_max_start: float = 70.0
    friction_intensity_max_limit: float = 100.0

    # --- Performance thresholds for progression ---
    # Agent must achieve ALL of these over the rolling window to advance
    success_rate_threshold: float = 0.80     # 80% goal completion
    mean_cte_threshold: float = 0.045        # Mean CTE < 4.5cm
    # If performance drops below these, difficulty decreases
    regression_success_threshold: float = 0.50  # Drop below 50% -> decrease
    regression_cte_threshold: float = 0.50      # CTE > 50cm -> decrease

    # --- Progression dynamics ---
    eval_window_size: int = 100         # Rolling window of episodes to evaluate
    intensity_step_up: float = 5.0      # Increase upper bound by 5% per advance
    intensity_step_down: float = 3.0    # Decrease by 3% per regression (slower)
    min_episodes_per_level: int = 20    # Minimum episodes before allowing change
    cooldown_episodes: int = 10         # Episodes to wait after any change

    # --- Coupled rotation ---
    # How many episodes to repeat the same (path, terrain, friction) config
    episodes_per_configuration: int = 4


class ADRCurriculum:
    """
    Manages automatic difficulty progression and coupled episode rotation.

    At each difficulty level, terrain is sampled uniformly from
    [terrain_intensity_min, current_terrain_max]. This ensures the agent
    revisits easy terrain even at high difficulty (prevents catastrophic
    forgetting of easy cases).

    Episode rotation: each (path, terrain_seed, friction) configuration is
    repeated for `episodes_per_configuration` episodes before rotating to a
    new random configuration.  This gives SAC's replay buffer locally
    consistent transitions while still ensuring diversity.

    Usage:
        adr = ADRCurriculum(config)

        # Each reset():
        should_rotate = adr.should_rotate_configuration()
        if should_rotate:
            terrain = adr.sample_terrain_intensity()
            friction = adr.sample_friction_intensity()
            terrain_seed = np.random.randint(0, 2**31)
            # ... use new config
        # else: reuse previous config

        # After episode ends:
        adr.report_episode(success=True, mean_cte=0.15)
    """

    def __init__(self, config: Optional[ADRConfig] = None):
        self.config = config or ADRConfig()

        # Current difficulty boundaries
        self.terrain_max = self.config.terrain_intensity_max_start
        self.friction_max = self.config.friction_intensity_max_start

        # Rolling performance window
        self.success_history = deque(maxlen=self.config.eval_window_size)
        self.cte_history = deque(maxlen=self.config.eval_window_size)

        # Tracking
        self.episodes_at_current_level = 0
        self.cooldown_remaining = 0
        self.total_episodes = 0
        self.difficulty_history = []  # (episode, terrain_max, friction_max, event)
        self.num_advances = 0
        self.num_regressions = 0

        # Coupled rotation tracking
        self._episodes_in_current_config = 0
        self._current_terrain_intensity = None
        self._current_friction_intensity = None
        self._current_terrain_seed = None

    def should_rotate_configuration(self) -> bool:
        """Check if it's time to rotate to a new (path, terrain, friction) config."""
        if self._current_terrain_intensity is None:
            # First episode ever — need initial config
            return True
        return self._episodes_in_current_config >= self.config.episodes_per_configuration

    def sample_terrain_intensity(self) -> float:
        """Sample terrain intensity from current ADR range and store it."""
        self._current_terrain_intensity = np.random.uniform(
            self.config.terrain_intensity_min,
            self.terrain_max
        )
        self._current_terrain_seed = np.random.randint(0, 2 ** 31)
        self._episodes_in_current_config = 0
        return self._current_terrain_intensity

    def sample_friction_intensity(self) -> float:
        """Sample friction intensity from current ADR range and store it."""
        self._current_friction_intensity = np.random.uniform(
            self.config.friction_intensity_min,
            self.friction_max
        )
        return self._current_friction_intensity

    def get_current_terrain_intensity(self) -> float:
        """Get the stored terrain intensity for this configuration block."""
        return self._current_terrain_intensity

    def get_current_friction_intensity(self) -> float:
        """Get the stored friction intensity for this configuration block."""
        return self._current_friction_intensity

    def get_current_terrain_seed(self) -> int:
        """Get the stored terrain seed for this configuration block."""
        return self._current_terrain_seed

    def report_episode(self, success: bool, mean_cte: float):
        """Report episode results for curriculum evaluation."""
        self.success_history.append(1.0 if success else 0.0)
        self.cte_history.append(mean_cte)
        self.episodes_at_current_level += 1
        self.total_episodes += 1
        self._episodes_in_current_config += 1

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def should_update_difficulty(self) -> bool:
        """Check if we have enough data and are past cooldown."""
        if len(self.success_history) < self.config.eval_window_size:
            return False
        if self.episodes_at_current_level < self.config.min_episodes_per_level:
            return False
        if self.cooldown_remaining > 0:
            return False
        return True

    def update_difficulty(self) -> str:
        """
        Evaluate performance and adjust difficulty.

        Returns:
            "advance" if difficulty increased,
            "regress" if difficulty decreased,
            "hold" if no change
        """
        if not self.should_update_difficulty():
            return "hold"

        success_rate = np.mean(list(self.success_history))
        mean_cte = np.mean(list(self.cte_history))

        # Check for advancement
        if (success_rate >= self.config.success_rate_threshold and
                mean_cte <= self.config.mean_cte_threshold):

            old_terrain = self.terrain_max
            self.terrain_max = min(
                self.terrain_max + self.config.intensity_step_up,
                self.config.terrain_intensity_max_limit
            )
            # Friction advances at half the rate of terrain
            self.friction_max = min(
                self.friction_max + self.config.intensity_step_up * 0.5,
                self.config.friction_intensity_max_limit
            )

            self.num_advances += 1
            self.episodes_at_current_level = 0
            self.cooldown_remaining = self.config.cooldown_episodes
            self.difficulty_history.append(
                (self.total_episodes, self.terrain_max, self.friction_max, "advance")
            )

            print(f"\n  [ADR] ADVANCE: terrain_max {old_terrain:.0f}% -> {self.terrain_max:.0f}% "
                  f"(success={success_rate:.0%}, CTE={mean_cte:.3f})")
            return "advance"

        # Check for regression
        elif (success_rate < self.config.regression_success_threshold or
              mean_cte > self.config.regression_cte_threshold):

            old_terrain = self.terrain_max
            self.terrain_max = max(
                self.terrain_max - self.config.intensity_step_down,
                self.config.terrain_intensity_max_start  # Never go below starting difficulty
            )
            self.friction_max = max(
                self.friction_max - self.config.intensity_step_down * 0.5,
                self.config.friction_intensity_max_start
            )

            self.num_regressions += 1
            self.episodes_at_current_level = 0
            self.cooldown_remaining = self.config.cooldown_episodes
            self.difficulty_history.append(
                (self.total_episodes, self.terrain_max, self.friction_max, "regress")
            )

            print(f"\n  [ADR] REGRESS: terrain_max {old_terrain:.0f}% -> {self.terrain_max:.0f}% "
                  f"(success={success_rate:.0%}, CTE={mean_cte:.3f})")
            return "regress"

        return "hold"

    def get_progress_pct(self) -> float:
        """Current difficulty as percentage of maximum."""
        total_range = self.config.terrain_intensity_max_limit - self.config.terrain_intensity_max_start
        if total_range <= 0:
            return 100.0
        current_range = self.terrain_max - self.config.terrain_intensity_max_start
        return (current_range / total_range) * 100.0

    def get_stats(self) -> dict:
        """Get curriculum statistics."""
        return {
            "total_episodes": self.total_episodes,
            "current_terrain_max": self.terrain_max,
            "current_friction_max": self.friction_max,
            "num_advances": self.num_advances,
            "num_regressions": self.num_regressions,
            "progress_pct": self.get_progress_pct(),
            "current_success_rate": (
                float(np.mean(list(self.success_history)))
                if self.success_history else 0.0
            ),
            "current_mean_cte": (
                float(np.mean(list(self.cte_history)))
                if self.cte_history else 0.0
            ),
            "episodes_in_current_config": self._episodes_in_current_config,
            "episodes_per_configuration": self.config.episodes_per_configuration,
        }


class ADRCallback(BaseCallback):
    """
    Stable-Baselines3 callback that integrates ADR curriculum with training.

    This callback:
    1. After each episode, reports success/CTE to the ADR controller
    2. Updates environment terrain/friction ranges when ADR advances/regresses
    3. Logs curriculum progression to CSV

    The coupled rotation logic is handled inside the environment's reset()
    method via the ADRCurriculum object stored on the environment.
    """

    def __init__(self, adr: ADRCurriculum, log_dir: str, warmup_steps: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.adr = adr
        self.log_dir = log_dir
        self.warmup_steps = warmup_steps

        # Per-environment episode tracking (dict keyed by env index)
        self._env_cte_sum = {}   # {env_idx: float}
        self._env_steps = {}     # {env_idx: int}

        # CSV logging
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "adr_curriculum.csv")
        with open(self.csv_path, 'w') as f:
            f.write("episode,terrain_max,friction_max,success_rate,mean_cte,"
                    "advances,regressions,progress_pct\n")

    def _push_limits_to_envs(self):
        """
        Push updated terrain/friction limits to all subprocess environments.

        With SubprocVecEnv, each env is a separate process with its own copy
        of the ADRCurriculum object. Updating self.adr.terrain_max in the main
        process does NOT propagate to the subprocesses. We must explicitly
        call a method on each env to sync the limits.
        """
        vec_env = self.training_env
        n_envs = vec_env.num_envs
        new_terrain_max = self.adr.terrain_max
        new_friction_max = self.adr.friction_max

        try:
            # env_method passes through Monitor wrapper to the underlying MyEnv2
            vec_env.env_method(
                'update_adr_limits',
                new_terrain_max,
                new_friction_max,
            )
            if self.verbose > 0:
                print(f"  [ADR] Pushed limits to {n_envs} envs: "
                      f"terrain_max={new_terrain_max:.0f}%, "
                      f"friction_max={new_friction_max:.0f}%")
        except Exception as e:
            # Fallback for DummyVecEnv: direct attribute access
            try:
                if hasattr(vec_env, 'envs'):
                    for i in range(len(vec_env.envs)):
                        unwrapped = vec_env.envs[i]
                        while hasattr(unwrapped, 'env'):
                            unwrapped = unwrapped.env
                        if hasattr(unwrapped, '_adr'):
                            unwrapped._adr.terrain_max = new_terrain_max
                            unwrapped._adr.friction_max = new_friction_max
                        unwrapped._terrain_intensity_max = new_terrain_max
                        unwrapped._friction_intensity_max = new_friction_max
                    if self.verbose > 0:
                        print(f"  [ADR] Pushed limits (DummyVecEnv fallback): "
                              f"terrain_max={new_terrain_max:.0f}%")
            except Exception as e2:
                print(f"  [ADR] WARNING: Could not push limits to envs: {e}, {e2}")

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        dones = self.locals.get('dones', [False])
        n_envs = len(infos)

        for env_idx in range(n_envs):
            info = infos[env_idx]
            # Initialize per-env accumulators if needed
            if env_idx not in self._env_cte_sum:
                self._env_cte_sum[env_idx] = 0.0
                self._env_steps[env_idx] = 0

            # With SubprocVecEnv, when done=True SB3 auto-resets the env.
            # The terminal step's info dict is nested under 'terminal_info'.
            # Without unwrapping, custom keys (cross_track_error, goal_reached,
            # flipped) read as defaults (0.0/False) and ADR gets bad data.
            if dones[env_idx] and 'terminal_info' in info:
                ep_info = info['terminal_info']
            else:
                ep_info = info

            cte = ep_info.get('cross_track_error', 0.0)
            self._env_cte_sum[env_idx] += abs(cte)
            self._env_steps[env_idx] += 1

            if dones[env_idx]:
                # Skip flipped episodes — they are discarded and retried
                if ep_info.get('flipped', False):
                    self._env_cte_sum[env_idx] = 0.0
                    self._env_steps[env_idx] = 0
                    continue

                # Skip episodes during warm-up — pure LQR performance would
                # artificially inflate success rate and cause premature
                # difficulty advancement before the residual even starts learning
                if self.num_timesteps < self.warmup_steps:
                    if self.verbose > 0 and self.adr.total_episodes == 0:
                        print(f"  [ADR] Ignoring warm-up episodes (step {self.num_timesteps}/{self.warmup_steps})")
                    self._env_cte_sum[env_idx] = 0.0
                    self._env_steps[env_idx] = 0
                    continue

                # Report to ADR
                success = ep_info.get('goal_reached', False)
                mean_cte = self._env_cte_sum[env_idx] / max(1, self._env_steps[env_idx])
                self.adr.report_episode(success=success, mean_cte=mean_cte)

                # Try to update difficulty
                if self.adr.should_update_difficulty():
                    result = self.adr.update_difficulty()

                    if result in ("advance", "regress"):
                        # SubprocVecEnv: each env is a separate process with its
                        # own ADRCurriculum copy. Must explicitly push the new
                        # limits to every subprocess environment.
                        self._push_limits_to_envs()

                # Log to CSV every 10 episodes
                stats = self.adr.get_stats()
                if stats['total_episodes'] % 10 == 0:
                    with open(self.csv_path, 'a') as f:
                        f.write(f"{stats['total_episodes']},{stats['current_terrain_max']:.1f},"
                                f"{stats['current_friction_max']:.1f},"
                                f"{stats['current_success_rate']:.3f},"
                                f"{stats['current_mean_cte']:.4f},"
                                f"{stats['num_advances']},{stats['num_regressions']},"
                                f"{stats['progress_pct']:.1f}\n")

                    if self.verbose >= 1:
                        print(f"\n  [ADR] Episode {stats['total_episodes']}: "
                              f"terrain_max={stats['current_terrain_max']:.0f}%, "
                              f"success={stats['current_success_rate']:.0%}, "
                              f"CTE={stats['current_mean_cte']:.3f}, "
                              f"progress={stats['progress_pct']:.0f}%")

                # Reset per-env episode accumulators
                self._env_cte_sum[env_idx] = 0.0
                self._env_steps[env_idx] = 0

        return True
