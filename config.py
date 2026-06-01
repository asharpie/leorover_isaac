# config.py
"""
Configuration module for training and screening experiments.


REWARD v14 - COST-BASED REWARD FUNCTION


Replaces the counterfactual reward (v9-v13) with a purely negative per-step
cost structure that eliminates the alive bias. Every timestep costs the agent
something; the only way to achieve positive total return is to reach the goal.

Based on analysis of ALL successful residual RL implementations:
- Johannink et al. (2018): purely negative dense rewards
- Silver et al. (2018): sparse binary or cost-based rewards
- Trumpp et al. (2023, RPL4F110): near-zero per-step, huge collision penalty
- Schaff & Walter (RSS 2020): minimize residual norm subject to constraints
- Vasan et al. (2024): constant -1/step outperforms dense shaping

KEY CHANGES FROM v13:
1. COST-BASED REWARD: r_t = -1 - w_cte*CTE² - w_heading*heading² - w_effort*||residual||²
   Every term is ≤ 0. No positive per-step components.
2. TERMINAL: +200 success, 0 failure. Only way to earn positive reward is to finish.
3. GAMMA = 0.999: Long horizon so terminal reward is visible throughout episode.
4. SAC: Off-policy algorithm with entropy regularization (dominant in residual RL lit).
5. ZERO-INIT: Residual policy output layer initialized to zero → starts as pure LQR.
6. CRITIC WARM-UP: 50K steps with frozen residual to learn LQR's value landscape.
7. REDUCED RESIDUAL AUTHORITY: 10-15% of baseline (was 50%).

WHY THIS WORKS:
With r_t ≈ -1.5 per step and γ=0.999:
  V(alive) = -1.5 / (1 - 0.999) = -1500
  A(goal at step 700) = +200 - (-1500) = +1700   → WANTS TO FINISH
  A(fail at step 2000) = 0 - (-1500) = +1500      → but less than finishing early

Longer episodes accumulate MORE negative reward → agent learns to complete quickly.
Failed episodes (longer) are ALWAYS worse than successful ones (shorter).
"""

# =============================================================================
# MASTER CONTROL VARIABLE
# =============================================================================

#set agent_mode = "Compare" to compare LQR and Hybrid
#set agent_mode = "NoiseCompare" to compare LQR vs NoisyLQR (URCA poster)
agent_mode = "PPO"
which_agent = "PPO"

# =============================================================================
# LQR EVALUATION CONFIGURATION
# =============================================================================


LQR_EVALUATION_EPISODES = 5000
LQR_PRINT_EVERY = 1

# =============================================================================
# CURRICULUM LEARNING — ADR (Automatic Domain Randomization)
# =============================================================================
# Replaces fixed curriculum stages with adaptive, performance-gated progression.
# Based on Akkaya et al. (2019) "Solving Rubik's Cube with a Robot Hand".
#
# How it works:
#   - Terrain difficulty starts low (ADR_TERRAIN_MAX_START) and increases
#     ONLY when the agent achieves performance thresholds over a rolling window.
#   - If performance drops, difficulty can decrease (bidirectional).
#   - At each episode, terrain intensity is sampled from [0, current_max],
#     so the agent always revisits easy terrain (prevents catastrophic forgetting).
#
# Quantitative justification:
#   - ADR outperforms fixed stages by 25-50% in sample efficiency (Akkaya 2019)
#   - Performance gates prevent wasting samples on too-easy terrain
#   - Bidirectional progression ensures agent never gets stuck

USE_CURRICULUM = False  # Legacy fixed stages — DEPRECATED, use ADR instead

# --- ADR Configuration ---
ADR_TERRAIN_MAX_START = 10.0     # v32 (2026-04-17): lowered 35 → 10. Diagnostic on run 20260416_223754
                                  #   showed 0% success over 11,301 eps with ADR pinned at the 35 floor
                                  #   (224 regression attempts, 0 advances). Rover never sees terrain it
                                  #   can master, so curriculum never starts. 10% ≈ mostly-flat ground,
                                  #   giving the policy a chance to learn basic "drive to goal" before
                                  #   terrain roughness becomes a factor. Will ramp to 100 via ADR.
                                  # Previous: 35.0 (v27), 60.0 (v26), 15.0 (v25)
ADR_TERRAIN_MAX_LIMIT = 100.0    # Maximum reachable upper bound
ADR_SUCCESS_THRESHOLD = 0.70     # v32: 0.85 → 0.70. With easier starting terrain, earlier advancement
                                  #   keeps the curriculum progressing; final mastery happens at the
                                  #   higher levels. Too strict a threshold stalls curriculum on lower
                                  #   levels while the policy overfits. VertiSelector / RL-Wheeled-
                                  #   Mobility both use ~70% as advancement criterion.
ADR_CTE_THRESHOLD = 0.10          # Must have mean CTE < 10cm to advance (relaxed for hilly terrain)
ADR_REGRESSION_SUCCESS_THRESHOLD = 0.50  # Drop below 50% success -> decrease difficulty
ADR_REGRESSION_CTE_THRESHOLD = 0.50      # CTE > 50cm -> decrease difficulty
ADR_EVAL_WINDOW = 200            # Rolling window — 200 eps gives stable estimate before advancing
ADR_STEP_UP = 3.0                # Smaller steps — 3% per advance (~22 advances to reach 100%)
ADR_STEP_DOWN = 3.0              # Decrease by 3% per regression (symmetric with advance)
ADR_MIN_EPISODES_PER_LEVEL = 50  # Must train 50 eps at each level before evaluating
ADR_COOLDOWN_EPISODES = 30       # 30 eps cooldown — lets VecNormalize adapt after difficulty change

# --- Coupled Episode Rotation ---
# Each (path, terrain_heightfield, friction) configuration is repeated for
# this many episodes before rotating to a new one. This gives SAC's replay
# buffer locally consistent transitions while ensuring diversity.
#
# Quantitative justification:
#   - Cobbe et al. (2020): 5 repeats/config outperforms 100 by 30%
#   - Peng et al. (2018): 3-5 repeats sufficient for motor skill learning
#   - Too few (1): uncorrelated dynamics in replay buffer -> high variance
#   - Too many (50+): overfits to specific terrain features
EPISODES_PER_CONFIGURATION = 4

# =============================================================================
# PARALLEL ENVIRONMENTS
# =============================================================================
# Number of parallel environments for SubprocVecEnv.
# Each environment runs in its own process with an independent PyBullet server.
# More envs = faster sample collection, but uses more CPU/RAM.
# Set to 1 to disable parallelism (single-env, original behavior).
NUM_PARALLEL_ENVS = 12   # 12 physical cores. Hyperthreads don't help CPU-bound physics.
                        # With n_steps=4096: 4096×8 = 32768 steps/update ≈ 16-40 episodes.
                        # Papers typically use 16-64; 8 is a practical compromise for
                        # PyBullet sim cost.

# =============================================================================
# RESIDUAL RL CONFIGURATION — RESEARCH VALIDATED
# =============================================================================
# Research consensus: Residual authority should be 10-15% of baseline commands.
# RPL4F110: steering residual = 0.05/0.42 ≈ 12% of full authority
# I-CTRL: action_scale = 0.1 (10% of full range)
#
# Reduced from v13 (0.3, 0.8) to prevent residual from fighting LQR.
# These values give the residual enough authority for terrain corrections
# without allowing it to override LQR's core tracking behavior.

# v33.9 (2026-05-26): authority reduced 0.35 → 0.15 (velocity) and 0.70 → 0.30
# (omega) to match the literature's 10-30% residual range. The previous 88%/67%
# range let PPO actively fight LQR — Compare run showed Hybrid losing to LQR by
# 3pp aggregate and 6pp at terrain 80-100. Smaller authority forces PPO to make
# targeted small corrections instead of large competing commands. Backed by:
# Johannink ICRA 2019, Silver RSS 2018, Trumpp racing 2023 (all use small range).
MAX_RESIDUAL_VELOCITY = 0.15  # Cap on residual velocity — 38% of v_max=0.4
MAX_RESIDUAL_OMEGA = 0.30     # Cap on residual omega — 29% of ω_max=1.047
USE_ADAPTIVE_GATING = False  # Legacy CTE-based gating — DEPRECATED

# --- Velocity-Error Adaptive Authority ---
# SAC's effective authority scales with how badly LQR is tracking.
# When LQR tracks perfectly (vel_error ≈ 0), SAC can only make tiny corrections.
# When LQR struggles (stuck on hill, slipping), SAC gets full authority.
#
# authority_level = clip(vel_error / threshold, min_scale, max_scale)
# effective_max_residual = MAX_RESIDUAL_* × authority_level
#
# Examples with these defaults (threshold=0.15):
#   vel_error=0.02 (good tracking):  authority=0.13 → eff_vel=0.065, eff_omega=0.20
#   vel_error=0.08 (moderate error): authority=0.53 → eff_vel=0.27,  eff_omega=0.80
#   vel_error=0.15 (LQR struggling): authority=1.00 → eff_vel=0.50,  eff_omega=1.50
USE_VELOCITY_ERROR_AUTHORITY = False   # DISABLED: authority gate removed. Residual bounds are small
                                       # enough (0.10 vel, 0.30 omega) that full [-1,1] action range
                                       # maps to safe corrections. No need to attenuate by LQR error.
AUTHORITY_VEL_ERROR_THRESHOLD = 0.15  # (unused when authority disabled)
AUTHORITY_MIN_SCALE = 1.0             # Fixed at 1.0 — full residual always applied
AUTHORITY_MAX_SCALE = 1.0             # Fixed at 1.0

# Velocity profiling: max speed on straight path sections
PATH_V_MAX = 0.4  # m/s — Leo rover max linear speed ~0.4 m/s

# Hardware velocity/omega hard clips (match real Leo rover physical limits)
MAX_VELOCITY_CLIP = 0.4    # m/s — Leo rover max linear speed
MAX_OMEGA_CLIP = 1.047     # rad/s — Leo rover max angular speed (~60 deg/s)

RESIDUAL_SCALE_VELOCITY = MAX_RESIDUAL_VELOCITY
RESIDUAL_SCALE_OMEGA = MAX_RESIDUAL_OMEGA

# =============================================================================
# COST-BASED REWARD CONFIGURATION (v14)
# =============================================================================
# EVERY per-step term is ≤ 0. The agent accumulates cost by existing.
# Only reaching the goal gives a positive terminal reward.
# This eliminates the alive bias that plagued v9-v13.
#
# Per-step: r_t = -TIME_COST - CTE_COST_WEIGHT * CTE² - HEADING_COST_WEIGHT * heading²
#                 - EFFORT_COST_WEIGHT * ||residual||²
#
# Terminal: +GOAL_REWARD on success, -FAILURE_PENALTY on fail/timeout/OOB
#
# With these values and ~800-step successful episodes:
#   Episode return (success) = 800*(-1.5) + 200 = -1000
#   Episode return (fail@2000) = 2000*(-1.5) + 0 = -3000
#   Success is ALWAYS better than failure.

USE_COST_REWARD = False  # DISABLED: replaced by counterfactual reward (v15)
# Cost reward failed because it measures absolute state quality, not SAC's marginal contribution.
# With 10-15% residual authority, SAC's effect on CTE is dwarfed by LQR's behavior → SNR ≈ 1:100.
# Paired diagnostic (50 episodes): SAC wins 17, LQR wins 17, ties 16 — pure coin flip.
COST_TIME_PENALTY = 0.15
COST_CTE_WEIGHT = 50.0
COST_HEADING_WEIGHT = 5.0
COST_EFFORT_WEIGHT = 0.5
COST_PROGRESS_WEIGHT = 5.0
COST_GOAL_REWARD = 200.0
COST_FAILURE_PENALTY = 0.0

# =============================================================================
# RESIDUAL RL REWARD CONFIGURATION (v17 — cost-tracking with time penalty)
# =============================================================================
# v17 fixes the ALIVE BIAS in v16 where positive per-step rewards meant:
#   - Failed episodes (3400 steps × 1.0/step = 3400) got MORE reward than
#     successful episodes (1200 steps × 1.0/step + 10 bonus = 1210)
#   - SAC literally learned that FAILING IS BETTER than succeeding
#   - All top-5 highest-rewarded episodes were FAILURES
#
# v17 fix: TIME PENALTY makes per-step reward NET NEGATIVE:
#   r = -time_penalty + w_track*exp(-cte²/2σ²) + w_head*exp(-θ²/2σ²)
#       + w_prog*Δprogress - w_effort*(||res||²)  +  terminal_bonus/penalty
#
# With time_penalty=1.3, the quality offsets can never exceed the cost:
#   Perfect tracking (CTE=0): per_step = -1.3 + 1.0 + 0.3 = 0.0 (break even)
#   LQR quality (CTE=0.04):   per_step = -1.3 + 0.80 + 0.26 = -0.24 (net cost)
#   Bad tracking (CTE=0.15):  per_step = -1.3 + 0.04 + 0.04 = -1.22 (heavy cost)
#
# Expected episode returns:
#   Success (800 steps, CTE=0.04): 800×(-0.24) + 750 = +558  ← BEST
#   Success (1200 steps, CTE=0.05): 1200×(-0.35) + 750 = +330
#   Failure (3400 steps, CTE=0.05): 3400×(-0.35) - 200 = -1390  ← WORST
# Success ALWAYS beats failure. Faster ALWAYS beats slower.

USE_RESIDUAL_REWARD = True   # v25: ENABLED — progress-focused reward with terrain-adaptive CTE
# === v25 PROGRESS-FOCUSED REWARD ===
# Diagnostic findings (lqr_diagnostic.py, March 2026):
#   - LQR CTE is already good (0.025-0.06m) on completable terrain
#   - The real failure mode is getting STUCK on slopes >= ~34°
#   - Lateral slip is negligible (std ~0.008 m/s at all friction levels)
#   - PPO's job: maintain forward progress, detect stuck, take evasive action
#
# Design principles:
#   1. Forward progress is the PRIMARY reward (not CTE)
#   2. CTE is a soft Gaussian constraint (wide sigma permits detours)
#   3. Stagnation penalty escalates → PPO learns "stuck → try something different"
#   4. Terrain-adaptive CTE: on steep slopes, widen sigma to permit route deviation
#   5. Time penalty ensures every step costs → faster = better
#
# Previous values preserved in comments for easy revert.
RESIDUAL_SIGMA_CTE = 0.10         # Gaussian width for CTE (WIDENED from 0.15 to permit detours)
                                   # CTE=0.06: 83% credit, CTE=0.10: 61%, CTE=0.15: 33%
                                   # Wide enough to allow hill avoidance without catastrophic penalty
                                   # Previous: 0.15
RESIDUAL_SIGMA_HEADING = 0.15     # Gaussian width for heading (unchanged)
RESIDUAL_W_TRACKING = 1.0         # CTE tracking (REDUCED from 3.0 — CTE is constraint, not objective)
                                   # Previous: 3.0
RESIDUAL_W_HEADING = 0.3          # Heading alignment (unchanged)
RESIDUAL_W_PROGRESS = 500.0       # Path advance (v27: 5.0→500.0)
                                   # v26 had 5.0 but per-step progress signal was 257× weaker than
                                   # tracking → invisible to PPO. At 500, progress ≈ 0.32/step vs
                                   # tracking ≈ 0.83/step (2.6:1). PPO can now detect that steering
                                   # around slopes (CTE loss -0.31) is worthwhile if it resumes
                                   # progress (gain +0.32). Previous: 5.0, 3.0
RESIDUAL_W_EFFORT = 0.05          # Small effort penalty (v27: terrain gate removed, restore light
                                   #   pressure toward zero residual on flat terrain. Terrain-adaptive
                                   #   reduction (0.8) scales this to 0.01 on steep slopes.)
                                   # Previous: 0.0 (v26), 0.05 (v25), 0.3 (v17)
RESIDUAL_TIME_PENALTY = 1.5       # Per-step cost (REDUCED from 3.5)
                                   # MUST exceed max quality offsets (1.0+0.3=1.3) so per-step
                                   # is always ≤ 0. Lower than before because we shifted weight
                                   # to progress, which is variable, not Gaussian-capped.
                                   #   Flat, good tracking: -1.5 + 1.0 + 0.3 + 0.2 = 0.0 (break even)
                                   #   Hill, moving:        -1.5 + 0.7 + 0.2 + 0.1 = -0.5 (modest cost)
                                   #   Hill, stuck:         -1.5 + 0.8 + 0.3 + 0.0 = -0.4 + stag penalty
                                   #   Detouring (CTE=0.15): -1.5 + 0.3 + 0.1 + 0.15 = -0.95
                                   # Previous: 3.5
RESIDUAL_SUCCESS_BONUS = 2000.0    # v28: 750→2000. With gamma=0.99, 750 was nearly invisible
                                    # at step 800 (750*0.99^800≈1.6). At 2000, discounted value
                                    # at step 400 is 2000*0.99^400≈36, at step 200 is ≈270.
                                    # Still small per-step but now within PPO's advantage horizon.
RESIDUAL_FAILURE_PENALTY = 200.0  # Failure/timeout — large enough to separate from success

# === v25 TERRAIN-ADAPTIVE CTE WIDTH ===
# On steep terrain, widen the CTE Gaussian to permit detours around hills.
# sigma_effective = RESIDUAL_SIGMA_CTE * (1 + RESIDUAL_TERRAIN_CTE_EXPANSION * terrain_difficulty)
# terrain_difficulty = min(1.0, slope_magnitude / RESIDUAL_TERRAIN_SLOPE_THRESHOLD)
#   Flat:  sigma_eff = 0.10 * (1 + 0) = 0.10 (tight tracking)
#   30° slope: sigma_eff = 0.10 * (1 + 1.0) = 0.20 (relaxed, permits detours)
RESIDUAL_TERRAIN_SLOPE_THRESHOLD = 0.5  # Slope magnitude where terrain_difficulty saturates (~30°)
RESIDUAL_TERRAIN_CTE_EXPANSION = 1.0    # How much to widen CTE sigma on slopes (1.0 = 2× at max)
RESIDUAL_TERRAIN_EFFORT_REDUCTION = 0.8  # Reduce effort penalty on steep terrain (0.8 = 80% reduction at max slope)
                                          # On flat: full effort penalty keeps residuals near zero (trust LQR)
                                          # On 30° slope: effort penalty × 0.2 → allows large evasive maneuvers
                                          # Same pattern as CF reward's cf_terrain_effort_reduction

# === v26 TERRAIN-GATED RESIDUAL ===
# PPO's residual is GATED by terrain difficulty: zero on flat, full on slopes.
# This solves the gradient dilution problem: 95% of flat-terrain steps produce
# zero gradient (PPO output is masked), so 100% of training signal comes from
# actual terrain challenges where PPO action matters.
#
# Gate function: smooth_ramp(terrain_difficulty, onset, full)
#   terrain_difficulty < onset:  gate = 0.0 (pure LQR, PPO zeroed)
#   onset < td < full:           gate = smooth ramp 0→1
#   terrain_difficulty > full:   gate = 1.0 (full PPO authority)
#
# terrain_difficulty = min(1.0, slope_magnitude / TERRAIN_SLOPE_THRESHOLD)
#   slope_magnitude ≈ 0.0 on flat, 0.34 at 20°, 0.50 at 30°, 0.58 at 34°
#   With threshold=0.5: td=0.0 flat, td=0.68 at 20°, td=1.0 at 30°
#
# Onset=0.3 → PPO starts engaging at ~17° slope (td=0.3)
# Full=0.7  → full PPO authority at ~24° slope (td=0.7)
# Stuck threshold from diagnostic: ~34° — PPO acts WELL BEFORE this
USE_TERRAIN_GATE = False             # v27: disabled — PPO always has authority, learns via reward
TERRAIN_GATE_ONSET = 0.3             # terrain_difficulty below this → gate=0 (pure LQR)
TERRAIN_GATE_FULL = 0.7              # terrain_difficulty above this → gate=1 (full PPO)
                                      # Previous: N/A (new in v26)

# === v25 SUSTAINED RECOVERY THRESHOLD ===
# Recovery bonus requires SUSTAINED forward velocity, not just a single
# oscillation above threshold. Rover must maintain velocity for this many
# consecutive steps to earn recovery credit.
RESIDUAL_RECOVERY_SUSTAIN_STEPS = 10    # Must stay above vel threshold for 10 steps to count

# === STAGNATION DETECTION (stuck recovery) ===
STAGNATION_VEL_THRESHOLD = 0.02   # Forward velocity below this = "stuck" (m/s)
STAGNATION_WINDOW = 5             # Grace period (steps) — shorter to detect stagnation faster
STAGNATION_PENALTY_SCALE = 0.5    # How fast penalty grows per 10 stuck steps
STAGNATION_MAX_PENALTY = 3.0      # Cap on per-step stagnation penalty
STAGNATION_TERMINATION_STEPS = 600  # v32 (2026-04-17): 200 → 600. At 200, diagnostic on run
                                     # 20260416_223754 showed 97.3% of episodes ended between
                                     # 211-500 steps — stagnation was firing before the policy could
                                     # learn crawl-through-rough-patch recovery. CaT (IROS 2024)
                                     # argues that velocity-based terminations fired too early teach
                                     # the policy to avoid learning recovery. 600 steps ≈ 24s sim
                                     # time at 0.04s/step — enough to attempt recovery on harder
                                     # terrain, still bounded so genuinely stuck episodes end.
                                     # Previous: 200 (v25), 500 (original)
# --- Anti-exploit proportional recovery ---
# Fixed recovery bonus was exploitable: stop-start cycling could net positive.
# Proportional recovery: bonus = ratio × accumulated_penalty, capped.
# Net reward from any stagnation episode is ALWAYS negative (50% payback).
RECOVERY_PAYBACK_RATIO = 0.5     # Recover 50% of accumulated stagnation penalty
RECOVERY_PAYBACK_CAP = 2.0       # Cap on recovery bonus (prevents late-episode windfalls)

# =============================================================================
# COUNTERFACTUAL REWARD CONFIGURATION (DISABLED — kept for backward compat)
# =============================================================================

USE_COUNTERFACTUAL_REWARD = False  # v25: DISABLED — replaced by v25 progress-focused residual reward
                                   # CF reward can't learn evasive maneuvers: single-step detours
                                   # always look worse than LQR, so PPO never learns to go around hills.
                                   # Previous: True
# v19 counterfactual: reward = how much SAC improves over pure LQR on the same step.
# Environmental factors (terrain, friction, path) cancel out because both
# hybrid and pure-LQR face the exact same conditions.
# Only needed during training — deployed model just runs forward inference.

# v19 counterfactual weights — simpler than v15, focused on SAC's marginal contribution
CF_W_CTE_IMPROVEMENT = 200.0      # (cf_cte - hybrid_cte) — positive when SAC reduces CTE. 4x amplified for sharper signal.
CF_W_HEADING_IMPROVEMENT = 15.0   # (|cf_heading| - |hybrid_heading|) — reduced from 80 (v23): was exploitable
                                   # on flat terrain where tiny heading diffs dominated total reward.
CF_HEADING_DEAD_ZONE = 0.005      # rad — ignore heading improvements smaller than 0.005 rad (0.3°),
                                   # which are noise rather than real improvements.
CF_W_VELOCITY_TRACKING = 40.0     # (|cf_vel_err| - |hybrid_vel_err|) — positive when SAC improves velocity tracking. 4x amplified.
CF_VELOCITY_DEAD_ZONE = 0.03      # Only fires when LQR has >0.03 m/s velocity error
CF_W_PROGRESS_IMPROVEMENT = 0.0   # Disabled — progress is implicit in not failing
CF_W_ABSOLUTE_CTE = 0.0           # No absolute CTE penalty (counterfactual handles it)
CF_MAX_ABSOLUTE_CTE_PENALTY = 0.0 # No cap needed
CF_W_EFFORT = 0.05                # Reduced effort penalty (v22) — 0.1 was 27% of productive CF signal,
                                   # suppressing necessary corrections on steep terrain
CF_SUCCESS_BONUS = 200.0           # Goal reached (reduced from 750 — per-step CF is primary signal)
CF_FAILURE_PENALTY = 50.0          # Failure penalty (reduced — per-step CF is primary signal)

# ── v21 Terrain-Adaptive Counterfactual Reward ──
# On steep terrain (hills), the rover physically cannot follow the direct path.
# Instead of penalizing CTE deviation, we:
#   1. Reduce CTE/heading penalties proportionally to terrain difficulty
#   2. Boost forward-progress reward so the agent learns to go AROUND hills
#   3. The agent uses gravity_x, gravity_y in its observation to sense slope
#
# On flat terrain: CTE dominates → stay on path
# On steep terrain: progress dominates → find a traversable route, then return to path
#
# terrain_difficulty = min(1.0, slope_magnitude / CF_TERRAIN_SLOPE_THRESHOLD)
#   where slope_magnitude = sqrt(gx² + gy²) from body-frame gravity vector
#   Slope examples: 10°→0.17, 20°→0.34, 30°→0.50, 45°→0.71
#
# cte_weight_scale = 1.0 - CF_TERRAIN_CTE_REDUCTION × terrain_difficulty
# progress_weight_scale = 1.0 + CF_TERRAIN_PROGRESS_BOOST × terrain_difficulty
CF_TERRAIN_SLOPE_THRESHOLD = 0.5  # Gravity tilt magnitude where terrain_difficulty saturates (~30°)
CF_TERRAIN_CTE_REDUCTION = 0.7    # CTE/heading weight drops to 30% of normal on max-difficulty terrain
CF_TERRAIN_PROGRESS_BOOST = 3.0   # Progress weight becomes 4× its base value on max-difficulty terrain
CF_TERRAIN_EFFORT_REDUCTION = 0.8  # Effort penalty drops to 20% of normal on max-difficulty terrain (v22)
                                   # Allows larger corrections where they're most needed
CF_W_FORWARD_PROGRESS = 50.0      # Base weight for direct forward-progress reward
                                   # Per-step progress ~0.8% at 0.4 m/s on 10m path → ~0.4 reward/step
                                   # On slopes this is boosted to ~1.6/step (dominant over reduced CTE)

# =============================================================================
# EXPECTED REWARD BEHAVIOR (v14 cost-based)
# =============================================================================
"""
Per-step reward breakdown (all ≤ 0):

FLAT terrain, good tracking (CTE~0.03, heading~0.02):
  r_time = -1.0
  r_cte = -5.0 * 0.03² = -0.0045
  r_heading = -1.0 * 0.02² = -0.0004
  r_effort = -2.0 * 0.001 = -0.002  (near-zero residual)
  TOTAL: ≈ -1.007/step

MODERATE hill, correcting (CTE~0.08, heading~0.05):
  r_time = -1.0
  r_cte = -5.0 * 0.08² = -0.032
  r_heading = -1.0 * 0.05² = -0.0025
  r_effort = -2.0 * 0.01 = -0.02
  TOTAL: ≈ -1.055/step

HARD hill, large corrections (CTE~0.2, heading~0.15):
  r_time = -1.0
  r_cte = -5.0 * 0.2² = -0.2
  r_heading = -1.0 * 0.15² = -0.0225
  r_effort = -2.0 * 0.05 = -0.1
  TOTAL: ≈ -1.323/step

Episode totals:
  Flat, quick success (700 steps):   700*(-1.007) + 200 = -505
  Hill, success (900 steps):         900*(-1.055) + 200 = -750
  Failed/timeout (2000 steps):       2000*(-1.3) + 0    = -2600

KEY: Successes are ALWAYS less negative than failures.
     Faster completion → less negative → better.
     No alive bias. No episode extension incentive.
"""

# =============================================================================
# LEGACY REWARD CONFIGURATION (v7/v8 — used for non-Hybrid modes)
# =============================================================================


USE_PAPER_REWARD = True

# --- v7 Multiplicative Components (kept) ---
W_CTE_EXP = 2.0  # Exponential CTE decay rate
CTE_MAX = 2.0  # CTE normalization maximum
CTE_TERMINATION = 2.0  # Terminate if CTE exceeds this (reduced from 3.0)
W_HEADING_EXP = 1.5  # Exponential heading decay rate
V_MAX = 0.6  # Maximum velocity for normalization
USE_VELOCITY_PROGRESS = True  # Use velocity-based progress
W_SMOOTHNESS = 0.1  # Action smoothness penalty weight
ALIVE_BONUS = 0.01  # Per-step survival bonus

# --- v8 Components ---
W_RESIDUAL_PENALTY = 0.02  # Quadratic residual penalty
TIME_PENALTY = -0.1  # Per-step time cost

# --- Terminal Rewards ---
GOAL_REWARD = 50.0
OFF_TRACK_PENALTY = -20.0

# --- Episode Limits ---
MAX_EPISODE_STEPS = 2000

# =============================================================================
# LEGACY REWARD PARAMETERS (kept for backwards compatibility)
# =============================================================================


ALPHA1_CTE = 10.0
ALPHA2_HEADING = 2.0
ALPHA3_EFFORT = 0.0
ALPHA4_SLIP = 0.0
ALPHA5_ENERGY = 0.0
ALPHA6_PROGRESS = 1.0

W_CTE = 10.0
W_HEADING = 2.0
W_VELOCITY = 0.2
W_PROGRESS = 1.0
SUCCESS_BONUS = 0.0
TIMEOUT_PENALTY = 0.0
WAYPOINT_BONUS = 0.0
GOAL_BONUS = 0.0

RESIDUAL_PENALTY_WEIGHT = 0.0
CTE_LINEAR_WEIGHT = -2.5
CTE_SQUARED_WEIGHT = -2.0
HEADING_WEIGHT = 0.8

TRUST_BONUS_WEIGHT_HIGH = 0.0
TRUST_BONUS_WEIGHT_MEDIUM = 0.0
TRUST_BONUS_DECAY_HIGH = 5.0
TRUST_BONUS_DECAY_MEDIUM = 3.0
TRUST_BONUS_CTE_THRESHOLD_HIGH = 0.05
TRUST_BONUS_CTE_THRESHOLD_MEDIUM = 0.10

FIGHTING_PENALTY_WEIGHT = 0.0
FIGHTING_PENALTY_RESIDUAL_THRESHOLD = 0.3

USE_CTE_IMPROVEMENT_BONUS = False
CTE_IMPROVEMENT_BONUS_POSITIVE = 0.0
CTE_IMPROVEMENT_BONUS_NEGATIVE = 0.0
CTE_TREND_DETECTION_THRESHOLD = 0.01

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================


TRAINING_EPISODES = 20000
RESUME_TRAINING = False   # Fresh start — reward normalization changed (norm_reward=False)

# Set to a specific checkpoint path to resume from, or None to auto-detect latest
PPO_RESUME_CHECKPOINT = None  # r"logs\ppo_training_20260402_234756\checkpoints\ppo_pure_4320000_steps.zip"

TRAINING_TERRAIN_MIN = 10.0  # v27: lowered from 30. Without terrain gate, PPO always has
                              #   authority, so flat terrain is fine (effort penalty teaches
                              #   PPO to stay quiet). Mix of easy+hard gives PPO clear
                              #   contrast between what works on flat vs slopes.
                              # Previous: 30.0 (v26), 0.0 (v25)
TRAINING_TERRAIN_MAX = 50.0  # Initial max (ADR may override this for SAC training)
TRAINING_FRICTION_MIN = 50.0  # Was 30.0 — narrowed to reduce variance
TRAINING_FRICTION_MAX = 90.0  # Was 100.0 — narrowed to reduce variance

TRAINING_USE_RANDOM_PATHS = True
TRAINING_MIN_CURVATURE_ANGLE = 25.0
TRAINING_MAX_CURVATURE_ANGLE = 120.0
TRAINING_TOTAL_PATH_DISTANCE = 10.0
TRAINING_EPISODES_PER_PATH = 4  # Match EPISODES_PER_CONFIGURATION for coupled rotation
TRAINING_NUM_RANDOM_PATHS = 500

# =============================================================================
# SCREENING CONFIGURATION
# =============================================================================


SCREENING_EPISODES_PER_COMBINATION = 100

FACTORS = {
    'max_slope': {
        'low': 0.0,
        'high': 70.0,
    },
    'avg_slope': {
        'low': 0.0,
        'high': 70.0,
    },
    'friction': {
        'low': 30.0,
        'high': 100.0,
    },
    'sharp_turn': {
        'low': 'gentle',
        'high': 'sharp',
    },
}

GENTLE_PATH_CURVATURE_MIN = 15.0
GENTLE_PATH_CURVATURE_MAX = 45.0
SHARP_PATH_CURVATURE_MIN = 60.0
SHARP_PATH_CURVATURE_MAX = 120.0

SCREENING_TOTAL_PATH_DISTANCE = 10.0
SCREENING_EPISODES_PER_PATH = 1
SCREENING_NUM_PATHS = 100

# =============================================================================
# MODEL FILE PATHS
# =============================================================================


MODELS_DIR = "models"
PPO_MODEL_NAME = "ppo_trained_model.zip"
HYBRID_MODEL_NAME = "hybrid_trained_model.zip"
SAC_HYBRID_MODEL_NAME = "sac_hybrid_trained_model.zip"
SCREENING_PPO_CHECKPOINT = "screening_ppo_checkpoint.zip"
SCREENING_HYBRID_CHECKPOINT = "screening_hybrid_checkpoint.zip"

# Override model path for comparisons (set via GUI; empty = auto-detect)
COMPARISON_MODEL_PATH = ""

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================


BASE_LOG_DIR = "logs"
SCREENING_LOG_DIR = "screening_results"

# =============================================================================
# SAC HYPERPARAMETERS (for Hybrid training)
# =============================================================================
# SAC is the dominant algorithm in residual RL literature:
# - Johannink et al. uses TD3 (off-policy, similar family)
# - Schaff & Walter uses SAC explicitly
# - Baek et al. (Hybrid LMC) reports SAC+LQR succeeds where DDPG+LQR fails
#
# Key advantages over PPO for residual RL:
# - Off-policy: much more sample-efficient (reuses past experience)
# - Entropy regularization: natural exploration compatible with LQR baseline
# - Continuous action space native (no need for clip_range)
# - Stable with single environment (PPO needs parallel envs)


LEARNING_RATE = 3e-4
GAMMA = 0.995  # Longer horizon so terminal rewards are visible to PPO (v22)
# Effective horizon = 1/(1-0.995) = 200 steps. With early stagnation termination
# shortening failed episodes, gamma=0.995 makes the success_bonus (200) and
# failure_penalty (50) contribute meaningfully to the value function.
# At 1400 steps: 0.995^1400 ≈ 0.0009 → success bonus worth ~0.18 at t=0.
BUFFER_SIZE = 500_000  # Larger replay buffer for better sample diversity
BATCH_SIZE = 512  # Larger batch for more stable gradients
LEARNING_STARTS = 1000  # Random exploration before learning
TAU = 0.005  # Soft target update coefficient
ENT_COEF = 'auto'  # Auto-tune entropy (SAC's key feature)
POLICY_KWARGS = dict(
    net_arch=[256, 256],  # Standard SAC architecture
)

# Critic warm-up: train Q-functions for this many steps with zero residual
# before enabling policy updates. This lets the critic learn LQR's value
# landscape before the policy starts exploring.
# Reduced from 50K: with counterfactual reward, the critic learns faster
# because per-step signal is zero-centered (no landscape to pre-learn).
WARMUP_STEPS = 0  # No warmup needed for PPO (on-policy, no replay buffer to pre-fill)

# SAC target entropy: controls exploration level.
# Default SB3: -dim(A) = -2.0. Too high for residual RL (SAC explores too aggressively).
# -1.0 encourages moderate exploration: std≈0.37 per dim → residuals stay moderate.
TARGET_ENTROPY = -1.0

SAC_CONFIG = {
    'learning_rate': LEARNING_RATE,
    'gamma': GAMMA,
    'buffer_size': BUFFER_SIZE,
    'batch_size': BATCH_SIZE,
    'learning_starts': LEARNING_STARTS,
    'tau': TAU,
    'ent_coef': ENT_COEF,
    'target_entropy': TARGET_ENTROPY,
    'policy_kwargs': POLICY_KWARGS,
}

# =============================================================================
# PPO HYPERPARAMETERS — PRIMARY ALGORITHM FOR HYBRID TRAINING (v20)
# =============================================================================
# PPO computes advantages A(s,a) = Q(s,a) - V(s) via GAE, automatically
# stripping the constant baseline from the counterfactual reward. This
# eliminates the Q-value SNR problem that caused SAC to diverge.
#
# Key design choices:
#   - n_steps=2048: ~2 episodes per update with 4 parallel envs (8192 total)
#   - gamma=0.95: short horizon for dense per-step CF signal
#   - gae_lambda=0.95: standard GAE bias-variance tradeoff
#   - net_arch=[256,256]: same capacity as SAC (was [64,64] for pure PPO)
#   - ent_coef=0.01: moderate exploration for residual corrections
#   - n_epochs=10: standard PPO update intensity
#   - batch_size=256: stable gradient estimates

# v33.6 (2026-05-15): 3e-4 → 1.5e-4. The lambda LR schedule wraps base_lr
# in `lambda progress_remaining: base_lr * progress_remaining`, but
# estimated_steps = 500000 * TRAINING_EPISODES * 10 ≈ 2.5e13 (with
# TRAINING_EPISODES=5e6), so progress_remaining ≈ 1.0 essentially forever
# and the LR never decays in practice. Halving the base value gives gentler
# updates from the start, pairing with the lower ent_coef to let the policy
# converge instead of perpetually bouncing around at full step size. If a
# future run uses a smaller TRAINING_EPISODES (e.g., 50k) the schedule
# might actually anneal — at that point reconsider this value.
PPO_LEARNING_RATE = 1.5e-4
# v33.2 (2026-05-02): LR pinned for resume training. The PerformanceRollback
# callback halves LR mid-training but stores the change only in the optimizer
# (not in model.learning_rate), so PPO.load() loses it. Without this constant,
# resume rebuilds the LR schedule from the original 3e-4 lambda which causes
# a 4x LR jump and policy regression. 7.5e-5 = LR after two rollbacks
# (3e-4 × 0.5 × 0.5), matching the original 50k-episode run's end state.
# Tune this if you start a new training run that ends at a different LR.
PPO_RESUME_LR = 7.5e-5
PPO_N_STEPS = 4096            # v28: 2048→4096. Steps per env before update (4096 × 8 envs = 32768 total)
                               # More steps = more trajectory diversity per update, stabler advantages.
PPO_CLIP_RANGE = 0.2
# v31 (2026-04-16): ent_coef 0.05 → 0.0 (SB3 default, matches RL-Zoo3 PyBullet configs).
# At 0.05 the entropy bonus caused catastrophic log_std runaway: std went from 1.0
# to 6.8e17 over 1189 iterations, collapsing ep_rew_mean from -488 (peak) to -13000.
# v32 (2026-04-17): 0.0 → 0.005. Safe to reintroduce a small entropy bonus now that
# SafeMlpPolicy clamps log_std ∈ [-4.0, 0.5] (max std ≈ 1.65). Run 20260416_223754
# showed std collapsing to ≈0.10 with ent_coef=0 — policy was freezing into a
# greedy "end episode ASAP" local optimum. 0.005 is half legged_gym's default (0.01);
# conservative because our clamp range is tight. Watch train/std; if it drifts > 1.0
# the SafeMlpPolicy clamp is either misconfigured or being bypassed.
# v33.6 (2026-05-15): 0.005 → 0.001. Hybrid run 20260513_091206 showed std
# drifting upward from 0.14 → 1.02 over 766 PPO rollouts (~25k episodes),
# approaching the SafeMlpPolicy clamp at 1.65 with no sign of plateau. The
# upward pressure was pure entropy-bonus drift (no counter-pressure since
# action bounds make large std harmless to immediate reward). 0.001 keeps
# a small exploration bonus without overpowering the policy gradient's
# natural pressure to narrow std once the policy has converged.
PPO_ENT_COEF = 0.001
# v31: Initial log_std for the diagonal Gaussian policy. At -1.0, initial action std ≈ 0.37
# — moderate exploration that's appropriate for bounded [-1, 1] actions. Default (0.0)
# gave std = 1.0 which saturates the action bounds too aggressively.
PPO_LOG_STD_INIT = -1.0
# v31: KL early-stopping threshold. PPO will stop the update epoch if approx_kl exceeds
# this value — prevents a single destructive rollout from wrecking the policy. 0.02 is
# a standard "trust region" ceiling (Schulman's PPO paper discusses KL ≈ 0.01-0.05).
PPO_TARGET_KL = 0.02
# v31: Hard safety clamp on log_std inside the policy's forward pass. Even if future
# regularization changes try to push log_std beyond this range, this clamp keeps the
# policy from diverging. [-4.0, 0.5] → std in [0.018, 1.65], well-matched to bounded actions.
PPO_LOG_STD_MIN = -4.0
PPO_LOG_STD_MAX = 0.5
PPO_POLICY_KWARGS = dict(net_arch=[256, 256], log_std_init=PPO_LOG_STD_INIT)  # Same capacity as SAC network
PPO_GAMMA = 0.99               # v28: 0.995→0.99. Shorter horizon so terminal rewards are more visible.
                               # At 0.995, success bonus 2000 at step 800: 2000*0.995^800≈35.
                               # At 0.99: 2000*0.99^800≈6.6 (still small but advantage horizon
                               # 1/(1-0.99*0.95) = ~190 steps is a better match for episode length).
PPO_GAE_LAMBDA = 0.95
PPO_BATCH_SIZE = 256            # Larger batches for stable gradients
PPO_N_EPOCHS = 5               # v28: 10→5. Reduces overfitting to current batch.
                               # With 32768 steps / 256 batch = 128 minibatches × 5 = 640 updates.
PPO_VF_COEF = 0.5
PPO_MAX_GRAD_NORM = 0.5

PPO_CONFIG = {
    'learning_rate': PPO_LEARNING_RATE,
    'n_steps': PPO_N_STEPS,
    'batch_size': PPO_BATCH_SIZE,
    'n_epochs': PPO_N_EPOCHS,
    'gamma': PPO_GAMMA,
    'gae_lambda': PPO_GAE_LAMBDA,
    'clip_range': PPO_CLIP_RANGE,
    'ent_coef': PPO_ENT_COEF,
    'vf_coef': PPO_VF_COEF,
    'max_grad_norm': PPO_MAX_GRAD_NORM,
    'target_kl': PPO_TARGET_KL,
    'policy_kwargs': PPO_POLICY_KWARGS,
}

HYBRID_CONFIG = PPO_CONFIG.copy()

# =============================================================================
# PURE PPO ANTI-PLASTICITY-LOSS CONFIGURATION (v30)
# =============================================================================
# Addresses PPO plasticity loss (Juliani & Ash, NeurIPS 2024):
# 1. LayerNorm after each hidden layer — stabilizes activations, prevents
#    weight drift that causes gradual loss of learning capability.
# 2. Linear LR annealing — reduces update magnitude late in training,
#    preventing catastrophic single-update collapses.
# 3. AdamW with weight decay — L2 regularization keeps weights near
#    initialization, maintaining gradient flow throughout training.
#
# These settings apply ONLY to pure PPO (train_ppo), NOT hybrid PPO.

import torch
PURE_PPO_POLICY_KWARGS = dict(
    net_arch=[256, 256],
    activation_fn=torch.nn.ReLU,
    optimizer_class=torch.optim.AdamW,
    optimizer_kwargs=dict(weight_decay=1e-3),
    # v31: initialize log_std at -1.0 (std ≈ 0.37). Default was 0.0 (std = 1.0) which
    # saturates the [-1, 1] action bounds and is a major contributor to log_std runaway.
    log_std_init=PPO_LOG_STD_INIT,
)

# Linear LR schedule: 3e-4 → 0 over training
PURE_PPO_LR_SCHEDULE = True    # When True, use linear annealing instead of constant LR

PURE_PPO_CONFIG = {
    'learning_rate': PPO_LEARNING_RATE,  # will be wrapped in schedule if PURE_PPO_LR_SCHEDULE
    'n_steps': PPO_N_STEPS,
    'batch_size': PPO_BATCH_SIZE,
    'n_epochs': PPO_N_EPOCHS,
    'gamma': PPO_GAMMA,
    'gae_lambda': PPO_GAE_LAMBDA,
    'clip_range': PPO_CLIP_RANGE,
    'ent_coef': PPO_ENT_COEF,
    'vf_coef': PPO_VF_COEF,
    'max_grad_norm': PPO_MAX_GRAD_NORM,
    'target_kl': PPO_TARGET_KL,
    'policy_kwargs': PURE_PPO_POLICY_KWARGS,
}

# =============================================================================
# ALGORITHM SELECTION FOR HYBRID TRAINING
# =============================================================================
# v20: Switch from SAC to PPO for Hybrid training.
# SAC's off-policy Q-learning cannot learn from weak, dense counterfactual
# reward — the per-step action-dependent signal (~0.02) is <1% of Q-values
# (~12.0 at gamma=0.99), causing actor loss to diverge.
# PPO's on-policy advantage estimation via GAE automatically strips the
# constant baseline, giving clean action-dependent gradients.
USE_PPO_FOR_HYBRID = True   # True = PPO (v20), False = SAC (v19, deprecated)

# =============================================================================
# ENVIRONMENT SETTINGS
# =============================================================================


DISPLAY = False
DEBUG = False
CHECKPOINT_FREQ = 20_000

# Comprehensive step-level logging — logs EVERY step of EVERY episode
# across all parallel envs. Produces ~2GB per env (~8GB total for 4 envs).
# Files: logs/<run>/detailed_steps/env_{id}_steps.csv + env_{id}_episodes.jsonl
DETAILED_STEP_LOGGING = True

# =============================================================================
# NOISY-LQR EXPERIMENT CONFIGURATION (URCA Poster)
# =============================================================================
# Noise type: "ou" (Ornstein-Uhlenbeck) or "gaussian" (i.i.d. per step)
NOISE_TYPE = "ou"           # "ou" or "gaussian"

# Ornstein-Uhlenbeck noise for temporally-correlated perturbation of LQR commands.
# OU process: dx = theta * (mu - x) * dt + sigma * dW
# Produces smooth, mean-reverting perturbations that gently push the rover
# in a sustained direction — much better than uncorrelated Gaussian jitter.
# Parameters matched to the successful February 2026 experiment.
NOISE_V_STD = 0.03         # m/s — OU sigma for linear velocity noise
NOISE_V_MAX = 0.12         # m/s — hard clip for linear velocity noise
NOISE_OMEGA_STD = 0.10     # rad/s — OU sigma for angular velocity noise
NOISE_OMEGA_MAX = 0.30     # rad/s — hard clip for angular velocity noise
NOISE_OU_THETA = 0.15      # OU mean-reversion rate (higher = faster revert to zero)

# Adaptive noise scaling — noise magnitude scales with IMU-sensed terrain tilt
# tilt = sqrt(gravity_x² + gravity_y²)  (0 on flat, ~1 on 90° slope)
# noise_scale = clamp(NOISE_ADAPT_MIN + (1 - NOISE_ADAPT_MIN) * tilt / NOISE_ADAPT_TILT_THRESHOLD, NOISE_ADAPT_MIN, 1.0)
NOISE_ADAPTIVE = True           # True = terrain-adaptive noise, False = constant noise
NOISE_ADAPT_MIN = 0.05         # Minimum noise scale on flat ground (5% of max — nearly silent)
NOISE_ADAPT_TILT_THRESHOLD = 0.35  # Tilt at which noise reaches full strength (~20° slope)

# Experiment sizing
NOISE_COMPARE_EPISODES_PER_AGENT = 500
NOISE_COMPARE_EPISODES_PER_CONFIG = 1
NOISE_COMPARE_LEGACY_LQR = True  # True = disable trajectory profiling + stagnation (reproduces pre-Feb 2026 LQR)
NUM_COMPARISON_PARALLEL = 1     # Number of parallel simulation processes for comparison modes (1 = sequential)

# Terrain/friction range for the noise comparison
# Rough terrain only: 60-100% intensity
NOISE_COMPARE_TERRAIN_MIN = 0.0
NOISE_COMPARE_TERRAIN_MAX = 100.0
NOISE_COMPARE_FRICTION_MIN = 30.0
NOISE_COMPARE_FRICTION_MAX = 100.0


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_ppo_config(mode: str = "PPO") -> dict:
    if mode == "Hybrid":
        return HYBRID_CONFIG.copy()
    elif mode == "PPO":
        return PPO_CONFIG.copy()
    else:
        return HYBRID_CONFIG.copy()


def get_sac_config() -> dict:
    """Get SAC configuration for Hybrid training."""
    return SAC_CONFIG.copy()


def get_residual_config():
    return {
        'max_residual_velocity': MAX_RESIDUAL_VELOCITY,
        'max_residual_omega': MAX_RESIDUAL_OMEGA,
        'use_adaptive_gating': USE_ADAPTIVE_GATING,
        'v_max': PATH_V_MAX,
        # Velocity-error adaptive authority
        'use_velocity_error_authority': USE_VELOCITY_ERROR_AUTHORITY,
        'authority_vel_error_threshold': AUTHORITY_VEL_ERROR_THRESHOLD,
        'authority_min_scale': AUTHORITY_MIN_SCALE,
        'authority_max_scale': AUTHORITY_MAX_SCALE,
    }


def get_residual_scale():
    return (MAX_RESIDUAL_VELOCITY, MAX_RESIDUAL_OMEGA)


def get_cost_reward_config():
    """Get cost-based reward configuration (v14)."""
    return {
        'use_cost_reward': USE_COST_REWARD,
        'cost_time_penalty': COST_TIME_PENALTY,
        'cost_cte_weight': COST_CTE_WEIGHT,
        'cost_heading_weight': COST_HEADING_WEIGHT,
        'cost_effort_weight': COST_EFFORT_WEIGHT,
        'cost_progress_weight': COST_PROGRESS_WEIGHT,
        'cost_goal_reward': COST_GOAL_REWARD,
        'cost_failure_penalty': COST_FAILURE_PENALTY,
    }


def get_counterfactual_config():
    """Get counterfactual reward configuration (v21 terrain-adaptive)."""
    return {
        'use_counterfactual_reward': USE_COUNTERFACTUAL_REWARD,
        'cf_w_cte_improvement': CF_W_CTE_IMPROVEMENT,
        'cf_w_heading_improvement': CF_W_HEADING_IMPROVEMENT,
        'cf_w_progress_improvement': CF_W_PROGRESS_IMPROVEMENT,
        'cf_w_velocity_tracking': CF_W_VELOCITY_TRACKING,
        'cf_velocity_dead_zone': CF_VELOCITY_DEAD_ZONE,
        'cf_heading_dead_zone': CF_HEADING_DEAD_ZONE,
        'cf_w_absolute_cte': CF_W_ABSOLUTE_CTE,
        'cf_max_absolute_cte_penalty': CF_MAX_ABSOLUTE_CTE_PENALTY,
        'cf_w_effort': CF_W_EFFORT,
        'cf_success_bonus': CF_SUCCESS_BONUS,
        'cf_failure_penalty': CF_FAILURE_PENALTY,
        # v21 terrain-adaptive parameters
        'cf_terrain_slope_threshold': CF_TERRAIN_SLOPE_THRESHOLD,
        'cf_terrain_cte_reduction': CF_TERRAIN_CTE_REDUCTION,
        'cf_terrain_progress_boost': CF_TERRAIN_PROGRESS_BOOST,
        'cf_terrain_effort_reduction': CF_TERRAIN_EFFORT_REDUCTION,
        'cf_w_forward_progress': CF_W_FORWARD_PROGRESS,
    }


def get_residual_reward_config():
    """Get residual RL reward configuration (v25 — progress-focused with terrain-adaptive CTE)."""
    return {
        'use_residual_reward': USE_RESIDUAL_REWARD,
        'residual_sigma_cte': RESIDUAL_SIGMA_CTE,
        'residual_sigma_heading': RESIDUAL_SIGMA_HEADING,
        'residual_w_tracking': RESIDUAL_W_TRACKING,
        'residual_w_heading': RESIDUAL_W_HEADING,
        'residual_w_progress': RESIDUAL_W_PROGRESS,
        'residual_w_effort': RESIDUAL_W_EFFORT,
        'residual_time_penalty': RESIDUAL_TIME_PENALTY,
        'residual_success_bonus': RESIDUAL_SUCCESS_BONUS,
        'residual_failure_penalty': RESIDUAL_FAILURE_PENALTY,
        # v25 terrain-adaptive CTE
        'residual_terrain_slope_threshold': RESIDUAL_TERRAIN_SLOPE_THRESHOLD,
        'residual_terrain_cte_expansion': RESIDUAL_TERRAIN_CTE_EXPANSION,
        'residual_terrain_effort_reduction': RESIDUAL_TERRAIN_EFFORT_REDUCTION,
        # v26 terrain gate
        'use_terrain_gate': USE_TERRAIN_GATE,
        'terrain_gate_onset': TERRAIN_GATE_ONSET,
        'terrain_gate_full': TERRAIN_GATE_FULL,
        # v25 sustained recovery
        'residual_recovery_sustain_steps': RESIDUAL_RECOVERY_SUSTAIN_STEPS,
        # Stagnation detection
        'stagnation_velocity_threshold': STAGNATION_VEL_THRESHOLD,
        'stagnation_window': STAGNATION_WINDOW,
        'stagnation_penalty_scale': STAGNATION_PENALTY_SCALE,
        'stagnation_max_penalty': STAGNATION_MAX_PENALTY,
        'stagnation_termination_steps': STAGNATION_TERMINATION_STEPS,
        'recovery_payback_ratio': RECOVERY_PAYBACK_RATIO,
        'recovery_payback_cap': RECOVERY_PAYBACK_CAP,
    }


def get_reward_config():
    """Get reward configuration for v14."""
    config = {
        'use_paper_reward': USE_PAPER_REWARD,
        # v7 multiplicative components
        'w_cte_exp': W_CTE_EXP,
        'cte_max': CTE_MAX,
        'cte_termination': CTE_TERMINATION,
        'w_heading_exp': W_HEADING_EXP,
        'v_max': V_MAX,
        'use_velocity_progress': USE_VELOCITY_PROGRESS,
        'w_smoothness': W_SMOOTHNESS,
        'alive_bonus': ALIVE_BONUS,
        'goal_reward': GOAL_REWARD,
        'off_track_penalty': OFF_TRACK_PENALTY,
        'max_episode_steps': MAX_EPISODE_STEPS,
        # v8 components
        'w_residual_penalty': W_RESIDUAL_PENALTY,
        'time_penalty': TIME_PENALTY,
        # Legacy
        'alpha1_cte': ALPHA1_CTE,
        'alpha2_heading': ALPHA2_HEADING,
        'alpha3_effort': ALPHA3_EFFORT,
        'alpha4_slip': ALPHA4_SLIP,
        'alpha5_energy': ALPHA5_ENERGY,
        'alpha6_progress': ALPHA6_PROGRESS,
        'w_cte': W_CTE,
        'w_heading': W_HEADING,
        'w_velocity': W_VELOCITY,
        'waypoint_bonus': WAYPOINT_BONUS,
        'goal_bonus': GOAL_BONUS,
    }
    # Add counterfactual config (backward compat)
    config.update(get_counterfactual_config())
    # Add cost-based config
    config.update(get_cost_reward_config())
    return config


def get_reward_weights():
    """Get all reward weights (legacy compatibility + v14 cost-based)."""
    return {
        'use_paper_reward': USE_PAPER_REWARD,
        # v14 cost-based
        'use_cost_reward': USE_COST_REWARD,
        'cost_time_penalty': COST_TIME_PENALTY,
        'cost_cte_weight': COST_CTE_WEIGHT,
        'cost_heading_weight': COST_HEADING_WEIGHT,
        'cost_effort_weight': COST_EFFORT_WEIGHT,
        'cost_progress_weight': COST_PROGRESS_WEIGHT,
        'cost_goal_reward': COST_GOAL_REWARD,
        'cost_failure_penalty': COST_FAILURE_PENALTY,
        # v9 counterfactual (disabled)
        'use_counterfactual_reward': USE_COUNTERFACTUAL_REWARD,
        'cf_w_cte_improvement': CF_W_CTE_IMPROVEMENT,
        'cf_w_heading_improvement': CF_W_HEADING_IMPROVEMENT,
        'cf_w_progress_improvement': CF_W_PROGRESS_IMPROVEMENT,
        'cf_w_velocity_tracking': CF_W_VELOCITY_TRACKING,
        'cf_velocity_dead_zone': CF_VELOCITY_DEAD_ZONE,
        'cf_heading_dead_zone': CF_HEADING_DEAD_ZONE,
        'cf_w_absolute_cte': CF_W_ABSOLUTE_CTE,
        'cf_w_effort': CF_W_EFFORT,
        'cf_success_bonus': CF_SUCCESS_BONUS,
        'cf_failure_penalty': CF_FAILURE_PENALTY,
        # v7/v8 parameters
        'w_cte_exp': W_CTE_EXP,
        'cte_max': CTE_MAX,
        'cte_termination': CTE_TERMINATION,
        'w_heading_exp': W_HEADING_EXP,
        'v_max': V_MAX,
        'use_velocity_progress': USE_VELOCITY_PROGRESS,
        'w_smoothness': W_SMOOTHNESS,
        'alive_bonus': ALIVE_BONUS,
        'goal_reward': GOAL_REWARD,
        'off_track_penalty': OFF_TRACK_PENALTY,
        'w_residual_penalty': W_RESIDUAL_PENALTY,
        'time_penalty': TIME_PENALTY,
        # Legacy parameters
        'alpha1_cte': ALPHA1_CTE,
        'alpha2_heading': ALPHA2_HEADING,
        'alpha3_effort': ALPHA3_EFFORT,
        'alpha4_slip': ALPHA4_SLIP,
        'alpha5_energy': ALPHA5_ENERGY,
        'alpha6_progress': ALPHA6_PROGRESS,
        'w_cte': W_CTE,
        'w_heading': W_HEADING,
        'w_velocity': W_VELOCITY,
        'waypoint_bonus': WAYPOINT_BONUS,
        'goal_bonus': GOAL_BONUS,
        'max_episode_steps': MAX_EPISODE_STEPS,
        'cte_linear': CTE_LINEAR_WEIGHT,
        'cte_squared': CTE_SQUARED_WEIGHT,
        'heading': HEADING_WEIGHT,
        'residual_penalty': RESIDUAL_PENALTY_WEIGHT,
        'trust_bonus_weight_high': TRUST_BONUS_WEIGHT_HIGH,
        'trust_bonus_weight_medium': TRUST_BONUS_WEIGHT_MEDIUM,
        'trust_bonus_decay_high': TRUST_BONUS_DECAY_HIGH,
        'trust_bonus_decay_medium': TRUST_BONUS_DECAY_MEDIUM,
        'trust_bonus_cte_threshold_high': TRUST_BONUS_CTE_THRESHOLD_HIGH,
        'trust_bonus_cte_threshold_medium': TRUST_BONUS_CTE_THRESHOLD_MEDIUM,
        'fighting_penalty_weight': FIGHTING_PENALTY_WEIGHT,
        'fighting_penalty_residual_threshold': FIGHTING_PENALTY_RESIDUAL_THRESHOLD,
        'use_cte_improvement': USE_CTE_IMPROVEMENT_BONUS,
        'cte_improvement_bonus_positive': CTE_IMPROVEMENT_BONUS_POSITIVE,
        'cte_improvement_bonus_negative': CTE_IMPROVEMENT_BONUS_NEGATIVE,
        'cte_trend_detection_threshold': CTE_TREND_DETECTION_THRESHOLD,
    }


# =============================================================================
# CAMERA LOOKAHEAD (body-frame terrain sensing)
# =============================================================================
# Independent flag — can be used by pure PPO, hybrid, or waypoint-offset mode.
# When True, adds 6 terrain slope features to the observation.
USE_CAMERA_LOOKAHEAD = True   # Master flag for camera observation features


# =============================================================================
# PURE PPO PATH TRACKING REWARD
# =============================================================================
# Used when agent_mode = "PPO" and USE_PURE_PPO_REWARD = True.
# Pure PPO directly outputs velocity and omega — no LQR baseline.
# Reward is designed around research findings for RL path tracking:
#   - Quadratic CTE as primary signal
#   - Heading error as secondary signal
#   - Velocity reward CONDITIONAL on tracking quality
#   - Action smoothness penalty
#   - Survival/time penalty to prevent sitting still
#   - Moderate terminal bonuses/penalties

USE_PURE_PPO_REWARD = True     # Use this reward when agent_mode = "PPO"

# v33.9 (2026-05-26): reward rebalanced for hybrid residual learning. Compare
# data showed dense state-quality reward dominated by LQR's behavior — PPO's
# marginal contribution was buried in terrain noise (~1% of per-step reward
# variance). The rebalance shifts gradient signal toward TERMINAL outcomes
# (success vs failure) and away from per-step CTE, matching the sparse-ish
# reward design that Johannink (ICRA 2019), Silver (RSS 2018), and the Volt-Var
# PID-residual paper (2408.06790) all use for residual learning on strong
# baselines. The new PPO_W_EFFORT applies an L2 penalty on residual magnitude
# (hybrid only — gated by use_lqr_baseline in env), pulling PPO toward zero
# residual unless a correction is actively helpful.
PPO_W_CTE = 5.0                # v33.9: 10 → 5. Less dominant per-step signal.
PPO_W_HEADING = 0.5            # v32.2: 2.0 → 0.5. Sharp-curve observation showed policy
                                #   rocking back-and-forth extensively to correct heading.
                                #   CTE is the real objective; heading is a surrogate.
PPO_W_VELOCITY = 0.5           # v33.9: 1.0 → 0.5. Reduced to balance with new effort term.
PPO_CTE_OK_THRESHOLD = 0.5     # v33: 0.3 → 0.5. Vel_scale ramps from +1 at cte=0 down
                                # to 0 at cte=0.5, then stays at 0 (no negative).
PPO_W_SMOOTHNESS = 0.5         # Action smoothness penalty (-w * (|Δv| + |Δω|))
PPO_W_ALIVE = 0.0              # v32: 0.1 → 0.0. Prevents "end episode ASAP" exploit.
PPO_W_PROGRESS = 10.0          # v33.9: 20 → 10. Rebalanced with reduced terminal dominance.
# v33.9 (2026-05-26): L2 penalty on residual magnitude. Only applies in hybrid
# mode (gated by use_lqr_baseline in env). Pulls PPO toward zero residual unless
# a correction actively helps state. Pure PPO ignores this (no LQR baseline →
# no separate residual to penalize; the action IS the policy).
# Term: r_effort = -PPO_W_EFFORT * (residual_v_norm^2 + residual_omega_norm^2)
# where residual_*_norm = actual residual / MAX_RESIDUAL_*  (so range [-1, +1])
PPO_W_EFFORT = 0.5
# v33.9 (2026-05-26): SUCCESS_BONUS 50 → 200, FAILURE_PENALTY 30 → 50. With per-step
# reward weights reduced by ~2x, terminal magnitudes had to grow to maintain
# their visibility in cumulative episode reward. Total episodic reward
# magnitudes balanced so terminals contribute meaningfully to gradient signal.
PPO_SUCCESS_BONUS = 200.0      # Goal reached terminal bonus
PPO_FAILURE_PENALTY = 50.0     # Terminal penalty (CTE too large, stagnation, timeout)
PPO_MAX_CTE_TERMINATION = 2.0  # Terminate episode if CTE exceeds this (meters)


def get_pure_ppo_reward_config():
    """Get pure PPO path tracking reward configuration."""
    return {
        'use_pure_ppo_reward': USE_PURE_PPO_REWARD,
        'ppo_w_cte': PPO_W_CTE,
        'ppo_w_heading': PPO_W_HEADING,
        'ppo_w_velocity': PPO_W_VELOCITY,
        'ppo_cte_ok_threshold': PPO_CTE_OK_THRESHOLD,
        'ppo_w_smoothness': PPO_W_SMOOTHNESS,
        'ppo_w_alive': PPO_W_ALIVE,
        'ppo_w_progress': PPO_W_PROGRESS,
        'ppo_w_effort': PPO_W_EFFORT,
        'ppo_success_bonus': PPO_SUCCESS_BONUS,
        'ppo_failure_penalty': PPO_FAILURE_PENALTY,
        'ppo_max_cte_termination': PPO_MAX_CTE_TERMINATION,
        'use_camera_lookahead': USE_CAMERA_LOOKAHEAD,
    }


# =============================================================================
# v29 WAYPOINT-OFFSET ARCHITECTURE
# =============================================================================
# Instead of PPO adding velocity/omega residuals on top of LQR, PPO shifts
# LQR's target waypoint laterally (route planning) and scales speed.
# LQR tracks the shifted waypoint with full authority — no fighting.
#
# Coupled with body-frame terrain lookahead via raycasting (simulating a
# forward-facing stereo camera), PPO gets 3-4 seconds of lead time to plan
# detours around steep terrain.
#
# When USE_WAYPOINT_OFFSET = False, the system behaves identically to v28
# (residual architecture). All new code is behind feature flags.

USE_WAYPOINT_OFFSET = False  # Master flag: False = v28 residual, True = v29 waypoint-offset

# --- Waypoint Offset Parameters ---
MAX_LATERAL_OFFSET = 3.0     # meters — max perpendicular shift of LQR target waypoint
                              # 3m is enough to route around a 2m-wide hill feature
SPEED_SCALE_MIN = 0.3        # Minimum speed factor (fraction of profiled velocity)
SPEED_SCALE_MAX = 1.0        # Maximum speed factor
# Mapping: speed = SPEED_SCALE_MIN + (SPEED_SCALE_MAX - SPEED_SCALE_MIN) * (action[1] + 1) / 2
# action[1] = -1 → speed = 0.3, action[1] = 0 → speed = 0.65, action[1] = +1 → speed = 1.0

# --- Camera Raycasting (body-frame terrain lookahead) ---
# Simulates XVisio SeerSense DS80 stereo camera mounted on rover body.
# Rays cast from rover body position, in rover's facing direction.
CAMERA_FORWARD_OFFSET = 0.15   # meters forward of rover center (body frame)
CAMERA_HEIGHT_OFFSET = 0.10    # meters above rover center (body frame)
CAMERA_MAX_RANGE = 2.0         # meters — max raycast distance
CAMERA_H_ANGLES = [-40.0, -20.0, 0.0, 20.0, 40.0]   # degrees, horizontal fan
CAMERA_V_ANGLES = [-25.0, -10.0, 5.0]                  # degrees, vertical fan
# Zone boundaries for binning hit points (meters from rover)
CAMERA_ZONE_NEAR = 0.7    # 0 to 0.7m
CAMERA_ZONE_MID = 1.2     # 0.7 to 1.2m
# Far: 1.2 to CAMERA_MAX_RANGE (2.0m)
CAMERA_MIN_HITS_PER_BIN = 3   # fewer hits → sentinel value -1.0

# --- Camera Domain Randomization (off by default) ---
CAMERA_NOISE_STD = 0.0            # Gaussian noise on hit positions (0 = off)
CAMERA_FAR_DROPOUT_PROB = 0.0     # Probability of far bin returning sentinel (0 = off)

# --- Lookahead Authority Gate ---
# PPO's lateral offset and speed scaling are gated by lookahead data.
# Flat terrain ahead → gate=0 (pure LQR on original path).
# Steep terrain ahead → gate ramps to 1.0 (full PPO authority).
LOOKAHEAD_GATE_SLOPE_THRESHOLD = 0.5   # slope value where difficulty saturates
LOOKAHEAD_GATE_ONSET = 0.2    # smoothstep onset (difficulty below this → gate=0)
LOOKAHEAD_GATE_FULL = 0.6     # smoothstep full (difficulty above this → gate=1)

# --- Waypoint-Offset Reward Weights ---
WO_TIME_PENALTY = 1.0            # constant per-step cost
WO_W_PROXIMITY = 0.5             # soft CTE Gaussian weight
WO_PROXIMITY_SIGMA = 1.5         # CTE Gaussian sigma (meters) — MUCH wider than v28's 0.10m
WO_W_HEADING = 0.2               # heading alignment weight
WO_HEADING_SIGMA = 0.15          # heading Gaussian sigma (radians)
WO_W_PROGRESS = 500.0            # path advance weight (PRIMARY signal)
WO_TERRAIN_PROGRESS_BOOST = 2.0  # extra progress multiplier on hard terrain
WO_W_TERRAIN = 5.0               # penalty for routing through steep terrain
WO_SAFE_SLOPE_THRESHOLD = 0.35   # slope below this is safe (~20°)
WO_SUCCESS_BONUS = 500.0         # terminal success reward
WO_FAILURE_PENALTY = 150.0       # terminal failure penalty
WO_PROGRESS_CAP = 0.002          # cap per-step progress delta (same as v28)


def get_waypoint_offset_config():
    """Get v29 waypoint-offset architecture configuration."""
    return {
        'use_waypoint_offset': USE_WAYPOINT_OFFSET,
        'max_lateral_offset': MAX_LATERAL_OFFSET,
        'speed_scale_min': SPEED_SCALE_MIN,
        'speed_scale_max': SPEED_SCALE_MAX,
        # Camera raycasting
        'camera_forward_offset': CAMERA_FORWARD_OFFSET,
        'camera_height_offset': CAMERA_HEIGHT_OFFSET,
        'camera_max_range': CAMERA_MAX_RANGE,
        'camera_h_angles': CAMERA_H_ANGLES,
        'camera_v_angles': CAMERA_V_ANGLES,
        'camera_zone_near': CAMERA_ZONE_NEAR,
        'camera_zone_mid': CAMERA_ZONE_MID,
        'camera_min_hits_per_bin': CAMERA_MIN_HITS_PER_BIN,
        'camera_noise_std': CAMERA_NOISE_STD,
        'camera_far_dropout_prob': CAMERA_FAR_DROPOUT_PROB,
        # Lookahead gate
        'lookahead_gate_slope_threshold': LOOKAHEAD_GATE_SLOPE_THRESHOLD,
        'lookahead_gate_onset': LOOKAHEAD_GATE_ONSET,
        'lookahead_gate_full': LOOKAHEAD_GATE_FULL,
        # Reward weights
        'wo_time_penalty': WO_TIME_PENALTY,
        'wo_w_proximity': WO_W_PROXIMITY,
        'wo_proximity_sigma': WO_PROXIMITY_SIGMA,
        'wo_w_heading': WO_W_HEADING,
        'wo_heading_sigma': WO_HEADING_SIGMA,
        'wo_w_progress': WO_W_PROGRESS,
        'wo_terrain_progress_boost': WO_TERRAIN_PROGRESS_BOOST,
        'wo_w_terrain': WO_W_TERRAIN,
        'wo_safe_slope_threshold': WO_SAFE_SLOPE_THRESHOLD,
        'wo_success_bonus': WO_SUCCESS_BONUS,
        'wo_failure_penalty': WO_FAILURE_PENALTY,
        'wo_progress_cap': WO_PROGRESS_CAP,
        # Stagnation (reuse existing settings)
        'stagnation_velocity_threshold': STAGNATION_VEL_THRESHOLD,
        'stagnation_window': STAGNATION_WINDOW,
        'stagnation_penalty_scale': STAGNATION_PENALTY_SCALE,
        'stagnation_max_penalty': STAGNATION_MAX_PENALTY,
        'stagnation_termination_steps': STAGNATION_TERMINATION_STEPS,
        'recovery_payback_ratio': RECOVERY_PAYBACK_RATIO,
        'recovery_payback_cap': RECOVERY_PAYBACK_CAP,
    }


# ── GUI Override Mechanism ─────────────────────────────────────────────
# When launched from experiment_gui.py, overrides are passed via the
# EXPERIMENT_OVERRIDES environment variable as a JSON string.
import os as _os
import json as _json

_overrides_json = _os.environ.get('EXPERIMENT_OVERRIDES')
if _overrides_json:
    try:
        _overrides = _json.loads(_overrides_json)
        _g = globals()
        for _key, _value in _overrides.items():
            if _key in _g:
                _g[_key] = _value
        print(f"[config] Applied {len(_overrides)} GUI overrides: {list(_overrides.keys())}")
    except _json.JSONDecodeError as _e:
        print(f"[config] Warning: Failed to parse EXPERIMENT_OVERRIDES: {_e}")


# ── ADR mode-conditional defaults (v33.5, 2026-05-12) ──────────────────
# Hybrid mode has an LQR baseline that already drives the rover competently
# from episode 1, so ADR settings calibrated for pure PPO (which starts from
# random noise on near-flat terrain) are too lenient for hybrid:
#   - START=10% is trivially easy for LQR alone, so the residual policy
#     gets free wins it doesn't learn from for the first ~5k episodes.
#   - SUCCESS_THRESHOLD=0.70 advances ADR before the residual has truly
#     mastered each level, because LQR is padding the success rate.
#   - CTE_THRESHOLD=0.10m doesn't gate anything because LQR already
#     tracks well below that on its own.
# Applied AFTER the GUI override loader above, so any explicit per-run
# override (if added to the GUI in the future) still wins.
if agent_mode == "Hybrid":
    ADR_TERRAIN_MAX_START = 30.0    # was 10.0; LQR can handle 30% from step 1
    # v33.6 (2026-05-15): SUCCESS_THRESHOLD 0.85 → 0.75. At 0.85, ADR stalled
    # at terrain 63% in the 20260513_091206 run because the bar might be
    # physically unreachable at higher terrain (slopes >30° at 100% intensity
    # may genuinely cap real-world success at 70-80% regardless of policy
    # quality). 0.75 still accounts for LQR padding (pure PPO uses 0.70).
    ADR_SUCCESS_THRESHOLD = 0.75
    # v33.6 (2026-05-15): CTE_THRESHOLD 0.05 → 0.085. The 0.05 gate was the
    # binding constraint in the 20260513_091206 run — success rate reached
    # 0.90 at terrain 63% but mean CTE stayed at 0.07-0.08 m, so ADR never
    # advanced even though the success gate was clearly cleared. 0.085 is
    # ~10% above the realistic terrain-63 mean_cte, letting ADR advance once
    # the policy is "good enough" rather than demanding "perfect." Still
    # tighter than the pure-PPO 0.10 default.
    ADR_CTE_THRESHOLD     = 0.085
    # v33.7 (2026-05-15): tighten REGRESSION thresholds. Defaults (0.50 / 0.50m)
    # only fire on catastrophic policy collapse, which left an asymmetric
    # deadband: a policy can advance on a lucky 200-ep window (~15% chance of
    # observing >=0.75 even at true 0.70 capability) but almost never regress
    # once over-leveled (would need 6σ-below-mean window to observe <0.50).
    # Tighter regression (0.65 success / 0.15m CTE) maintains a 10pp hysteresis
    # vs the advance gate, catching stuck-above-level policies within a few
    # evaluation windows instead of locking them at unmasterable levels.
    ADR_REGRESSION_SUCCESS_THRESHOLD = 0.65    # was 0.50
    ADR_REGRESSION_CTE_THRESHOLD     = 0.15    # was 0.50
    print(f"[config] Hybrid mode: ADR tuned for LQR baseline "
          f"(start={ADR_TERRAIN_MAX_START:.0f}%, "
          f"success_thresh={ADR_SUCCESS_THRESHOLD:.2f}, "
          f"cte_thresh={ADR_CTE_THRESHOLD:.3f}m, "
          f"regress_succ={ADR_REGRESSION_SUCCESS_THRESHOLD:.2f}, "
          f"regress_cte={ADR_REGRESSION_CTE_THRESHOLD:.3f}m)")