# controllers/ — GPU-vectorized controllers

The PyBullet code had a `Controller2` class that ran one rover's LQR
computation per `step()` call. In Isaac Lab, we run thousands of rovers in
parallel each step, so the controller must operate on `[num_envs, ...]`
tensors using PyTorch ops, not numpy + Python loops.

## What lives here

| file | port phase | corresponds to in PyBullet |
|------|------------|----------------------------|
| `lqr.py` | Phase 4 | `Controller2._compute_lqr_action` |
| `controller_utils.py` | Phase 4 | misc helpers from `Controller2` |

## Why a GPU-side LQR matters

The whole point of Isaac Lab is that the policy network, environment
physics, and observation construction all live on the GPU. If LQR runs on
CPU and the result has to be copied back to GPU for action application,
you serialize across the PCIe bus every step — that single sync can eat
30%+ of your training throughput.

LQR is just matrix multiplication. Implementing it in torch:

```python
# state_tensor: [num_envs, 3] — (cte, heading_error, omega_z) in body frame
# K_lqr: [2, 3] — precomputed constant
action_tensor = -state_tensor @ K_lqr.T   # [num_envs, 2]
```

That's the entire LQR step, fully vectorized, GPU-resident, no sync.

## Gotchas

- The `K_lqr` matrix is constant after Q/R are picked. Pre-compute it in
  `__init__`, register it as a buffer (`self.register_buffer("K", ...)`),
  not a parameter. Buffers move to cuda:0 automatically when the parent
  module does.
- The reference state in the PyBullet code is in the *waypoint* frame, not
  the world frame. The body-frame error computation needs porting too —
  use `quat_rotate_inverse` from `isaaclab.utils.math`.
- The PyBullet `Controller2` includes velocity smoothing and a
  `_smoothed_target_yaw` low-pass filter. Both should port cleanly to a
  per-env EMA tensor maintained as state on the env.
