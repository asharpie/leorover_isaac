# soil.py
"""Terramechanics-lite: a wheel-level soft-soil (sand) model on rigid terrain.

Real granular simulation (PBD/MPM particles) is computationally impossible at
the 1000+ parallel envs training needs, and PhysX colliders are static after
cooking, so runtime ground deformation is closed off. This module implements
the standard real-time rover-sim alternative: keep the rigid collider and add
the three soft-soil effects Coulomb friction cannot express, as an external
wrench computed per wheel each control step (pure torch on [num_envs, 4]):

  1. SINKAGE DRAG      rolling resistance grows as a wheel digs in. Sinkage is
                       a per-wheel state in [0, 1] that accumulates with slip
                       and recovers with clean rolling (Bekker-inspired).
  2. SLIP-THRUST DECAY beyond a peak slip ratio sand thrust FALLS (spinning
                       digs, it does not pull) - unlike Coulomb's flat kinetic
                       cap. Modeled as a force opposing the wheel's drive
                       direction, growing with max(0, |s| - s_peak).
  3. LATERAL SHEAR     sand resists side-slip weakly and plastically; modeled
                       as a saturating (tanh) lateral resistance, so slopes
                       produce real downhill creep the controller must fight.

Every effect scales with a SOIL ZONE MAP: seeded value-noise over the world
(2 m cells, box-smoothed), zone in [0, 1] = firm ... deep sand. The map is a
pure function of the seed, so all controllers in a paired evaluation face the
byte-identical soil field, and training worlds are reproducible.

The summed wrench (force + moment equivalent about the base) is applied to the
BASE link in its body frame, so the spinning wheel frames never matter. All
force terms are resistive (signed against local motion or against drive
thrust), so the model cannot inject energy. Deliberately out of scope for v1:
per-wheel load transfer, pitch-over moments from ground-height force offset.

Enable with LEOROVER_SOIL=1 (or config.SOIL_MODEL = True). Tunables (env):
  LEOROVER_SOIL_CRR    rolling-resistance coeff, firm ground     (default 0.04)
  LEOROVER_SOIL_CSINK  extra drag coeff at full sinkage          (default 0.30)
  LEOROVER_SOIL_SPEAK  peak-thrust slip ratio                    (default 0.25)
  LEOROVER_SOIL_CSLIP  thrust-decay coeff past the peak          (default 0.50)
  LEOROVER_SOIL_CLAT   lateral shear coeff                       (default 0.35)
  LEOROVER_SOIL_KDIG   sinkage growth rate (/s at full slip)     (default 0.50)
  LEOROVER_SOIL_KREC   sinkage recovery rate (/s clean rolling)  (default 0.25)
  LEOROVER_SOIL_SAND   zone shaping exponent (higher = firmer)   (default 1.5)
"""

from __future__ import annotations

import os
import torch
import torch.nn.functional as _F


def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


class TerramechanicsLite:
    """Vectorized soft-soil wheel forces + per-wheel sinkage state + zone map."""

    def __init__(self, num_envs: int, device, seed: int,
                 wheel_x_off, wheel_y_off,          # [4] tensors, chassis frame (m)
                 load_per_wheel_n: float,           # static normal load (N)
                 half_x: float = 42.0, half_y: float = 18.0, cell_m: float = 2.0):
        self.n = num_envs
        self.device = device
        self.crr = _envf("LEOROVER_SOIL_CRR", 0.04)
        self.csink = _envf("LEOROVER_SOIL_CSINK", 0.30)
        self.speak = _envf("LEOROVER_SOIL_SPEAK", 0.25)
        self.cslip = _envf("LEOROVER_SOIL_CSLIP", 0.50)
        self.clat = _envf("LEOROVER_SOIL_CLAT", 0.35)
        self.kdig = _envf("LEOROVER_SOIL_KDIG", 0.50)
        self.krec = _envf("LEOROVER_SOIL_KREC", 0.25)
        sand_exp = _envf("LEOROVER_SOIL_SAND", 1.5)

        self.load_n = float(load_per_wheel_n)
        self.x_off = wheel_x_off.to(device).view(1, 4)
        self.y_off = wheel_y_off.to(device).view(1, 4)
        self.half_x, self.half_y = float(half_x), float(half_y)

        # ── seeded soil zone map: value noise -> 2x box blur -> normalize -> shape ──
        self.nx = max(8, int(round(2 * half_x / cell_m)))
        self.ny = max(8, int(round(2 * half_y / cell_m)))
        g = torch.Generator().manual_seed(int(seed))
        raw = torch.rand((self.ny, self.nx), generator=g)
        for _ in range(2):
            p = _F.pad(raw[None, None], (1, 1, 1, 1), mode="replicate")
            raw = _F.avg_pool2d(p, 3, stride=1)[0, 0]
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        self.zone = raw.pow(sand_exp).to(device)          # [ny, nx] in [0, 1]
        self.sand_frac = float((self.zone > 0.5).float().mean())

        # per-wheel sinkage state, normalized [0, 1]
        self.sink = torch.zeros(num_envs, 4, device=device)

    # ------------------------------------------------------------------ zones
    def zone_at(self, xy_world: torch.Tensor) -> torch.Tensor:
        """Nearest-cell zone lookup for world XY positions. [n] in [0, 1]."""
        ix = ((xy_world[:, 0] + self.half_x) / (2 * self.half_x) * (self.nx - 1))
        iy = ((xy_world[:, 1] + self.half_y) / (2 * self.half_y) * (self.ny - 1))
        ix = ix.round().long().clamp(0, self.nx - 1)
        iy = iy.round().long().clamp(0, self.ny - 1)
        return self.zone[iy, ix]

    def reset_idx(self, env_ids):
        self.sink[env_ids] = 0.0

    # ------------------------------------------------------------------ model
    @torch.no_grad()
    def compute(self, v_long, v_lat, omega_z, wheel_surface_speed, xy_world, dt):
        """One control-step of the soil model.

        Args:
            v_long, v_lat, omega_z: base linear (body-frame) and yaw velocities [n]
            wheel_surface_speed: signed wheel rim speed omega_i * r  [n, 4]
            xy_world: base position in world XY [n, 2]
            dt: control step (s)
        Returns:
            force  [n, 3]  chassis-frame force to apply at the base link
            torque [n, 3]  chassis-frame torque (yaw moment equivalent)
            slip   [n, 4]  per-wheel longitudinal slip ratio in [-1, 1]
            sink   [n, 4]  per-wheel sinkage in [0, 1]
        """
        # hub velocities in the chassis frame (rigid-body kinematics)
        hub_long = v_long.unsqueeze(1) - omega_z.unsqueeze(1) * self.y_off   # [n,4]
        hub_lat = v_lat.unsqueeze(1) + omega_z.unsqueeze(1) * self.x_off

        # longitudinal slip ratio, gated near wheel standstill
        denom = wheel_surface_speed.abs().clamp(min=0.05)
        slip = ((wheel_surface_speed - hub_long) / denom).clamp(-1.0, 1.0)
        slip = slip * (wheel_surface_speed.abs() > 0.02)

        zone = self.zone_at(xy_world).unsqueeze(1)                            # [n,1]

        # sinkage dynamics: dig with slip (scaled by soil), recover by rolling
        moving = (hub_long.abs() > 0.02).float()
        self.sink = (self.sink + dt * (self.kdig * slip.abs() * zone
                                       - self.krec * (1.0 - slip.abs()) * moving)
                     ).clamp(0.0, 1.0)

        N = self.load_n
        # 1) rolling + sinkage drag, opposes hub motion (saturating sign)
        f_drag = -(self.crr + self.csink * self.sink) * zone * N * torch.tanh(hub_long / 0.05)
        # 2) thrust decay past peak slip, opposes the DRIVE direction
        f_decay = -self.cslip * zone * N * (slip.abs() - self.speak).clamp(min=0.0) \
            * torch.sign(wheel_surface_speed)
        # 3) lateral shear, opposes side-slip
        f_lat = -self.clat * zone * N * torch.tanh(hub_lat / 0.10)

        f_long = f_drag + f_decay                                             # [n,4]
        force = torch.stack([f_long.sum(1), f_lat.sum(1),
                             torch.zeros_like(v_long)], dim=-1)               # [n,3]
        # yaw moment of the per-wheel forces about the base: tau_z = x*Fy - y*Fx
        tau_z = (self.x_off * f_lat - self.y_off * f_long).sum(1)
        torque = torch.stack([torch.zeros_like(tau_z), torch.zeros_like(tau_z),
                              tau_z], dim=-1)
        return force, torque, slip, self.sink
