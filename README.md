# leorover_isaac

GPU-accelerated Isaac Lab port of the Leo Rover RL stack. Sibling repo to
[leoroverpybullet_share](../leoroverpybullet_share). The PyBullet version
remains the production codebase for the URCA paper and any near-term work;
this repo is the long-horizon successor.

**Status:** scaffolding only. No working code yet. The directory tree,
documentation, and stub files exist so the project has a defined shape
before any phase-1 work begins.

---

## Quick start

This repo is checked out as a sibling of leoroverpybullet_share, e.g.::

    Downloads/
        leoroverpybullet_share - Checkpoint Working (3) (1)/
        leorover_isaac/    ← you are here

If you haven't moved it yet:

1. Move this folder out of the PyBullet repo to be a sibling of it.
2. Run `git init && git add . && git commit -m "Initial scaffolding"` inside.
3. Push to a new GitHub repo (`gh repo create leorover_isaac --public ...`).

Then follow **[CLOUD_SETUP.md](CLOUD_SETUP.md)** to set up cloud GPU
access via RunPod. This is the recommended path — you keep Windows + your
games entirely untouched and rent a GPU on-demand (~$10–40/month for
part-time research). First training run in under an hour from signup.

If you'd rather have a fully local setup, [INSTALL.md](INSTALL.md) covers
the dual-boot Ubuntu 22.04 alternative. The original WSL2 plan turned out
to be unworkable on this Windows build — see [SETUP_HISTORY.md](SETUP_HISTORY.md)
for the debugging story.

After install (either path), follow [PORTING_ROADMAP.md](PORTING_ROADMAP.md)
for the phase-by-phase port plan.

---

## Repo map

```
leorover_isaac/
├── README.md                    ← you are here
├── CLOUD_SETUP.md               ← RunPod cloud GPU setup (recommended)
├── INSTALL.md                   ← Ubuntu 22.04 dual-boot (alternative)
├── PORTING_ROADMAP.md           ← phase-by-phase port plan + concept mapping
├── SETUP_HISTORY.md             ← what we tried, what didn't work, why
├── LICENSE                      ← MIT
├── pyproject.toml               ← Python package metadata
├── .gitignore                   ← Python / Isaac / generated USD ignores
│
├── leorover_isaac/              ← the importable Python package
│   ├── __init__.py              ← package root, registers gym tasks
│   ├── assets/leo_rover/        ← Leo Rover URDF source + generated USD
│   ├── envs/                    ← DirectRLEnv subclasses (Phase 2+)
│   ├── controllers/             ← Vectorized LQR (Phase 4)
│   ├── tasks/                   ← Task configs + gym registrations (Phase 2+)
│   ├── terrain/                 ← Mars heightfield generation (Phase 3)
│   └── utils/                   ← Shared math, logging glue
│
├── scripts/                     ← train.py, eval.py, comparison entry points
├── tests/                       ← unit tests (port validation against PyBullet)
└── docs/                        ← additional notes (sim2real, etc.)
```

Each subdirectory has its own README explaining what belongs there and
how it maps to the PyBullet codebase.

---

## Relationship to leoroverpybullet_share

This repo does NOT replace the PyBullet one. They will coexist indefinitely
for these reasons:

1. **Reverting is free.** If Isaac Lab turns out to have a fatal limitation
   for Mars terrain, the PyBullet code continues to work — there's no
   bridge or dependency between the two.
2. **PyBullet is the URCA paper's experimental platform.** All paper data,
   all checkpoints, all reproducible results live in that repo. Don't break
   that for anything.
3. **Validation requires comparison.** The port is considered "done" when
   the Isaac Lab version reproduces PyBullet success rates within 2 pp on
   identical terrain seeds. That comparison requires both codebases
   functioning.

Useful reference points in the PyBullet repo:

- `config.py` lines 1080–1135 — v33.9 reward weights
- `leoroverpybullet/envs/environment2.py` — the env logic to port
- `controller2.py` (in same folder) — LQR controller to vectorize
- `v33.9_changes.txt` — what each tuning decision was for

---

## Why Isaac Lab over PyBullet

Headline numbers from typical wheeled-robot RL benchmarks:

| metric | PyBullet (12 envs) | Isaac Lab (4096 envs) | speedup |
|--------|--------------------|------------------------|---------|
| sim steps / sec | ~5,000 | ~500,000 | 100x |
| wall-clock for 50M steps | ~5 days | ~1 day | 5x |
| time-to-first-success on a new reward | hours of debugging | minutes | qualitative |
| sim2real domain randomization throughput | bottlenecked | trivial | qualitative |

The 5x wall-clock speedup is less than the 100x sim speedup because PPO
overhead, env reset, and observation construction don't scale linearly
with parallelism. Still substantial.

---

## What is NOT in this port

Some PyBullet features intentionally don't migrate (see PORTING_ROADMAP.md
"What to NOT port" section for the full list). The big ones:

- The Tk-based `experiment_gui.py` — replaced by Isaac Lab's CLI + wandb
- `multiprocessing.Pool` parallelization in `comparison_workers.py` —
  Isaac Lab's native vectorization replaces it
- The `PerformanceRollbackCallback` — Isaac Lab's training is stable enough
  to not need it

---

## Reading order for a new contributor

1. [README.md](README.md) (this file) — orientation
2. [INSTALL.md](INSTALL.md) — get the dev env working
3. [PORTING_ROADMAP.md](PORTING_ROADMAP.md) — what to build and in what order
4. [leorover_isaac/envs/README.md](leorover_isaac/envs/README.md) — env API mapping
5. [leorover_isaac/controllers/README.md](leorover_isaac/controllers/README.md) — vectorized LQR notes
6. [leorover_isaac/terrain/README.md](leorover_isaac/terrain/README.md) — Mars terrain notes
7. [leorover_isaac/tasks/README.md](leorover_isaac/tasks/README.md) — task config pattern
8. [leorover_isaac/assets/leo_rover/README.md](leorover_isaac/assets/leo_rover/README.md) — asset conversion

That's enough background to start on Phase 1.
