#!/usr/bin/env python
# scripts/eval_report.py
"""
Side-by-side comparison of evaluation CSVs (hybrid vs LQR vs PPO, or any set).

Reads one or more CSVs written by evaluate_policy.py (the episode_metrics schema)
and prints an overall table plus success-and-progress broken down by terrain
level, so you can compare algorithms directly. Pure stdlib -- runs anywhere
(laptop or box), no pandas/numpy.

Usage:
    python3 scripts/eval_report.py hybrid=evals/hybrid_x.csv lqr=evals/lqr_x.csv ppo=evals/ppo_x.csv
    python3 scripts/eval_report.py evals/hybrid_x.csv evals/lqr_x.csv      # labels inferred from filename

Each argument is either "label=path" or just "path" (label taken from the
filename prefix before the first underscore).
"""
from __future__ import annotations
import sys, os, csv


def _f(r, k):
    try:
        return float(r[k])
    except Exception:
        return 0.0


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        out.append({
            "success": _f(r, "success"),
            "progress": _f(r, "path_progress"),
            "cte": _f(r, "mean_cte"),
            "steps": _f(r, "steps"),
            "terr": _f(r, "terrain_intensity"),
        })
    return out


def pct(xs):
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else float("nan")


def main():
    raw_args = sys.argv[1:]
    if not raw_args:
        print(__doc__)
        sys.exit(1)

    datasets = []  # (label, rows)
    for a in raw_args:
        if "=" in a:
            label, path = a.split("=", 1)
        else:
            path = a
            base = os.path.basename(path)
            label = base.split("_")[0] if "_" in base else os.path.splitext(base)[0]
        if not os.path.isfile(path):
            print(f"!! not found: {path}")
            continue
        try:
            rows = load(path)
        except Exception as e:
            print(f"!! could not read {path}: {e}")
            continue
        if not rows:
            print(f"!! no episodes in {path}")
            continue
        datasets.append((label, rows))

    if not datasets:
        print("no readable eval CSVs given")
        sys.exit(1)

    labels = [d[0] for d in datasets]

    # ---------------- overall ----------------
    print("\n=================== EVAL COMPARISON ===================")
    print("OVERALL  (deterministic, held-out across the terrain sweep)")
    print(f"  {'algorithm':<10} {'episodes':>9} {'success%':>9} {'medprog%':>9} {'meanCTE':>8} {'parked%':>8}")
    for label, rows in datasets:
        succ = [r["success"] for r in rows]
        prog = [r["progress"] for r in rows]
        cte = [r["cte"] for r in rows]
        parked = [1.0 if r["progress"] < 5 else 0.0 for r in rows]
        print(f"  {label:<10} {len(rows):>9} {pct(succ):>9.1f} {median(prog):>9.1f} "
              f"{sum(cte)/len(cte):>8.3f} {pct(parked):>8.1f}")

    # ---------------- per-terrain-level tables ----------------
    # bin terrain to the nearest integer percent (all algos share the same --levels)
    def levels_of(rows):
        return sorted({round(r["terr"]) for r in rows})
    all_levels = sorted({L for _, rows in datasets for L in levels_of(rows)})

    def by_level(rows, L, key, as_pct):
        sub = [r[key] for r in rows if round(r["terr"]) == L]
        if not sub:
            return None
        return pct(sub) if as_pct else sum(sub) / len(sub)

    def table(title, key, as_pct, dec=1):
        print(f"\n{title}")
        head = "  " + f"{'terrain%':>9}" + "".join(f"{lab:>10}" for lab in labels)
        print(head)
        for L in all_levels:
            cells = []
            for _, rows in datasets:
                v = by_level(rows, L, key, as_pct)
                cells.append("        --" if v is None else f"{v:>10.{dec}f}")
            print("  " + f"{L:>9}" + "".join(cells))

    table("SUCCESS % BY TERRAIN LEVEL  (higher = better)", "success", True, 1)
    table("MEAN PROGRESS % BY TERRAIN LEVEL  (higher = better)", "progress", False, 1)
    table("MEAN CTE BY TERRAIN LEVEL  (lower = better, metres)", "cte", False, 3)

    print("\n======================================================")
    print("Reading it: 'success%' is the fraction of episodes that reached the goal.")
    print("A good hybrid should match or beat LQR at every terrain level; if LQR wins")
    print("everywhere, the PPO residual is not yet adding value. PPO is the from-scratch")
    print("learner (no LQR baseline) -- expect it to trail on the harder rows.\n")


if __name__ == "__main__":
    main()
