#!/usr/bin/env python3
# scripts/multiworld_stats.py
"""
Pooled paired statistics over a MULTI-WORLD paired evaluation (see `leo multieval`).

Input layout (produced by cmd_multieval in leo.sh):

    evals/multiworld/<ts>/
        scenarios.npz            # one shared scenario list (not read here)
        world_<seed>/hybrid.csv  # paired legs for one regenerated world
        world_<seed>/lqr.csv

Within a world, rows join on scenario_id (identical path + spawn + terrain cell +
friction for both controllers). Across worlds the scenario list is the same; only the
world realization (terrain bank + soil map) differs. This script reports:

  1. POOLED paired tests over all (world, scenario) pairs:
       H1 tracking  : paired t on mean CTE, Cohen's d_z  (the primary metric)
       H2 completion: McNemar on success
  2. A PER-WORLD table (the generalization view: does the sign hold world after world?)
     plus world-level sign tests, treating each world as one unit - the robust answer
     to "is this specific to one terrain draw?"
  3. Pooled per-terrain-level breakdown.

Pure stdlib (csv/math) so it runs in the Isaac bundled python or any python3.

Usage:  python3 scripts/multiworld_stats.py evals/multiworld/<ts> [--min-n 100]
"""
from __future__ import annotations
import csv, math, os, sys, glob


def _read_pairs(hyb_path: str, lqr_path: str):
    """scenario_id -> (cte, success, level) per controller; first occurrence wins."""
    def load(p):
        d = {}
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    sid = int(float(r["scenario_id"]))
                except (KeyError, ValueError):
                    continue
                if sid < 0 or sid in d:
                    continue
                try:
                    d[sid] = (float(r["mean_cte"]), int(float(r["success"])),
                              float(r["terrain_intensity"]),
                              float(r.get("friction_intensity", 0.0) or 0.0))
                except (KeyError, ValueError):
                    continue
        return d
    H, L = load(hyb_path), load(lqr_path)
    keys = sorted(set(H) & set(L))
    return [(k, H[k], L[k]) for k in keys]


def _mean(xs): return sum(xs) / len(xs) if xs else float("nan")


def _sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _p_norm(z):  # two-sided normal tail (fine at these n)
    return math.erfc(abs(z) / math.sqrt(2.0))


def _p_chi2_1df(x):  # chi-square 1 df upper tail
    return math.erfc(math.sqrt(max(x, 0.0) / 2.0))


def _sign_test(k_better, n):
    """two-sided binomial sign test, p = 2 * P(X >= max(k, n-k)) at p0=0.5."""
    if n == 0:
        return float("nan")
    k = max(k_better, n - k_better)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    root = sys.argv[1]
    min_n = 100
    if "--min-n" in sys.argv:
        min_n = int(sys.argv[sys.argv.index("--min-n") + 1])

    worlds = []
    for d in sorted(glob.glob(os.path.join(root, "world_*"))):
        hp, lp = os.path.join(d, "hybrid.csv"), os.path.join(d, "lqr.csv")
        if not (os.path.isfile(hp) and os.path.isfile(lp)):
            print(f"[skip] {os.path.basename(d)}: missing a leg")
            continue
        pairs = _read_pairs(hp, lp)
        if len(pairs) < min_n:
            print(f"[skip] {os.path.basename(d)}: only {len(pairs)} matched pairs (<{min_n})")
            continue
        worlds.append((os.path.basename(d).replace("world_", ""), pairs))
    if not worlds:
        print("no complete worlds found under", root); sys.exit(1)

    print("\n================ MULTI-WORLD PAIRED EVALUATION ================")
    print(f"root: {root}")
    print(f"worlds: {len(worlds)}   (each = fresh terrain bank + soil map; "
          f"same scenario list everywhere)\n")

    # ---------- per-world table + world-level aggregates ----------
    print(f"{'world':>8} {'n':>7} {'succH%':>7} {'succL%':>7} {'dSucc':>7} "
          f"{'cteH':>7} {'cteL':>7} {'dCTE':>8}")
    w_dcte, w_dsucc = [], []
    all_d, all_b, all_c, all_h1, all_l1 = [], 0, 0, 0, 0
    by_level, by_fric = {}, {}
    for name, pairs in worlds:
        cteH = [h[0] for _, h, _ in pairs]; cteL = [l[0] for _, _, l in pairs]
        sH = [h[1] for _, h, _ in pairs];   sL = [l[1] for _, _, l in pairs]
        dc = _mean(cteH) - _mean(cteL); ds = 100.0 * (_mean(sH) - _mean(sL))
        w_dcte.append(dc); w_dsucc.append(ds)
        print(f"{name:>8} {len(pairs):>7} {100*_mean(sH):>7.1f} {100*_mean(sL):>7.1f} "
              f"{ds:>+7.1f} {_mean(cteH):>7.4f} {_mean(cteL):>7.4f} {dc:>+8.4f}")
        for _, h, l in pairs:
            all_d.append(h[0] - l[0])
            if h[1] and not l[1]: all_b += 1
            if l[1] and not h[1]: all_c += 1
            all_h1 += h[1]; all_l1 += l[1]
            lv = round(h[2])
            acc = by_level.setdefault(lv, [0, 0, 0, 0.0, 0.0])  # n, sH, sL, cteH, cteL
            acc[0] += 1; acc[1] += h[1]; acc[2] += l[1]; acc[3] += h[0]; acc[4] += l[0]
            fb = "low <40" if h[3] < 40.0 else ("mid 40-70" if h[3] < 70.0 else "high >=70")
            fa = by_fric.setdefault(fb, [0, 0, 0, 0.0, 0.0])
            fa[0] += 1; fa[1] += h[1]; fa[2] += l[1]; fa[3] += h[0]; fa[4] += l[0]

    n = len(all_d)
    W = len(worlds)
    print(f"\n{'MEAN':>8} {'':>7} {'':>7} {'':>7} {_mean(w_dsucc):>+7.1f} "
          f"{'':>7} {'':>7} {_mean(w_dcte):>+8.4f}   (+/- sd "
          f"{_sd(w_dsucc):.1f} / {_sd(w_dcte):.4f})")

    kc = sum(1 for d in w_dcte if d < 0)
    ks = sum(1 for d in w_dsucc if d > 0)
    print(f"\nWORLD-LEVEL SIGN TESTS (each world = one unit; the generalization answer)")
    print(f"  CTE lower in    {kc}/{W} worlds   sign-test p = {_sign_test(kc, W):.4f}")
    print(f"  success higher  {ks}/{W} worlds   sign-test p = {_sign_test(ks, W):.4f}")

    # ---------- pooled paired tests ----------
    md, sd = _mean(all_d), _sd(all_d)
    t = md / (sd / math.sqrt(n)) if sd > 0 else float("nan")
    dz = md / sd if sd > 0 else float("nan")
    print(f"\nPOOLED PAIRED TESTS over {n} (world, scenario) pairs")
    print(f"  H1 tracking : meanDeltaCTE = {md:+.4f} m   t = {t:+.2f}   "
          f"p {'<1e-4' if _p_norm(t) < 1e-4 else f'= {_p_norm(t):.4f}'}   d_z = {dz:+.3f}")
    chi2 = ((all_b - all_c) ** 2) / (all_b + all_c) if (all_b + all_c) else float("nan")
    print(f"  H2 completion: success {100.0*all_h1/n:.1f}% vs {100.0*all_l1/n:.1f}%   "
          f"McNemar b/c = {all_b}/{all_c}   chi2 = {chi2:.1f}   "
          f"p {'<1e-4' if _p_chi2_1df(chi2) < 1e-4 else f'= {_p_chi2_1df(chi2):.4f}'}")
    print("  NOTE: pooled scenarios within a world are not independent of that world's "
          "draw; cite the world-level sign tests alongside the pooled tests.")

    # ---------- pooled per-level ----------
    print(f"\nPOOLED BY TERRAIN LEVEL (all worlds)")
    print(f"{'level%':>7} {'n':>7} {'succH%':>7} {'succL%':>7} {'cteH':>7} {'cteL':>7}")
    for lv in sorted(by_level):
        nn, s1, s2, c1, c2 = by_level[lv]
        print(f"{lv:>7} {nn:>7} {100.0*s1/nn:>7.1f} {100.0*s2/nn:>7.1f} "
              f"{c1/nn:>7.4f} {c2/nn:>7.4f}")

    # ---------- pooled per-friction (only informative with scenario-locked friction) ----
    if len(by_fric) > 1:
        print(f"\nPOOLED BY FRICTION INTENSITY (all worlds; scenario-locked per-wheel mu)")
        print(f"{'friction':>10} {'n':>7} {'succH%':>7} {'succL%':>7} {'cteH':>7} {'cteL':>7}")
        for fb in ("low <40", "mid 40-70", "high >=70"):
            if fb not in by_fric:
                continue
            nn, s1, s2, c1, c2 = by_fric[fb]
            print(f"{fb:>10} {nn:>7} {100.0*s1/nn:>7.1f} {100.0*s2/nn:>7.1f} "
                  f"{c1/nn:>7.4f} {c2/nn:>7.4f}")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
# eof
