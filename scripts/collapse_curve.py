#!/usr/bin/env python3
"""collapse_curve.py - aggregate a collapse_sweep manifest into the collapse curve.

Usage: python3 scripts/collapse_curve.py evals/collapse_manifest_<stamp>.txt

Per harshness point H prints success%, mean CTE, parked%, and the hybrid-vs-LQR
CTE gap; then reports each controller's collapse threshold (first H with
success < 5%) and the suggested eval cap (last H where any controller >= 5%).
Stdlib only.
"""
import csv
import sys


def col(rows, *needles, exclude=()):
    for c in rows[0]:
        lc = c.lower()
        if all(n in lc for n in needles) and not any(x in lc for x in exclude):
            return c
    return None


def leg(path):
    try:
        rows = list(csv.DictReader(open(path)))
    except OSError:
        return None
    if not rows:
        return None
    sc = col(rows, "succ")
    cc = col(rows, "cte", exclude=("max",))
    pc = col(rows, "park")
    n = len(rows)
    succ = 100.0 * sum(float(r[sc]) > 0.5 for r in rows) / n if sc else float("nan")
    cte = sum(float(r[cc]) for r in rows) / n if cc else float("nan")
    park = 100.0 * sum(float(r[pc]) > 0.5 for r in rows) / n if pc else 0.0
    return dict(n=n, succ=succ, cte=cte, park=park)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    pts = []
    for line in open(sys.argv[1]):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        h, d = float(parts[0]), parts[1]
        if d == "FAILED":
            print(f"  H={h:.2f}: FAILED point, skipped")
            continue
        hy, lq = leg(f"{d}/hybrid.csv"), leg(f"{d}/lqr.csv")
        py = leg(f"{d}/ppo.csv")
        if hy and lq:
            pts.append((h, d, hy, lq, py))
        else:
            print(f"  H={h:.2f}: missing/empty CSVs in {d}, skipped")
    if not pts:
        sys.exit("no usable points")
    pts.sort(key=lambda p: p[0])

    print("\n================= COLLAPSE CURVE (paired, same scenarios per point) =================")
    print(f"{'H':>5} {'hyb_succ%':>10} {'lqr_succ%':>10} {'ppo_succ%':>10} {'hyb_CTE':>9} {'lqr_CTE':>9} "
          f"{'CTEgap%':>8} {'eps':>6}")
    for h, d, hy, lq, py in pts:
        gap = 100.0 * (hy["cte"] - lq["cte"]) / lq["cte"] if lq["cte"] else float("nan")
        pcell = f"{py['succ']:10.1f}" if py else "       n/a"
        print(f"{h:5.2f} {hy['succ']:10.1f} {lq['succ']:10.1f} {pcell} {hy['cte']:9.3f} {lq['cte']:9.3f} "
              f"{gap:8.1f} {min(hy['n'], lq['n']):6d}")

    def collapse_h(idx):
        for p in pts:
            c = p[idx]
            if c is not None and c["succ"] < 5.0:
                return p[0]
        return None

    ch, cl, cp = collapse_h(2), collapse_h(3), collapse_h(4)
    print("\nCollapse thresholds (first H with success < 5%):")
    print(f"  ppo    : {'H=%.2f' % cp if cp else 'not reached in sweep'}")
    print(f"  LQR    : {'H=%.2f' % cl if cl else 'not reached in sweep'}")
    print(f"  hybrid : {'H=%.2f' % ch if ch else 'not reached in sweep'}")
    alive = [p[0] for p in pts if max(p[2]["succ"], p[3]["succ"]) >= 5.0]
    if alive:
        print(f"\nSuggested eval cap: H={max(alive):.2f} "
              f"(last point where at least one controller still completes >= 5%)")
    print("Reading: CTEgap% negative = hybrid tracks better. Near collapse, CTE is censored")
    print("(conditioned on tiny progress) - trust the success columns there, not CTE.")


if __name__ == "__main__":
    main()
