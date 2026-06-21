#!/usr/bin/env python3
# scripts/leo_report.py
# Comprehensive, copy-pasteable performance report from a run's episode_metrics.csv.
# Pure standard library (no pandas/numpy) so it runs with any python3 on the box.
#
#   python3 scripts/leo_report.py [path/to/episode_metrics.csv]
# With no argument it auto-finds the latest hybrid run. `leo report` calls this.
from __future__ import annotations
import csv, sys, os, glob, statistics as st, datetime


def find_csv():
    runs = sorted(glob.glob("logs/leo_rover_mars_hybrid/*/csv/episode_metrics.csv"),
                  key=os.path.getmtime)
    return runs[-1] if runs else None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else find_csv()
    if not path or not os.path.isfile(path):
        print("No episode_metrics.csv found. Pass the path as an argument.")
        sys.exit(1)

    rows = []
    with open(path, newline="") as f:
        for d in csv.DictReader(f):
            try:
                rows.append(dict(
                    succ=float(d["success"]), prog=float(d["path_progress"]),
                    rew=float(d["total_reward"]), cte=float(d["mean_cte"]),
                    maxcte=float(d["max_cte"]), steps=float(d["steps"]),
                    terr=float(d["terrain_intensity"]),
                    rv=float(d["mean_residual_v_norm"]), rw=float(d["mean_residual_w_norm"]),
                ))
            except (KeyError, ValueError):
                continue  # skip header dupes / a half-written final line (CSV is live)

    n = len(rows)
    if n == 0:
        print("CSV has no complete episodes yet (run just started).")
        sys.exit(0)

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    col = lambda k, s=rows: [x[k] for x in s]

    line = "=" * 70
    print(line)
    print(f"LEO HYBRID PERFORMANCE REPORT")
    print(f"  source   : {path}")
    print(f"  episodes : {n}")
    print(f"  updated  : {datetime.datetime.fromtimestamp(os.path.getmtime(path)):%Y-%m-%d %H:%M}")
    print(line)

    print("OVERALL (whole run)")
    print(f"  success rate    : {100*mean(col('succ')):5.1f}%   ({int(sum(col('succ')))}/{n} reached the goal)")
    print(f"  path_progress   : mean {mean(col('prog')):5.1f}%   median {st.median(col('prog')):5.1f}%   best {max(col('prog')):5.1f}%")
    print(f"  reward/episode  : mean {mean(col('rew')):8.1f}   median {st.median(col('rew')):8.1f}")
    print(f"  mean CTE        : {mean(col('cte')):.3f} m   (avg per-episode max-CTE {mean(col('maxcte')):.3f} m)")
    print(f"  episode length  : mean {mean(col('steps')):6.0f} steps  (cap 2000)")
    print(f"  residual norms  : v {mean(col('rv')):.3f}   w {mean(col('rw')):.3f}   (~0 means PPO is not adding to the LQR)")

    k = max(1, n // 10)
    rec = rows[-k:]
    print(f"\nRECENT (last {k} episodes = current behaviour)")
    print(f"  success {100*mean(col('succ',rec)):5.1f}%   progress mean {mean(col('prog',rec)):5.1f}%   "
          f"reward {mean(col('rew',rec)):7.0f}   CTE {mean(col('cte',rec)):.3f}   steps {mean(col('steps',rec)):5.0f}")

    B = min(12, n)
    print(f"\nTREND over training ({B} bins, earliest -> latest)")
    print(f"  {'bin':>3} {'eps':>6} {'succ%':>6} {'prog%':>6} {'reward':>8} {'CTE':>6} {'steps':>6} {'resid_v':>7}")
    for i in range(B):
        s = rows[i * n // B:(i + 1) * n // B]
        if not s:
            continue
        print(f"  {i:>3} {len(s):>6} {100*mean(col('succ',s)):>6.1f} {mean(col('prog',s)):>6.1f} "
              f"{mean(col('rew',s)):>8.1f} {mean(col('cte',s)):>6.3f} {mean(col('steps',s)):>6.0f} {mean(col('rv',s)):>7.3f}")

    print(f"\nPROGRESS DISTRIBUTION (whole run)")
    for label, b in [("reached goal", lambda x: x['succ'] > 0.5),
                     (">=75% of path", lambda x: x['prog'] >= 75),
                     (">=50% of path", lambda x: x['prog'] >= 50),
                     (">=25% of path", lambda x: x['prog'] >= 25),
                     ("<5% (parked)", lambda x: x['prog'] < 5)]:
        c = sum(1 for x in rows if b(x))
        print(f"  {label:<16}: {c:>7}  ({100*c/n:4.1f}%)")

    print(f"\nBY TERRAIN DIFFICULTY (terrain_intensity %)")
    print(f"  {'band':>9} {'eps':>6} {'succ%':>6} {'prog%':>6}")
    for lo, hi in [(0, 10), (10, 25), (25, 50), (50, 101)]:
        s = [x for x in rows if lo <= x['terr'] < hi]
        if not s:
            continue
        print(f"  {f'{lo}-{hi-1}':>9} {len(s):>6} {100*mean(col('succ',s)):>6.1f} {mean(col('prog',s)):>6.1f}")
    print(line)


if __name__ == "__main__":
    main()
