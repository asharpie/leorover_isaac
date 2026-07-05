#!/usr/bin/env python
# scripts/paired_stats.py
"""
Paired statistics over scenario-locked eval CSVs from paired_eval.py.

Because every controller ran the IDENTICAL scenarios (same path + terrain patch + pose +
friction), rows join on scenario_id and we can run the paired tests the paper names:
  * MCTE / tracking accuracy : paired Student t-test + Cohen's d_z (H1: Hybrid < baselines)
  * task completion (success): McNemar's test on paired binary outcomes (H2: Hybrid > baselines)
  * Bonferroni correction across the pairwise family.
Plus matched breakdowns by path geometry (zig-zag/curved/polygon) and by terrain-slope bin.

Pure stdlib -- runs on the laptop, no numpy/scipy.

Usage:
    python3 scripts/paired_stats.py hybrid=evals/paired/hybrid.csv \
        lqr=evals/paired/lqr.csv ppo=evals/paired/ppo.csv
The controller labelled 'hybrid' (or the first one) is the reference for the pairwise tests.
"""
from __future__ import annotations
import sys, os, csv, math


# --------------------------------------------------------------------------- stats
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    """Two-sided p-value for a Student-t statistic with df degrees of freedom."""
    if df <= 0:
        return float("nan")
    if t == 0.0:
        return 1.0
    x = df / (df + t * t)
    return max(0.0, min(1.0, _betai(0.5 * df, 0.5, x)))


def paired_t(a, b):
    """Paired t-test on aligned samples a,b. Returns t, two-sided p, Cohen's d_z, mean diff."""
    diff = [ai - bi for ai, bi in zip(a, b)]
    n = len(diff)
    if n < 2:
        return None
    m = sum(diff) / n
    var = sum((x - m) ** 2 for x in diff) / (n - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    if sd == 0.0:
        return {"t": float("inf") if m != 0 else 0.0, "p": 0.0 if m != 0 else 1.0,
                "d": float("inf") if m != 0 else 0.0, "n": n, "mean_diff": m}
    t = m / (sd / math.sqrt(n))
    return {"t": t, "p": t_two_sided_p(t, n - 1), "d": m / sd, "n": n, "mean_diff": m}


def mcnemar(sa, sb):
    """McNemar's test (continuity-corrected) on aligned binary success lists."""
    b = sum(1 for x, y in zip(sa, sb) if x == 1 and y == 0)   # reference wins
    c = sum(1 for x, y in zip(sa, sb) if x == 0 and y == 1)   # other wins
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "chi2": 0.0, "p": 1.0}
    chi2 = (abs(b - c) - 1.0) ** 2 / n
    p = math.erfc(math.sqrt(chi2 / 2.0))                      # chi-square df=1 survival
    return {"b": b, "c": c, "chi2": chi2, "p": p}


# --------------------------------------------------------------------------- io
def _f(r, k, default=0.0):
    try:
        return float(r[k])
    except Exception:
        return default


def load(path):
    """scenario_id -> metrics dict, keeping the FIRST occurrence of each id (id >= 0)."""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                sid = int(float(r.get("scenario_id", -1)))
            except Exception:
                sid = -1
            if sid < 0 or sid in out:
                continue
            out[sid] = {
                "success": _f(r, "success"),
                "cte": _f(r, "mean_cte"),
                "slip": _f(r, "mean_slip"),
                "max_slip": _f(r, "max_slip"),
                "slope": _f(r, "terrain_avg_slope_deg"),
                "terr": _f(r, "terrain_intensity"),
                "progress": _f(r, "path_progress"),
                "ptype": (r.get("path_type") or "random").strip(),
            }
    return out


def pct(xs):
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _pstr(p):
    if p != p:
        return "  nan"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"


def main():
    raw = sys.argv[1:]
    if not raw:
        print(__doc__)
        sys.exit(1)

    datasets = []  # (label, dict)
    for a in raw:
        if "=" in a:
            label, path = a.split("=", 1)
        else:
            path = a
            base = os.path.basename(path)
            label = base.split("_")[0] if "_" in base else os.path.splitext(base)[0]
        if not os.path.isfile(path):
            print(f"!! not found: {path}")
            continue
        d = load(path)
        if not d:
            print(f"!! no scenario-tagged rows in {path} (was it produced by paired_eval.py?)")
            continue
        datasets.append((label, d))

    if len(datasets) < 2:
        print("need >=2 scenario-tagged eval CSVs to compare")
        sys.exit(1)

    labels = [d[0] for d in datasets]
    # matched scenario ids present in EVERY controller -> the paired sample
    common = set(datasets[0][1])
    for _, d in datasets[1:]:
        common &= set(d)
    ids = sorted(common)
    n = len(ids)
    if n == 0:
        print("no scenario_ids common to all files (nothing paired)")
        sys.exit(1)

    # reference controller for the pairwise tests
    ref_i = labels.index("hybrid") if "hybrid" in labels else 0
    ref_label, ref = datasets[ref_i]

    print("\n===================== PAIRED EVALUATION =====================")
    print(f"matched scenarios (present in all {len(datasets)} controllers): n = {n}")
    print(f"reference controller: {ref_label}\n")

    # ---------- overall matched table ----------
    print(f"  {'controller':<10} {'success%':>9} {'meanCTE':>9} {'meanSlip':>9} {'maxSlip':>9}")
    for label, d in datasets:
        succ = [d[i]["success"] for i in ids]
        cte = [d[i]["cte"] for i in ids]
        slip = [d[i]["slip"] for i in ids]
        mxs = [d[i]["max_slip"] for i in ids]
        print(f"  {label:<10} {pct(succ):>9.1f} {mean(cte):>9.4f} {mean(slip):>9.4f} {mean(mxs):>9.4f}")

    # ---------- pairwise paired tests (Bonferroni over the family) ----------
    others = [(lab, d) for j, (lab, d) in enumerate(datasets) if j != ref_i]
    m_comp = max(len(others), 1)   # comparisons per test family
    print(f"\n  PAIRED TESTS vs {ref_label}   (Bonferroni m = {m_comp})")
    print("  H1 tracking: MCTE paired t-test (negative mean-diff = reference is better)")
    print(f"    {'vs':<8} {'meanΔCTE':>9} {'t':>8} {'p':>7} {'p_bonf':>7} {'cohen_d':>8}")
    for lab, d in others:
        a = [ref[i]["cte"] for i in ids]
        b = [d[i]["cte"] for i in ids]
        r = paired_t(a, b)
        pb = min(1.0, r["p"] * m_comp)
        print(f"    {lab:<8} {r['mean_diff']:>9.4f} {r['t']:>8.2f} {_pstr(r['p']):>7} "
              f"{_pstr(pb):>7} {r['d']:>8.3f}")

    print("  H2 completion: McNemar on paired success (b = ref-only wins, c = other-only wins)")
    print(f"    {'vs':<8} {'b':>5} {'c':>5} {'chi2':>8} {'p':>7} {'p_bonf':>7}")
    for lab, d in others:
        sa = [int(ref[i]["success"]) for i in ids]
        sb = [int(d[i]["success"]) for i in ids]
        r = mcnemar(sa, sb)
        pb = min(1.0, r["p"] * m_comp)
        print(f"    {lab:<8} {r['b']:>5} {r['c']:>5} {r['chi2']:>8.2f} {_pstr(r['p']):>7} {_pstr(pb):>7}")

    # ---------- by path geometry (matched) ----------
    ptypes = sorted({ref[i]["ptype"] for i in ids})
    if ptypes and ptypes != ["random"]:
        print("\n  BY PATH GEOMETRY  (success% / meanCTE, matched scenarios)")
        head = "    " + f"{'geometry':<10}" + "".join(f"{lab:>16}" for lab in labels)
        print(head)
        for pt in ptypes:
            sub = [i for i in ids if ref[i]["ptype"] == pt]
            cells = []
            for _, d in datasets:
                s = pct([d[i]["success"] for i in sub])
                c = mean([d[i]["cte"] for i in sub])
                cells.append(f"{s:>7.1f}/{c:<8.3f}")
            print("    " + f"{pt:<10}" + "".join(f"{x:>16}" for x in cells))

    # ---------- by terrain-slope bin (matched; slope from the reference) ----------
    slopes = sorted(ref[i]["slope"] for i in ids)
    if slopes and slopes[-1] > slopes[0]:
        t1 = slopes[len(slopes) // 3]
        t2 = slopes[2 * len(slopes) // 3]

        def sbin(v):
            return "low" if v <= t1 else ("med" if v <= t2 else "high")

        print(f"\n  BY TERRAIN SLOPE  (deg tertiles: low<={t1:.1f}, med<={t2:.1f}, high>{t2:.1f})")
        head = "    " + f"{'slope':<10}" + "".join(f"{lab:>16}" for lab in labels)
        print(head)
        for band in ("low", "med", "high"):
            sub = [i for i in ids if sbin(ref[i]["slope"]) == band]
            if not sub:
                continue
            cells = []
            for _, d in datasets:
                s = pct([d[i]["success"] for i in sub])
                c = mean([d[i]["cte"] for i in sub])
                cells.append(f"{s:>7.1f}/{c:<8.3f}")
            print("    " + f"{band:<10}" + "".join(f"{x:>16}" for x in cells))

    print("\n=============================================================")
    print("success% / meanCTE cells are matched over the SAME scenarios, so differences are")
    print("attributable to the controller. p_bonf is the Bonferroni-adjusted p-value.\n")


if __name__ == "__main__":
    main()
