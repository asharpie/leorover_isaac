#!/usr/bin/env python3
"""
Leo Mission Control — one-stop web dashboard for the leorover_isaac project.

Zero third-party dependencies: pure Python 3 stdlib. If pandas is installed it
is used to read big CSVs faster, but everything works without it.

Run it next to the repo (any machine — lab box, laptop, PC):

    python3 dashboard/app.py                 # http://localhost:8321
    python3 dashboard/app.py --port 9000
    python3 dashboard/app.py --repo /path/to/leorover_isaac

On the lab box, then from your laptop:

    ssh -L 8321:localhost:8321 irl@10.115.102.210
    -> open http://localhost:8321 in your browser.

The server binds 127.0.0.1 only (use the SSH tunnel; --host 0.0.0.0 to expose).
"""

import argparse
import base64
import csv as _csv
import glob
import html
import io
import json
import math
import mimetypes
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

try:
    import pandas as _pd  # optional accelerator
except Exception:
    _pd = None

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
HOME = os.path.expanduser("~")
LOGDIR = os.path.join(HOME, "leo_logs")

EXPERIMENTS = ["leo_rover_mars_hybrid", "leo_rover_mars", "leo_rover_flat"]
ALIAS2EXP = {"hybrid": "leo_rover_mars_hybrid", "ppo": "leo_rover_mars", "flat": "leo_rover_flat"}

# episode_metrics.csv schema (authoritative — recorder.py _HEADER)
EP_COLS = ["episode", "mean_cte", "max_cte", "total_reward", "mean_reward_per_step",
           "mean_slip", "steps", "success", "terrain_intensity", "friction_intensity",
           "terrain_max_slope_deg", "terrain_avg_slope_deg", "mean_local_slope_deg",
           "path_progress", "roll_max", "pitch_max", "mean_residual_v_norm",
           "mean_residual_w_norm", "max_slip", "scenario_id", "path_type"]
NUMERIC_EP_COLS = [c for c in EP_COLS if c != "path_type"]

_csv_cache = {}          # path -> (mtime, size, columns dict)
_csv_cache_lock = threading.Lock()
_MAX_CACHE = 6

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _safe(path):
    """Resolve a user-supplied path and require it to live under HOME or REPO."""
    if not path:
        raise ValueError("empty path")
    p = os.path.realpath(os.path.expanduser(path))
    for root in (os.path.realpath(HOME), os.path.realpath(REPO)):
        if p == root or p.startswith(root + os.sep):
            return p
    raise ValueError("path outside allowed roots: %s" % path)


def _rel(p):
    p = os.path.realpath(p)
    r = os.path.realpath(REPO)
    if p.startswith(r + os.sep):
        return os.path.relpath(p, r)
    return p


def _run(cmd, timeout=30, cwd=None, env=None):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             cwd=cwd or REPO, env=env)
        return out.returncode, out.stdout, out.stderr
    except FileNotFoundError:
        return 127, "", "not found: %s" % cmd[0]
    except subprocess.TimeoutExpired:
        return 124, "", "timed out"


def _mtime_str(p):
    try:
        return datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return "?"


def _fsize(p):
    try:
        n = os.path.getsize(p)
    except OSError:
        return "?"
    for u in ["B", "KB", "MB", "GB"]:
        if n < 1024 or u == "GB":
            return ("%.1f %s" % (n, u)) if u != "B" else ("%d B" % n)
        n /= 1024.0


# ----------------------------------------------------------------------------
# CSV engine
# ----------------------------------------------------------------------------

def load_csv(path, max_rows=None):
    """Load a CSV into {col: list}. Numeric columns become floats (NaN-safe).
    Cached by (mtime,size). Tolerates half-written trailing rows."""
    path = _safe(path)
    st = os.stat(path)
    key = path
    with _csv_cache_lock:
        hit = _csv_cache.get(key)
        if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
            return hit[2]

    cols = {}
    if _pd is not None:
        try:
            df = _pd.read_csv(path, on_bad_lines="skip", nrows=max_rows)
            for c in df.columns:
                if df[c].dtype == object:
                    cols[c] = df[c].astype(str).tolist()
                else:
                    cols[c] = df[c].astype(float).tolist()
        except Exception:
            cols = {}
    if not cols:
        with open(path, "r", newline="", errors="replace") as f:
            rd = _csv.reader(f)
            try:
                header = next(rd)
            except StopIteration:
                return {}
            data = [[] for _ in header]
            n = 0
            for row in rd:
                if len(row) != len(header):
                    continue  # half-written live row
                for i, v in enumerate(row):
                    data[i].append(v)
                n += 1
                if max_rows and n >= max_rows:
                    break
        for i, h in enumerate(header):
            col = data[i]
            # numeric?
            conv = []
            ok = True
            for v in col[:50]:
                try:
                    float(v)
                except ValueError:
                    ok = False
                    break
            if ok:
                for v in col:
                    try:
                        conv.append(float(v))
                    except ValueError:
                        conv.append(float("nan"))
                cols[h] = conv
            else:
                cols[h] = col

    with _csv_cache_lock:
        _csv_cache[key] = (st.st_mtime, st.st_size, cols)
        while len(_csv_cache) > _MAX_CACHE:
            _csv_cache.pop(next(iter(_csv_cache)))
    return cols


def _apply_filters(cols, filters):
    """filters: {col: [min,max]} or {col: ["=", value]} -> row index list."""
    if not filters:
        return None
    n = len(next(iter(cols.values()))) if cols else 0
    keep = [True] * n
    for c, rng in filters.items():
        if c not in cols:
            continue
        v = cols[c]
        if isinstance(rng, list) and len(rng) == 2 and rng[0] == "=":
            for i in range(n):
                if keep[i] and str(v[i]) != str(rng[1]):
                    keep[i] = False
        elif isinstance(rng, list) and len(rng) == 2:
            lo = -math.inf if rng[0] is None else float(rng[0])
            hi = math.inf if rng[1] is None else float(rng[1])
            for i in range(n):
                if keep[i]:
                    try:
                        x = float(v[i])
                    except (TypeError, ValueError):
                        keep[i] = False
                        continue
                    if not (lo <= x <= hi) or x != x:
                        keep[i] = False
    return [i for i, k in enumerate(keep) if k]


def _series_binned(cols, ycols, xcol, bins, idx=None):
    """Bin rows (in idx order) into `bins` buckets; per-bucket mean of each ycol."""
    n = len(cols.get(xcol, [])) if xcol in cols else (len(idx) if idx else 0)
    rows = idx if idx is not None else list(range(len(next(iter(cols.values()), []))))
    n = len(rows)
    if n == 0:
        return {"x": [], "series": {c: [] for c in ycols}, "count": []}
    bins = max(1, min(bins, n))
    per = n / bins
    out_x, counts = [], []
    sums = {c: [] for c in ycols}
    for b in range(bins):
        lo, hi = int(b * per), int((b + 1) * per)
        if hi <= lo:
            continue
        seg = rows[lo:hi]
        cnt = 0
        acc = {c: 0.0 for c in ycols}
        accn = {c: 0 for c in ycols}
        xv = 0.0
        for i in seg:
            cnt += 1
            if xcol in cols:
                try:
                    xv = float(cols[xcol][i])
                except (TypeError, ValueError):
                    pass
            for c in ycols:
                try:
                    v = float(cols[c][i])
                except (TypeError, ValueError, KeyError):
                    continue
                if v == v:
                    acc[c] += v
                    accn[c] += 1
        out_x.append(xv if xcol in cols else hi)
        counts.append(cnt)
        for c in ycols:
            sums[c].append(acc[c] / accn[c] if accn[c] else None)
    return {"x": out_x, "series": sums, "count": counts}


def _hist(vals, nbins=40):
    xs = [v for v in vals if isinstance(v, float) and v == v]
    if not xs:
        return {"edges": [], "counts": []}
    lo, hi = min(xs), max(xs)
    if lo == hi:
        hi = lo + 1.0
    w = (hi - lo) / nbins
    counts = [0] * nbins
    for v in xs:
        b = int((v - lo) / w)
        if b == nbins:
            b -= 1
        counts[b] += 1
    edges = [lo + w * i for i in range(nbins + 1)]
    return {"edges": edges, "counts": counts}


# ----------------------------------------------------------------------------
# domain scans
# ----------------------------------------------------------------------------

def scan_runs():
    out = []
    for exp in EXPERIMENTS:
        base = os.path.join(REPO, "logs", exp)
        if not os.path.isdir(base):
            continue
        for d in sorted(glob.glob(os.path.join(base, "*/")), key=os.path.getmtime, reverse=True):
            d = d.rstrip("/")
            ckpts = sorted(glob.glob(os.path.join(d, "model_*.pt")), key=os.path.getmtime)
            csvp = os.path.join(d, "csv", "episode_metrics.csv")
            art = {}
            for k, rp in [("trace", "eval_trace/trace.png"), ("stall_diag", "stall_diag/stall_diag.png"),
                          ("stall_viz", "stall_viz/stall_viz.png")]:
                fp = os.path.join(d, rp)
                if os.path.isfile(fp):
                    art[k] = _rel(fp)
            tb = bool(glob.glob(os.path.join(d, "events.out.tfevents.*")))
            out.append({
                "experiment": exp, "path": _rel(d), "name": os.path.basename(d),
                "mtime": _mtime_str(d),
                "checkpoints": [{"name": os.path.basename(c), "mtime": _mtime_str(c),
                                 "size": _fsize(c)} for c in reversed(ckpts)],
                "csv": _rel(csvp) if os.path.isfile(csvp) else None,
                "csv_size": _fsize(csvp) if os.path.isfile(csvp) else None,
                "artifacts": art, "tensorboard": tb,
            })
    return out


def scan_evals():
    ev = {"paired": [], "multiworld": [], "quick": [], "demos": [], "collapse": [], "other_csv": []}
    base = os.path.join(REPO, "evals")
    pd_ = os.path.join(base, "paired")
    if os.path.isdir(pd_):
        for d in sorted(glob.glob(os.path.join(pd_, "*/")), key=os.path.getmtime, reverse=True):
            d = d.rstrip("/")
            files = {os.path.basename(p): _rel(p) for p in glob.glob(os.path.join(d, "*"))}
            ctrls = [c for c in ("hybrid", "lqr", "ppo") if (c + ".csv") in files]
            ev["paired"].append({"path": _rel(d), "name": os.path.basename(d),
                                 "mtime": _mtime_str(d), "controllers": ctrls,
                                 "stats": files.get("stats.txt"), "files": files})
    mw = os.path.join(base, "multiworld")
    if os.path.isdir(mw):
        for d in sorted(glob.glob(os.path.join(mw, "*/")), key=os.path.getmtime, reverse=True):
            d = d.rstrip("/")
            worlds = sorted(glob.glob(os.path.join(d, "world_*")))
            st = os.path.join(d, "stats_multiworld.txt")
            ev["multiworld"].append({"path": _rel(d), "name": os.path.basename(d),
                                     "mtime": _mtime_str(d), "worlds": len(worlds),
                                     "stats": _rel(st) if os.path.isfile(st) else None})
    if os.path.isdir(base):
        for p in sorted(glob.glob(os.path.join(base, "*_*.csv")), key=os.path.getmtime, reverse=True):
            ev["quick"].append({"path": _rel(p), "name": os.path.basename(p),
                                "mtime": _mtime_str(p), "size": _fsize(p)})
        for p in sorted(glob.glob(os.path.join(base, "demo_*.html")), key=os.path.getmtime, reverse=True):
            ev["demos"].append({"path": _rel(p), "name": os.path.basename(p), "mtime": _mtime_str(p)})
        for p in sorted(glob.glob(os.path.join(base, "collapse_manifest_*.txt")),
                        key=os.path.getmtime, reverse=True):
            ev["collapse"].append({"path": _rel(p), "name": os.path.basename(p), "mtime": _mtime_str(p)})
    # loose CSVs the user might drop anywhere common
    for extra in [os.path.join(REPO, "box_evals"), os.path.join(HOME, "Downloads", "leo_csvs")]:
        if os.path.isdir(extra):
            for p in sorted(glob.glob(os.path.join(extra, "*.csv"))):
                ev["other_csv"].append({"path": _rel(p), "name": os.path.basename(p),
                                        "mtime": _mtime_str(p), "size": _fsize(p)})
    root_csv = os.path.join(REPO, "episode_metrics.csv")
    if os.path.isfile(root_csv):
        ev["other_csv"].insert(0, {"path": _rel(root_csv), "name": "episode_metrics.csv (repo root)",
                                   "mtime": _mtime_str(root_csv), "size": _fsize(root_csv)})
    return ev


def gpu_status():
    rc, out, _ = _run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                       "--format=csv,noheader,nounits"], timeout=8)
    gpu = None
    if rc == 0 and out.strip():
        try:
            used, total, util = [float(x) for x in out.strip().splitlines()[0].split(",")]
            gpu = {"mem_used": used, "mem_total": total, "util": util}
        except ValueError:
            pass
    procs = []
    rc2, out2, _ = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                         "--format=csv,noheader"], timeout=8)
    if rc2 == 0:
        for ln in out2.strip().splitlines():
            if ln.strip():
                procs.append(ln.strip())
    return gpu, procs


def training_procs():
    rc, out, _ = _run(["bash", "-c",
                       "ps -o pid=,etime=,cmd= -u \"$USER\" | grep -E 'scripts/(train|paired_eval|evaluate_policy|trace_episode|diagnose_stalls|record_demo)\\.py' | grep -v grep"],
                      timeout=8)
    procs = []
    if rc == 0:
        for ln in out.strip().splitlines():
            parts = ln.strip().split(None, 2)
            if len(parts) == 3:
                procs.append({"pid": parts[0], "etime": parts[1], "cmd": parts[2][:220]})
    return procs


LOG_MARKERS = {
    "success": re.compile(r"success=([0-9.]+)%"),
    "reward": re.compile(r"Mean reward:\s*(-?[0-9.]+)"),
    "std": re.compile(r"Mean action noise std:\s*([0-9.]+)"),
    "iter": re.compile(r"[Ll]earning iteration\s+(\d+)"),
    "adr": re.compile(r"\[ADR-eval\] iter (\d+): det success ([0-9.]+)%, CTE ([0-9.]+)"),
}


def parse_log(path, tail_bytes=2_000_000):
    path = _safe(path)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size > tail_bytes:
            f.seek(size - tail_bytes)
        raw = f.read().decode("utf-8", errors="replace")
    succ, rew, std, adr = [], [], [], []
    cur_iter = 0
    for ln in raw.splitlines():
        m = LOG_MARKERS["iter"].search(ln)
        if m:
            cur_iter = int(m.group(1))
        m = LOG_MARKERS["success"].search(ln)
        if m:
            succ.append([cur_iter, float(m.group(1))])
        m = LOG_MARKERS["reward"].search(ln)
        if m:
            rew.append([cur_iter, float(m.group(1))])
        m = LOG_MARKERS["std"].search(ln)
        if m:
            std.append([cur_iter, float(m.group(1))])
        m = LOG_MARKERS["adr"].search(ln)
        if m:
            adr.append([int(m.group(1)), float(m.group(2)), float(m.group(3))])
    return {"success": succ[-400:], "reward": rew[-400:], "std": std[-400:], "adr": adr[-200:],
            "tail": "\n".join(raw.splitlines()[-80:]), "size": size, "truncated": size > tail_bytes}


# ----------------------------------------------------------------------------
# paired / multiworld / collapse plot data (NEW figures the repo lacks)
# ----------------------------------------------------------------------------

def _paired_join(dirpath, min_sid=0):
    dirpath = _safe(dirpath)
    ctrls = {}
    for c in ("hybrid", "lqr", "ppo"):
        p = os.path.join(dirpath, c + ".csv")
        if os.path.isfile(p):
            ctrls[c] = load_csv(p)
    if not ctrls:
        raise ValueError("no controller CSVs in " + dirpath)
    keyed = {}
    for c, cols in ctrls.items():
        sid = cols.get("scenario_id", [])
        d = {}
        for i, s in enumerate(sid):
            try:
                s = int(float(s))
            except (TypeError, ValueError):
                continue
            if s >= min_sid and s not in d:
                d[s] = i
        keyed[c] = d
    common = None
    for c, d in keyed.items():
        s = set(d.keys())
        common = s if common is None else (common & s)
    common = sorted(common or [])
    return ctrls, keyed, common


def paired_plotdata(dirpath):
    ctrls, keyed, common = _paired_join(dirpath)
    out = {"n": len(common), "controllers": list(ctrls.keys()), "by_level": {}, "by_geom": {},
           "delta_hist": None, "overall": {}}
    lv_of = {}
    geom_of = {}
    ref = "hybrid" if "hybrid" in ctrls else list(ctrls.keys())[0]
    rc = ctrls[ref]
    for s in common:
        i = keyed[ref][s]
        try:
            lv_of[s] = round(float(rc["terrain_intensity"][i]))
        except (TypeError, ValueError, KeyError):
            lv_of[s] = -1
        geom_of[s] = str(rc.get("path_type", ["?"] * (i + 1))[i])
    if len(set(lv_of.values())) > 13:  # continuous terrain values -> bin to nearest 10
        lv_of = {s: int(round(v / 10.0) * 10) for s, v in lv_of.items()}
    for c, cols in ctrls.items():
        succ_l, cte_l = {}, {}
        succ_g, cte_g = {}, {}
        tot_s = 0
        tot_c, tot_cn = 0.0, 0
        for s in common:
            i = keyed[c][s]
            try:
                sc = float(cols["success"][i])
                ct = float(cols["mean_cte"][i])
            except (TypeError, ValueError, KeyError):
                continue
            lv, gm = lv_of[s], geom_of[s]
            succ_l.setdefault(lv, [0, 0]); cte_l.setdefault(lv, [0.0, 0])
            succ_l[lv][0] += sc; succ_l[lv][1] += 1
            if ct == ct:
                cte_l[lv][0] += ct; cte_l[lv][1] += 1
                tot_c += ct; tot_cn += 1
            succ_g.setdefault(gm, [0, 0]); cte_g.setdefault(gm, [0.0, 0])
            succ_g[gm][0] += sc; succ_g[gm][1] += 1
            if ct == ct:
                cte_g[gm][0] += ct; cte_g[gm][1] += 1
            tot_s += sc
        out["by_level"][c] = {
            "levels": sorted(succ_l.keys()),
            "success": [100.0 * succ_l[k][0] / succ_l[k][1] for k in sorted(succ_l.keys())],
            "cte": [cte_l[k][0] / max(cte_l[k][1], 1) for k in sorted(cte_l.keys())],
        }
        out["by_geom"][c] = {
            "geoms": sorted(succ_g.keys()),
            "success": [100.0 * succ_g[k][0] / succ_g[k][1] for k in sorted(succ_g.keys())],
            "cte": [cte_g[k][0] / max(cte_g[k][1], 1) for k in sorted(cte_g.keys())],
        }
        out["overall"][c] = {"success": 100.0 * tot_s / max(len(common), 1),
                             "cte": tot_c / max(tot_cn, 1)}
    if "hybrid" in ctrls and "lqr" in ctrls:
        deltas = []
        for s in common:
            try:
                h = float(ctrls["hybrid"]["mean_cte"][keyed["hybrid"][s]])
                l = float(ctrls["lqr"]["mean_cte"][keyed["lqr"][s]])
            except (TypeError, ValueError, KeyError):
                continue
            if h == h and l == l:
                deltas.append(h - l)
        out["delta_hist"] = _hist(deltas, 60)
        out["delta_mean"] = sum(deltas) / max(len(deltas), 1)
    return out


def multiworld_plotdata(dirpath):
    dirpath = _safe(dirpath)
    worlds = []
    for wd in sorted(glob.glob(os.path.join(dirpath, "world_*"))):
        seed = os.path.basename(wd).split("_", 1)[-1]
        try:
            ctrls, keyed, common = _paired_join(wd)
        except ValueError:
            continue
        if "hybrid" not in ctrls or "lqr" not in ctrls or len(common) < 30:
            continue
        dc, sh, sl = [], 0.0, 0.0
        ch, cl = [], []
        for s in common:
            try:
                h = float(ctrls["hybrid"]["mean_cte"][keyed["hybrid"][s]])
                l = float(ctrls["lqr"]["mean_cte"][keyed["lqr"][s]])
                shx = float(ctrls["hybrid"]["success"][keyed["hybrid"][s]])
                slx = float(ctrls["lqr"]["success"][keyed["lqr"][s]])
            except (TypeError, ValueError, KeyError):
                continue
            if h == h and l == l:
                dc.append(h - l); ch.append(h); cl.append(l)
            sh += shx; sl += slx
        n = len(dc)
        if n < 30:
            continue
        mean = sum(dc) / n
        var = sum((x - mean) ** 2 for x in dc) / max(n - 1, 1)
        se = math.sqrt(var / n)
        worlds.append({"seed": seed, "n": n, "dcte": mean, "ci": 1.96 * se,
                       "succ_h": 100.0 * sh / len(common), "succ_l": 100.0 * sl / len(common),
                       "cte_h": sum(ch) / n, "cte_l": sum(cl) / n})
    return {"worlds": worlds}


def collapse_plotdata(manifest):
    manifest = _safe(manifest)
    rows = []
    with open(manifest, errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 2:
                continue
            H, d = parts[0], parts[1]
            if d == "FAILED":
                continue
            try:
                pdat = paired_plotdata(d if os.path.isabs(d) else os.path.join(REPO, d))
            except (ValueError, OSError):
                continue
            row = {"H": float(H)}
            for c, o in pdat["overall"].items():
                row[c + "_succ"] = o["success"]
                row[c + "_cte"] = o["cte"]
            rows.append(row)
    rows.sort(key=lambda r: r["H"])
    return {"rows": rows}


# ----------------------------------------------------------------------------
# launching
# ----------------------------------------------------------------------------

def can_launch():
    return os.path.isfile(os.path.join(REPO, "scripts", "run_lab.sh")) and \
        shutil.which("nvidia-smi") is not None


def build_command(kind, opts):
    """Return (argv-as-shell-string, env_extra) mirroring leo.sh semantics."""
    env = {}
    for k, v in (opts.get("env") or {}).items():
        if re.fullmatch(r"[A-Z0-9_]+", k) and str(v) != "":
            env[k] = str(v)
    leo = "bash scripts/leo.sh"

    def flag(name, key, cast=str):
        v = opts.get(key)
        return " %s %s" % (name, shlex.quote(str(cast(v)))) if v not in (None, "", False) else ""

    if kind == "train":
        alias = opts.get("alias", "hybrid")
        if alias not in ("hybrid", "ppo", "flat"):
            raise ValueError("bad alias")
        cmd = "%s train %s" % (leo, alias)
        cmd += flag("--envs", "envs") + flag("--iters", "iters") + flag("--ent", "ent")
        cmd += flag("--rollout", "rollout") + flag("--speed", "speed") + flag("--residual", "residual")
        if opts.get("raw"):
            cmd += " --raw"
        return cmd, env, "train"
    if kind == "paired":
        cmd = "%s eval" % leo
        cmd += flag("--paths", "paths") + flag("--friction", "friction") + flag("--n", "n")
        cmd += flag("--levels", "levels") + flag("--envs", "envs")
        return cmd, env, "bg"
    if kind == "multieval":
        cmd = "%s multieval" % leo
        for name, key in [("--worlds", "worlds"), ("--seedbase", "seedbase"), ("--seeds", "seeds"),
                          ("--n", "n"), ("--paths", "paths"), ("--friction", "friction"),
                          ("--pathbank", "pathbank"), ("--levels", "levels"), ("--envs", "envs")]:
            cmd += flag(name, key)
        return cmd, env, "bg"
    if kind == "quickeval":
        alias = opts.get("alias", "hybrid")
        if alias not in ("hybrid", "lqr", "ppo"):
            raise ValueError("bad alias")
        cmd = "%s quickeval %s" % (leo, alias)
        cmd += flag("--levels", "levels") + flag("--envs", "envs") + flag("--steps", "steps")
        return cmd, env, "bg"
    if kind == "trace":
        cmd = "%s trace" % leo
        if opts.get("model"):
            cmd += " " + shlex.quote(str(opts["model"]))
        cmd += flag("--envs", "envs") + flag("--steps", "steps")
        if opts.get("lqr"):
            cmd += " --lqr"
        return cmd, env, "bg"
    if kind == "diagnose":
        cmd = "%s diagnose" % leo
        if opts.get("model"):
            cmd += " " + shlex.quote(str(opts["model"]))
        cmd += flag("--terrain", "terrain")
        if opts.get("lqr"):
            cmd += " --lqr"
        return cmd, env, "bg"
    if kind == "record":
        mode = opts.get("mode", "pair")
        cmd = "%s record %s" % (leo, shlex.quote(mode))
        cmd += flag("--num", "num") + flag("--level", "level") + flag("--friction", "friction")
        cmd += flag("--seed", "seed") + flag("--ckpt", "ckpt")
        return cmd, env, "bg"
    if kind == "custom":
        c = str(opts.get("cmd", "")).strip()
        if not c:
            raise ValueError("empty command")
        return c, env, "bg"
    raise ValueError("unknown kind " + str(kind))


def launch(kind, opts):
    cmd, env, mode = build_command(kind, opts)
    full_env = dict(os.environ)
    full_env.update(env)
    os.makedirs(LOGDIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logp = os.path.join(LOGDIR, "gui_%s_%s.log" % (kind, stamp))
    envprefix = " ".join("%s=%s" % (k, shlex.quote(v)) for k, v in env.items())
    shown = (envprefix + " " + cmd).strip()
    if not opts.get("really"):
        return {"command": shown, "launched": False}
    if not can_launch():
        return {"command": shown, "launched": False,
                "error": "This machine has no run_lab.sh + GPU. Copy the command and run it on the lab box."}
    with open(logp, "w") as lf:
        lf.write("$ %s\n\n" % shown)
        lf.flush()
        subprocess.Popen(["bash", "-c", cmd], cwd=REPO, env=full_env,
                         stdout=lf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    return {"command": shown, "launched": True, "log": logp}


def stop_training(pid=None):
    if pid:
        rc, out, _ = _run(["bash", "-c", "ps -o cmd= -p %d" % int(pid)], timeout=5)
        if "scripts/" not in out:
            return {"ok": False, "error": "PID %s is not a leorover process" % pid}
        _run(["kill", "-9", str(int(pid))], timeout=5)
        return {"ok": True, "stopped": [str(pid)]}
    rc, out, _ = _run(["bash", "-c", "pgrep -u \"$USER\" -f 'scripts/train.py'"], timeout=5)
    pids = [p for p in out.split() if p.strip()]
    if pids:
        _run(["bash", "-c", "pkill -9 -u \"$USER\" -f 'scripts/train.py'"], timeout=8)
    return {"ok": True, "stopped": pids}


# ----------------------------------------------------------------------------
# reports (reuse the repo's stdlib analysis scripts)
# ----------------------------------------------------------------------------

def run_report(kind, args):
    py = sys.executable or "python3"
    if kind == "leo_report":
        cmd = [py, os.path.join(REPO, "scripts", "leo_report.py")]
        if args.get("csv"):
            cmd.append(_safe(args["csv"]))
    elif kind == "eval_report":
        cmd = [py, os.path.join(REPO, "scripts", "eval_report.py")]
        for lbl, p in (args.get("files") or {}).items():
            cmd.append("%s=%s" % (lbl, _safe(p)))
    elif kind == "paired_stats":
        d = _safe(args["dir"])
        cmd = [py, os.path.join(REPO, "scripts", "paired_stats.py")]
        for c in ("hybrid", "lqr", "ppo"):
            p = os.path.join(d, c + ".csv")
            if os.path.isfile(p):
                cmd.append("%s=%s" % (c, p))
    elif kind == "multiworld_stats":
        cmd = [py, os.path.join(REPO, "scripts", "multiworld_stats.py"), _safe(args["dir"])]
    elif kind == "collapse_curve":
        cmd = [py, os.path.join(REPO, "scripts", "collapse_curve.py"), _safe(args["manifest"])]
    else:
        raise ValueError("unknown report " + kind)
    rc, out, err = _run(cmd, timeout=180)
    return {"rc": rc, "out": out, "err": err, "cmd": " ".join(cmd)}


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "LeoMissionControl/1.0"

    def log_message(self, fmt, *a):
        pass  # quiet

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, msg, code=400):
        self._json({"error": str(msg)}, code)

    def _file(self, path, download=False):
        try:
            path = _safe(path)
            with open(path, "rb") as f:
                data = f.read()
        except (OSError, ValueError) as e:
            return self._err(e, 404)
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith((".log", ".txt")):
            mime = "text/plain; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        if download:
            self.send_header("Content-Disposition",
                             'attachment; filename="%s"' % os.path.basename(path))
        self.end_headers()
        self.wfile.write(data)

    # -------------------- GET --------------------
    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        p = u.path
        try:
            if p == "/" or p == "/index.html":
                return self._file(os.path.join(HERE, "static", "index.html"))
            if p.startswith("/static/"):
                return self._file(os.path.join(HERE, p.lstrip("/")))
            if p == "/api/status":
                gpu, gprocs = gpu_status()
                logs = sorted(glob.glob(os.path.join(LOGDIR, "*.log")),
                              key=os.path.getmtime, reverse=True)[:12]
                return self._json({
                    "host": os.uname().nodename, "repo": REPO, "home": HOME,
                    "can_launch": can_launch(), "gpu": gpu, "gpu_procs": gprocs,
                    "training": training_procs(),
                    "logs": [{"path": l, "name": os.path.basename(l), "mtime": _mtime_str(l),
                              "size": _fsize(l)} for l in logs],
                    "pandas": _pd is not None,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
            if p == "/api/runs":
                return self._json({"runs": scan_runs()})
            if p == "/api/evals":
                return self._json(scan_evals())
            if p == "/api/log":
                return self._json(parse_log(q["path"]))
            if p == "/api/tree":
                root = _safe(q.get("path") or REPO)
                if not os.path.isdir(root):
                    return self._err("not a directory", 404)
                items = []
                try:
                    names = sorted(os.listdir(root))
                except OSError as e:
                    return self._err(e, 403)
                for nm in names:
                    if nm in (".git", "__pycache__", ".venv", ".idea"):
                        continue
                    fp = os.path.join(root, nm)
                    isdir = os.path.isdir(fp)
                    items.append({"name": nm, "dir": isdir, "path": fp,
                                  "size": None if isdir else _fsize(fp),
                                  "mtime": _mtime_str(fp)})
                items.sort(key=lambda x: (not x["dir"], x["name"].lower()))
                return self._json({"path": root, "parent": os.path.dirname(root), "items": items})
            if p == "/api/file":
                return self._file(q["path"], download=q.get("dl") == "1")
            if p == "/api/csv/meta":
                cols = load_csv(q["path"])
                n = len(next(iter(cols.values()))) if cols else 0
                info = []
                for c, v in cols.items():
                    numeric = bool(v) and isinstance(v[0], float)
                    ent = {"name": c, "numeric": numeric}
                    if numeric:
                        xs = [x for x in v if x == x]
                        if xs:
                            ent.update({"min": min(xs), "max": max(xs),
                                        "mean": sum(xs) / len(xs)})
                    else:
                        vals = {}
                        for x in v:
                            vals[x] = vals.get(x, 0) + 1
                            if len(vals) > 24:
                                break
                        ent["values"] = sorted(vals.keys())[:24]
                    info.append(ent)
                return self._json({"rows": n, "columns": info, "path": q["path"]})
            if p == "/api/csv/series":
                cols = load_csv(q["path"])
                filters = json.loads(q.get("filters") or "{}")
                idx = _apply_filters(cols, filters)
                ycols = [c for c in (q.get("y") or "").split(",") if c]
                xcol = q.get("x") or "episode"
                bins = int(q.get("bins") or 400)
                if idx is None:
                    idx = list(range(len(next(iter(cols.values()), []))))
                return self._json(_series_binned(cols, ycols, xcol, bins, idx))
            if p == "/api/csv/hist":
                cols = load_csv(q["path"])
                filters = json.loads(q.get("filters") or "{}")
                idx = _apply_filters(cols, filters)
                v = cols.get(q.get("col"), [])
                if idx is not None:
                    v = [v[i] for i in idx]
                return self._json(_hist([x for x in v if isinstance(x, float)],
                                        int(q.get("bins") or 40)))
            if p == "/api/csv/scatter":
                cols = load_csv(q["path"])
                filters = json.loads(q.get("filters") or "{}")
                idx = _apply_filters(cols, filters)
                xs, ys = cols.get(q.get("x"), []), cols.get(q.get("y"), [])
                rows = idx if idx is not None else list(range(min(len(xs), len(ys))))
                cap = int(q.get("cap") or 4000)
                step = max(1, len(rows) // cap)
                pts = [[xs[i], ys[i]] for i in rows[::step]
                       if isinstance(xs[i], float) and isinstance(ys[i], float)
                       and xs[i] == xs[i] and ys[i] == ys[i]]
                return self._json({"points": pts, "total": len(rows), "shown": len(pts)})
            if p == "/api/csv/group":
                cols = load_csv(q["path"])
                filters = json.loads(q.get("filters") or "{}")
                idx = _apply_filters(cols, filters)
                by, val = q.get("by"), q.get("val")
                gv = cols.get(by, [])
                vv = cols.get(val, [])
                rows = idx if idx is not None else list(range(min(len(gv), len(vv))))
                nb = int(q.get("nbins") or 8)
                groups = {}
                numeric_by = bool(gv) and isinstance(gv[0], float)
                if numeric_by:
                    xs = [gv[i] for i in rows if gv[i] == gv[i]]
                    lo, hi = (min(xs), max(xs)) if xs else (0, 1)
                    w = (hi - lo) / nb or 1.0
                    for i in rows:
                        g = gv[i]
                        if not isinstance(g, float) or g != g:
                            continue
                        b = min(int((g - lo) / w), nb - 1)
                        lbl = "%.3g–%.3g" % (lo + b * w, lo + (b + 1) * w)
                        groups.setdefault((b, lbl), []).append(vv[i])
                else:
                    for i in rows:
                        groups.setdefault((0, str(gv[i])), []).append(vv[i])
                out = []
                for (b, lbl), vs in sorted(groups.items()):
                    xs = [x for x in vs if isinstance(x, float) and x == x]
                    if not xs:
                        continue
                    m = sum(xs) / len(xs)
                    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1))
                    out.append({"group": lbl, "n": len(xs), "mean": m, "std": sd,
                                "min": min(xs), "max": max(xs)})
                return self._json({"groups": out})
            if p == "/api/csv/table":
                cols = load_csv(q["path"])
                filters = json.loads(q.get("filters") or "{}")
                idx = _apply_filters(cols, filters)
                names = list(cols.keys())
                rows_all = idx if idx is not None else list(range(len(next(iter(cols.values()), []))))
                off, lim = int(q.get("offset") or 0), min(int(q.get("limit") or 100), 1000)
                sel = rows_all[off:off + lim]
                rows = [[cols[c][i] for c in names] for i in sel]
                return self._json({"columns": names, "rows": rows, "total": len(rows_all),
                                   "offset": off})
            if p == "/api/paired":
                return self._json(paired_plotdata(os.path.join(REPO, q["dir"])
                                                  if not os.path.isabs(q["dir"]) else q["dir"]))
            if p == "/api/multiworld":
                return self._json(multiworld_plotdata(os.path.join(REPO, q["dir"])
                                                      if not os.path.isabs(q["dir"]) else q["dir"]))
            if p == "/api/collapse":
                return self._json(collapse_plotdata(os.path.join(REPO, q["manifest"])
                                                    if not os.path.isabs(q["manifest"]) else q["manifest"]))
            return self._err("unknown endpoint " + p, 404)
        except (ValueError, KeyError, OSError) as e:
            return self._err(e)
        except Exception as e:  # pragma: no cover
            return self._err("%s: %s" % (type(e).__name__, e), 500)

    # -------------------- POST --------------------
    def do_POST(self):
        u = urlparse(self.path)
        try:
            ln = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(ln) or b"{}")
        except (ValueError, TypeError):
            return self._err("bad JSON body")
        try:
            if u.path == "/api/launch":
                return self._json(launch(body.get("kind"), body))
            if u.path == "/api/stop":
                return self._json(stop_training(body.get("pid")))
            if u.path == "/api/report":
                return self._json(run_report(body.get("kind"), body))
            return self._err("unknown endpoint " + u.path, 404)
        except (ValueError, KeyError, OSError) as e:
            return self._err(e)
        except Exception as e:  # pragma: no cover
            return self._err("%s: %s" % (type(e).__name__, e), 500)


def main():
    global REPO, LOGDIR
    ap = argparse.ArgumentParser(description="Leo Mission Control dashboard")
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--logdir", default=LOGDIR)
    a = ap.parse_args()
    REPO = os.path.abspath(a.repo)
    LOGDIR = os.path.abspath(os.path.expanduser(a.logdir))
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print("Leo Mission Control")
    print("  repo   : %s" % REPO)
    print("  logs   : %s" % LOGDIR)
    print("  launch : %s" % ("ENABLED (GPU + run_lab.sh found)" if can_launch()
                             else "view-only on this machine (no GPU/run_lab.sh)"))
    print("  open   : http://localhost:%d" % a.port)
    print("  remote : ssh -L %d:localhost:%d irl@10.115.102.210   (then open the same URL)"
          % (a.port, a.port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
