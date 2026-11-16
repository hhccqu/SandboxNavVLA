#!/usr/bin/env python3
"""TASK-20261112-003 / common.py

Shared constants and loaders for the N2-v02 -> H3pool-v02 two-stage run.

Paths are configured using environment variables.
"""
import json
import os

# --------------------------------------------------------------------- paths
ROOT = os.environ.get("T003V02_ROOT", "./data_root")
DATA_ROOT = os.environ.get("T003V02_DATA", "./data_root/raw")
EXPORT = os.environ.get("T003V02_EXPORT", "./data_root/training_export")

MANI = os.path.join(ROOT, "data/manifests/TASK-20261112-003")
SPLITS = os.path.join(ROOT, "data/splits/TASK-20261112-003")
CONF = os.path.join(ROOT, "configs/h3pool_v02")
EXP = os.path.join(ROOT, "experiments/TASK-20261112-003")

CMD_LABELS = os.path.join(MANI, "cmd_labels_v02.jsonl")
POOL = os.path.join(MANI, "pool_v02.jsonl")
SPLIT = os.path.join(SPLITS, "split_v02.json")
JUNCTIONS = os.path.join(CONF, "junctions_v02.json")

ARTIFACTS = os.environ.get("T003V02_ARTIFACTS", os.path.join(ROOT, "artifacts"))

# Configure the Hugging Face cache and checkpoint paths for your environment.
HF = os.environ.get("T003V02_HF", "./hf_cache")
BASE_CKPT = os.environ.get("T003V02_BASE", os.path.join(
    HF, "models--RenzKa--simlingo", "snapshots",
    "26c7c89e797d4e25bbf640013317af8da26a5454",
    "simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"))
N2_CKPT = os.environ.get("T003V02_N2", "./checkpoints/n2_final.pt")

os.environ.setdefault("HF_HOME", HF)
os.environ.setdefault("HF_HUB_CACHE", HF)


def sha256_file(path, chunk=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def json_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    return path


# ------------------------------------------------------------------ geometry
# Nominal training units. k is NOT a calibrated metre scale; never report these
# numbers as physical metres.
K = 0.0013263
N_ROUTE = 20
ROUTE_SPACING_RAW = 37.7          # legacy H3pool output definition, NOT 377
HORIZONS = [0.25 * (i + 1) for i in range(10)]

# -------------------------------------------------------------------- history
# History is matched by REAL SECONDS, never by frame index arithmetic.
TRAJ_LAGS_S = [0.25, 0.50, 0.75, 1.00, 1.25]
IMG_LAGS_S = [0.50, 1.00]
TRAJ_DT_TOL_S = 0.15
IMG_DT_TOL_S = 0.15
MAX_GAP_S = 0.60                  # a gap larger than this breaks any window

# ---------------------------------------------------------------------- splits
SPLIT_NAMES = ("train", "val", "test", "legacy_diag")


def load_split():
    with open(SPLIT) as f:
        return json.load(f)["splits"]


def load_cmd_labels():
    """{(run, idx): {stage, command, cot, speed_behavior, junction, direction}}"""
    out = {}
    with open(CMD_LABELS) as f:
        for line in f:
            o = json.loads(line)
            out[(o["run"], int(o["idx"]))] = o
    return out


def load_pool():
    """List of dicts, one per sample in the common pool."""
    with open(POOL) as f:
        return [json.loads(line) for line in f]


def load_valid():
    """{run: set(valid idx)} from the frozen filtered.csv."""
    import csv
    from collections import defaultdict
    per = defaultdict(set)
    with open(os.path.join(EXPORT, "filtered.csv")) as f:
        for r in csv.DictReader(f):
            if str(r["valid"]).strip() in ("1", "True", "true"):
                per[r["run"]].add(int(r["idx"]))
    return per


def pose_arrays(run):
    """t, x, y, heading, front-image path per frame index.

    Reads the raw collector log directly. Frame order == samples.jsonl line order,
    which is the same index space as filtered.csv `idx` and cmd_labels `idx`
    (verified by build_cmd_labels_v02 validation: 99.88 % agreement).
    """
    import numpy as np
    d = os.path.join(DATA_ROOT, run, "collector")
    ts, x, y, hd, img = [], [], [], [], []
    with open(os.path.join(d, "samples.jsonl")) as f:
        for line in f:
            o = json.loads(line)
            u = o.get("current_uwb_raw") or {}
            ts.append(float(o["capture_timestamp_unix_s"]))
            x.append(float(u.get("x", 0)))
            y.append(float(u.get("y", 0)))
            hd.append(float(u.get("heading", 0)))
            # The collector log stores an absolute driverhost path. Rebuild it
            # against DATA_ROOT so the same tree works on any host (kite mirror).
            _p = ((o.get("images") or {}).get("front") or {}).get("path")
            img.append(os.path.join(DATA_ROOT, run, "collector", "images",
                                    os.path.basename(_p)) if _p else None)
    ts = np.asarray(ts, dtype=np.float64)
    xy = np.stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)], axis=1)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    return {"t": ts, "xy": xy, "hd": np.asarray(hd, dtype=np.float64),
            "cum": cum, "img": img}


def causal_speed(pose, i, valid, lookback_s=1.0, max_gap_s=MAX_GAP_S):
    """Speed from past-only poses.

    Uses the same definition the old N2 labels used (displacement between the
    current pose and the pose one second earlier), but refuses to interpolate
    across a gap or through invalid frames. Returns (speed_raw, ok).
    """
    import numpy as np
    t, xy = pose["t"], pose["xy"]
    tq = t[i] - lookback_s
    if tq < t[0]:
        # not enough past: fall back to the longest usable past interval
        j = 0
        if t[i] - t[j] < 0.2:
            return 0.0, False
    else:
        j = int(np.searchsorted(t, tq, side="right") - 1)
        j = max(0, min(j, i - 1))
    if j >= i:
        return 0.0, False
    # every frame in (j, i] must be valid and the local gap must be acceptable
    if not all(valid[k] for k in range(j + 1, i + 1)):
        return 0.0, False
    if t[i] - t[j] > lookback_s + max_gap_s:
        return 0.0, False
    if j + 1 <= i and (t[j + 1] - t[j]) > max_gap_s:
        return 0.0, False
    # interpolate the position at tq between j and j+1 (both already validated)
    k = min(j + 1, i)
    if k == j or t[k] == t[j]:
        p = xy[j]
    else:
        f = (tq - t[j]) / (t[k] - t[j])
        p = xy[j] + f * (xy[k] - xy[j])
    return float(np.hypot(xy[i][0] - p[0], xy[i][1] - p[1])), True


def history_indices(pose, i, valid, lags, tol_s):
    """Resolve history observations by real seconds.

    For each target lag, the candidate is the frame whose timestamp is closest to
    t[i]-lag, but only if (a) it is strictly in the past, (b) the absolute timing
    error is within tol_s, (c) it is a valid frame, and (d) no gap or invalid
    frame lies between it and the current frame. No interpolation is ever used to
    manufacture a history observation.
    """
    import numpy as np
    t = pose["t"]
    idx, dts, ok = [], [], []
    for lag in lags:
        tq = t[i] - lag
        if tq < t[0]:
            idx.append(-1); dts.append(0.0); ok.append(False)
            continue
        j = int(np.searchsorted(t, tq, side="left"))
        # nearest of j-1, j
        cand = [c for c in (j - 1, j) if 0 <= c < i]
        if not cand:
            idx.append(-1); dts.append(0.0); ok.append(False)
            continue
        best = min(cand, key=lambda c: abs(t[c] - tq))
        err = abs(t[best] - tq)
        good = (err <= tol_s and i - best <= 12 and valid[best]
                and all(valid[k] for k in range(best + 1, i + 1))
                and (t[i] - t[best]) <= lag + tol_s + MAX_GAP_S)
        idx.append(best if good else -1)
        dts.append(float(t[i] - t[best]) if good else 0.0)
        ok.append(bool(good))
    return idx, dts, ok


def frame_masks(pose, i, valid, max_gap_s=MAX_GAP_S):
    """Per-point validity for the 10 time waypoints and the 20 route anchors.

    A point is valid only if its target time/arc is reachable inside this run AND
    the whole span from the current pose to it is gap-free and valid. Nothing is
    extrapolated or clipped-and-called-truth.
    """
    import numpy as np
    t, xy, cum = pose["t"], pose["xy"], pose["cum"]
    n = len(t)

    def span_clean(lo_i, hi_i):
        return all(valid[k] for k in range(lo_i, hi_i + 1))

    # ---- time waypoints
    wp_mask = np.zeros(10, dtype=np.float32)
    wp = np.zeros((10, 2), dtype=np.float64)
    for h_i, h in enumerate(HORIZONS):
        tq = t[i] + h
        if tq > t[n - 1]:
            continue
        k = int(np.searchsorted(t, tq, side="right") - 1)
        k = min(max(k, i), n - 1)
        lo, hi = min(k, k + 1), max(k, k + 1)
        if hi > n - 1:
            continue
        if not span_clean(i, hi):
            continue
        if t[hi] - t[lo] > max_gap_s:
            continue
        f = 0.0 if t[hi] == t[lo] else (tq - t[lo]) / (t[hi] - t[lo])
        wp[h_i] = xy[lo] + f * (xy[hi] - xy[lo])
        wp_mask[h_i] = 1.0

    # ---- geometric route anchors (arc length)
    route_mask = np.zeros(N_ROUTE, dtype=np.float32)
    rp = np.zeros((N_ROUTE, 2), dtype=np.float64)
    rp[0] = xy[i]
    route_mask[0] = 1.0
    for j in range(1, N_ROUTE):
        aq = cum[i] + j * ROUTE_SPACING_RAW
        if aq > cum[n - 1]:
            continue
        k = int(np.searchsorted(cum, aq, side="right") - 1)
        k = min(max(k, i), n - 2)
        lo, hi = max(k, i), k + 1
        if hi > n - 1:
            continue
        if not span_clean(i, hi):
            continue
        if t[hi] - t[lo] > max_gap_s:
            continue
        seg = cum[hi] - cum[lo]
        f = 0.0 if seg <= 1e-9 else (aq - cum[lo]) / seg
        rp[j] = xy[lo] + max(0.0, min(1.0, f)) * (xy[hi] - xy[lo])
        route_mask[j] = 1.0

    return wp, wp_mask, rp, route_mask
