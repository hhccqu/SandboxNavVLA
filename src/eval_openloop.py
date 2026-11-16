#!/usr/bin/env python3
"""TASK-20261112-003 / eval_openloop.py

Open-loop evaluation of one checkpoint on one split, with per-point masks.

Definitions (fixed before any test evaluation):
  route ADE   mean over VALID anchors j>=1 of ||pred_j - gt_j||, origin excluded
  route FDE   ||pred_k - gt_k|| at the LAST valid anchor k
  wps ADE     mean over valid time waypoints of ||pred - gt||
  wps FDE     ||pred - gt|| at the last valid time waypoint
All distances are in the legacy nominal unit (k = 0.0013263, NOT a calibrated
metre). Masked points never enter a metric.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "vendor")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
import engine as E  # noqa: E402
from collate_v02 import make_collate  # noqa: E402
from dataset_v02 import SandtableV02Pool  # noqa: E402


def summarize(pred, gt, mask):
    """pred/gt: (N,K,2) numpy; mask: (N,K). Returns metric dict."""
    d = np.linalg.norm(pred - gt, axis=-1)          # (N,K)
    m = mask > 0
    n_pts = int(m.sum())
    ade = float(d[m].mean()) if n_pts else float("nan")
    fde_vals = []
    for i in range(d.shape[0]):
        idx = np.where(m[i])[0]
        if len(idx):
            fde_vals.append(d[i, idx[-1]])
    fde = float(np.mean(fde_vals)) if fde_vals else float("nan")
    return {"ade": ade, "fde": fde, "n_points": n_pts,
            "n_samples": int((m.any(1)).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stage", type=int, required=True, choices=(1, 2))
    ap.add_argument("--split", required=True,
                    help="val|test|legacy_diag, or comma separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with_history = args.stage == 2
    cfg = E.build_cfg(with_history)
    proc = E.build_processor(cfg)
    dm = E.build_datamodule(cfg, proc)
    model = E.build_model(args.stage, cfg, proc, dm, init_ckpt=args.ckpt, verbose=True)
    model.to("cuda").eval()
    collate = make_collate(dm, use_img=with_history)

    splits = tuple(s.strip() for s in args.split.split(","))
    ds = SandtableV02Pool(splits=splits, use_history=with_history,
                          limit=(args.limit or None))

    rows = []
    P, G, M, PW, GW, MW = [], [], [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for k in range(len(ds)):
            s = ds[k]
            ex, hist, rm, wm, _ = collate([s])
            ex = E.move_batch(ex)
            rm = rm.cuda(); wm = wm.cuda()
            hist = E.hist_to_cuda(hist, use_img=with_history)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, ld = E.compute_total(model, ex, hist, rm, wm)
            pr = ld["route_prediction"].float().cpu().numpy()[0]
            pt = ld["speed_wps_prediction"].float().cpu().numpy()[0]
            gr = np.asarray(s.base.path, dtype=np.float64)
            gt = np.asarray(s.base.waypoints, dtype=np.float64)
            P.append(pr); G.append(gr); M.append(np.asarray(s.route_mask))
            PW.append(pt); GW.append(gt); MW.append(np.asarray(s.wp_mask))
            rows.append({"id": s.meta["id"], "run": s.meta["run"], "idx": s.meta["idx"],
                         "split": s.meta["split"], "stage": s.meta["stage"],
                         "command": s.meta["command"],
                         "n_route_valid": int(s.route_mask.sum()),
                         "n_wp_valid": int(s.wp_mask.sum()),
                         "route_ade": float(np.linalg.norm(
                             (pr - gr)[np.asarray(s.route_mask) > 0], axis=-1).mean()),
                         "wps_ade": float(np.linalg.norm(
                             (pt - gt)[np.asarray(s.wp_mask) > 0], axis=-1).mean())})
            if (k + 1) % 500 == 0:
                print("  %d/%d  %.0fs" % (k + 1, len(ds), time.time() - t0), flush=True)

    P = np.stack(P); G = np.stack(G); M = np.stack(M)
    PW = np.stack(PW); GW = np.stack(GW); MW = np.stack(MW)
    # origin excluded from route ADE (legacy convention)
    M2 = M.copy(); M2[:, 0] = 0.0
    route = summarize(P, G, M2)
    wps = summarize(PW, GW, MW)

    by_run = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)
    per_run = {run: {"n": len(v),
                     "route_ade": float(np.mean([x["route_ade"] for x in v])),
                     "wps_ade": float(np.mean([x["wps_ade"] for x in v]))}
               for run, v in by_run.items()}

    out = {"task": "TASK-20261112-003", "ckpt": args.ckpt, "stage": args.stage,
           "splits": list(splits), "n_samples": len(ds),
           "route": route, "wps": wps, "per_run": per_run,
           "definition": "ADE excludes route origin; FDE uses last valid point; "
                         "units are nominal (k=0.0013263), not metres",
           "elapsed_s": time.time() - t0}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    C.json_dump(out, args.out + ".json")
    with open(args.out + ".jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("\n=== %s  stage%s  n=%d ===" % (args.out, args.stage, len(ds)))
    print("  route ADE %.6f  FDE %.6f  (pts %d)" %
          (route["ade"], route["fde"], route["n_points"]))
    print("  wps   ADE %.6f  FDE %.6f  (pts %d)" %
          (wps["ade"], wps["fde"], wps["n_points"]))
    for run, v in sorted(per_run.items()):
        print("    %-46s n=%5d route %.6f wps %.6f" %
              (run[11:], v["n"], v["route_ade"], v["wps_ade"]))


if __name__ == "__main__":
    main()
