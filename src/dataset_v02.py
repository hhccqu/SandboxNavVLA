#!/usr/bin/env python3
"""TASK-20261112-003 / dataset_v02.py

Torch dataset over the frozen common pool.

Differences from the old SandtableV02 / SandtableCmd readers (all deliberate):
  1. speed is CAUSAL and gap-aware (see common.causal_speed); the old reader used
     the t -> t+1 displacement, which leaked the future into a numeric input.
  2. no target point is fed to the model (target_points zeros, empty
     placeholder_values), matching the old N2 command-only setup and closing the
     route-head-copies-its-input path.
  3. geometric route / time waypoints carry PER-POINT validity masks; points past
     the end of the run (or across a gap) are masked instead of being clamped to
     the last real position and treated as truth.
  4. history (stage 2 only) is matched by real seconds. Never by frame index.
"""
import json
import math
import os
import sys

import cv2
import numpy as np
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "vendor")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common as C  # noqa: E402
from simlingo_training.utils.custom_types import DatasetOutput  # noqa: E402

PROMPT = ("Current speed: {speed:.3f} m/s. Command: {cmd} Predict the waypoints.")


class Sample:
    """DatasetOutput plus the v02 extras (masks and history block)."""

    __slots__ = ("base", "route_mask", "wp_mask", "hist_images", "hist_traj",
                 "hist_traj_valid", "hist_img_valid", "hist_traj_dt",
                 "hist_img_dt", "meta", "speed_valid")

    def __init__(self, base, route_mask, wp_mask, hist_images, hist_traj,
                 hist_traj_valid, hist_img_valid, hist_traj_dt, hist_img_dt,
                 meta, speed_valid):
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "route_mask", route_mask)
        object.__setattr__(self, "wp_mask", wp_mask)
        object.__setattr__(self, "hist_images", hist_images)
        object.__setattr__(self, "hist_traj", hist_traj)
        object.__setattr__(self, "hist_traj_valid", hist_traj_valid)
        object.__setattr__(self, "hist_img_valid", hist_img_valid)
        object.__setattr__(self, "hist_traj_dt", hist_traj_dt)
        object.__setattr__(self, "hist_img_dt", hist_img_dt)
        object.__setattr__(self, "meta", meta)
        object.__setattr__(self, "speed_valid", speed_valid)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "base"), name)


class SandtableV02Pool(Dataset):
    def __init__(self, splits=("train",), use_history=False, limit=None):
        self.use_history = bool(use_history)
        self._pose = {}
        self._img_cache = {}
        self.items = []
        with open(C.POOL) as f:
            for line in f:
                o = json.loads(line)
                if splits and o["split"] not in splits:
                    continue
                self.items.append(o)
        if limit:
            self.items = self.items[:limit]
        print(f"[ds_v02] splits={splits} use_history={self.use_history} "
              f"samples={len(self.items)} "
              f"runs={len(set(o['run'] for o in self.items))}", flush=True)

    def __len__(self):
        return len(self.items)

    def pose(self, run):
        if run not in self._pose:
            p = C.pose_arrays(run)
            n = len(p["t"])
            vid = C.load_valid().get(run, set())
            p["valid"] = np.array([1 if i in vid else 0 for i in range(n)], dtype=np.int8)
            self._pose[run] = p
        return self._pose[run]

    def _read_front(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h = img.shape[0]
        return img[: int(h - (h * 4.8) // 16), :, :]

    def __getitem__(self, index):
        o = self.items[index]
        run, i = o["run"], int(o["idx"])
        pose = self.pose(run)
        t, xy, hd, cum = pose["t"], pose["xy"], pose["hd"], pose["cum"]
        valid = pose["valid"]
        k = C.K
        th = math.radians(float(hd[i]))
        c0, s0 = math.cos(th), math.sin(th)
        p0 = xy[i]

        # ---- labels with per-point masks -----------------------------------
        wp, wp_mask, rp, route_mask = C.frame_masks(pose, i, valid)

        def to_ego(pts):
            dx = pts[:, 0] - p0[0]
            dy = pts[:, 1] - p0[1]
            return np.stack([c0 * dx + s0 * dy, -s0 * dx + c0 * dy], axis=1) * k

        waypoints = to_ego(wp).astype(np.float32)
        route = to_ego(rp).astype(np.float32)
        route[0] = 0.0
        # masked points carry the current pose (zero in ego frame) so that a
        # careless consumer sees a plausible value, but the mask is authoritative.
        waypoints[wp_mask <= 0] = 0.0
        route[route_mask <= 0] = 0.0

        # ---- causal speed ---------------------------------------------------
        # The old N2 label is `speed_ms` = raw 1-s displacement * k, i.e. already
        # in nominal training units. Feeding raw units (754x larger) would put the
        # prompt far outside the distribution N2 was trained on.
        speed_raw, speed_ok = C.causal_speed(pose, i, valid)
        speed = speed_raw * C.K

        # ---- current image --------------------------------------------------
        img = self._read_front(pose["img"][i])
        chw = np.transpose(img, (2, 0, 1)).astype(np.float32)[None, ...]

        cmd = o["command"]
        prompt = PROMPT.format(speed=speed, cmd=cmd)

        base = DatasetOutput(
            conversation=[
                {"role": "user", "content": [{"type": "text", "text": prompt},
                                             {"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Waypoints:"}]},
            ],
            answer=[{"role": "assistant",
                     "content": [{"type": "text", "text": "Waypoints:"}]}],
            image_ff=chw,
            image_ff_org_size=chw,
            waypoints=waypoints,
            waypoints_1d=np.zeros((10, 2), dtype=np.float32),
            path=route,
            target_points=np.zeros((2, 2), dtype=np.float32),
            speed=float(speed),
            placeholder_values={},
            measurement_path=f"{run}#{i}",
            dataset="driving",
        )

        # ---- history (stage 2) ---------------------------------------------
        traj = np.zeros((len(C.TRAJ_LAGS_S), 7), dtype=np.float32)
        traj_valid = np.zeros(len(C.TRAJ_LAGS_S), dtype=bool)
        traj_dt = np.zeros(len(C.TRAJ_LAGS_S), dtype=np.float32)
        img_valid = np.zeros(len(C.IMG_LAGS_S), dtype=bool)
        img_dt = np.zeros(len(C.IMG_LAGS_S), dtype=np.float32)
        hist_images = np.zeros((len(C.IMG_LAGS_S),) + chw.shape[1:], dtype=np.float32)

        if self.use_history:
            ti, tdts, tok = C.history_indices(pose, i, valid, C.TRAJ_LAGS_S, C.TRAJ_DT_TOL_S)
            ii, idts, iok = C.history_indices(pose, i, valid, C.IMG_LAGS_S, C.IMG_DT_TOL_S)
            traj_valid[:] = tok
            traj_dt[:] = tdts
            img_valid[:] = iok
            img_dt[:] = idts
            for kk in range(len(C.TRAJ_LAGS_S)):
                if not tok[kk]:
                    continue
                j = ti[kk]
                dx = xy[j][0] - p0[0]
                dy = xy[j][1] - p0[1]
                dpsi = math.radians(float(hd[j]) - float(hd[i]))
                # history speed feature follows the old spec: backward difference
                # at j, per second, normalised by the route spacing constant.
                if j - 1 >= 0 and t[j] - t[j - 1] > 0:
                    ddx = xy[j][0] - xy[j - 1][0]
                    ddy = xy[j][1] - xy[j - 1][1]
                    spd_hist = math.hypot(ddx, ddy) / (t[j] - t[j - 1]) / C.ROUTE_SPACING_RAW
                else:
                    spd_hist = 0.0
                traj[kk] = np.array([
                    (c0 * dx + s0 * dy) / C.ROUTE_SPACING_RAW,
                    (-s0 * dx + c0 * dy) / C.ROUTE_SPACING_RAW,
                    tdts[kk],
                    math.sin(dpsi), math.cos(dpsi),
                    spd_hist,
                    1.0,
                ], dtype=np.float32)
            for kk in range(len(C.IMG_LAGS_S)):
                if not iok[kk]:
                    continue
                j = ii[kk]
                im = self._read_front(pose["img"][j])
                if im.shape[:2] != img.shape[:2]:
                    im = cv2.resize(im, (img.shape[1], img.shape[0]))
                hist_images[kk] = np.transpose(im, (2, 0, 1)).astype(np.float32)

        meta = {"id": f"{run}#{i}", "run": run, "idx": i, "split": o["split"],
                "stage": o["stage"], "command": cmd,
                "t": float(t[i] - t[0]), "speed_ok": bool(speed_ok),
                "n_route_valid": int(route_mask.sum()),
                "n_wp_valid": int(wp_mask.sum())}

        return Sample(base, route_mask.astype(np.float32), wp_mask.astype(np.float32),
                      hist_images, traj, traj_valid, img_valid, traj_dt, img_dt,
                      meta, bool(speed_ok))
