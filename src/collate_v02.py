#!/usr/bin/env python3
"""TASK-20261112-003 / collate_v02.py

Collate wrapper: the ORIGINAL SimLingo collate for the current frame, plus the
v02 extras (per-point masks, and optionally the history block).
"""
import numpy as np
import torch

from simlingo_training.utils.internvl2_utils import preprocess_image_batch

IMG_SIZE = 448
IMG_TILES = 1


def make_collate(dm, use_img=False):
    def collate(data):
        ex = dm.dl_collate_fn(data)
        route_mask = torch.from_numpy(np.stack([d.route_mask for d in data])).float()
        wp_mask = torch.from_numpy(np.stack([d.wp_mask for d in data])).float()
        speed_ok = torch.tensor([d.speed_valid for d in data], dtype=torch.bool)
        hist = build_hist(data) if use_img else None
        return ex, hist, route_mask, wp_mask, speed_ok
    return collate


def build_hist(data):
    B = len(data)
    traj = torch.from_numpy(np.stack([d.hist_traj for d in data])).float()
    tvalid = torch.from_numpy(np.stack([d.hist_traj_valid for d in data])).bool()
    ivalid = torch.from_numpy(np.stack([d.hist_img_valid for d in data])).bool()
    tdt = torch.from_numpy(np.stack([d.hist_traj_dt for d in data])).float()
    idt = torch.from_numpy(np.stack([d.hist_img_dt for d in data])).float()
    out = {"traj_feats": traj, "traj_valid": tvalid, "img_valid": ivalid,
           "traj_dt": tdt, "img_dt": idt, "use_history": True,
           "ids": [d.meta["id"] for d in data]}
    K = len(data[0].hist_images)
    px = []
    for d in data:
        imgs = [torch.from_numpy(d.hist_images[k]) for k in range(K)]
        p = preprocess_image_batch(imgs, input_size=IMG_SIZE,
                                   use_global_img=False, max_num_grid=IMG_TILES)
        px.append(p["pixel_values"])
    out["img_pixels"] = torch.stack(px).float()
    return out
