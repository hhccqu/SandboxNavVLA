#!/usr/bin/env python3
"""TASK-20261112-003 / engine.py

Model construction, trainable scope, masked loss and checkpoint I/O for the two
stages.

Stage 1 (N2-v02)      : single-frame model. current front image + causal speed +
                        semantic command. Trains adaptors.driving + LoRA.
Stage 2 (H3pool-v02)  : same, plus a NEWLY INITIALISED history branch (5 ego
                        trajectory observations, 2 history images, pooled fusion,
                        zero-init residual on the 30 driving queries). Trains
                        history branch + adaptors.driving + LoRA.

Checkpoints are FULL, self-contained state dicts (fp16), not deltas against the
previous file. The dependency chain is recorded alongside so a successor can
prove which weights were loaded in which order.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor")
for _p in (HERE, VENDOR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common as C  # noqa: E402

# Loss weights: the old N2/H3pool recipe, kept explicitly.
#   route  x1, speed_wps x5, language CE x1  (the CE really was in the old total
#   loss: old H3pool step-0 log shows loss 4.975 against route 0.0072 + 5*0.0006)
WEIGHTS = {"route_loss": 1.0, "speed_wps_loss": 5.0, "language_loss": 1.0}


# --------------------------------------------------------------------- config
def build_cfg(with_history):
    from hydra import compose, initialize_config_dir
    import simlingo_training.config  # noqa: F401
    cfg_dir = os.path.join(VENDOR, "simlingo_training", "config")
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.1"):
        cfg = compose(config_name="config", overrides=["experiment=simlingo_seed1"])
    if with_history:
        cfg.model._target_ = "model_h3pool.TemporalDrivingModel"
    return cfg


def build_processor(cfg):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(cfg.model.vision_model.variant,
                                         trust_remote_code=True)


def build_datamodule(cfg, processor):
    from omegaconf import OmegaConf
    import simlingo_training.config  # noqa: F401
    from simlingo_training.dataloader.datamodule import DataModule
    return DataModule(base_dataset=OmegaConf.create({"use_1d_wps": False}),
                      processor=processor, predict=False,
                      encoder_variant=cfg.model.vision_model.variant,
                      use_global_img=False)


# ---------------------------------------------------------------------- model
def build_model(stage, cfg, processor, dm, init_ckpt=None, verbose=True):
    """Build the model for `stage` and load its initialisation chain.

    stage 1: official SimLingo base (STRICT) -> old N2 increment.
    stage 2: this round's N2-v02_best as a COMPLETE state dict (STRICT), then the
             history branch is attached with zero-initialised output projections.
             The old H3pool_best is never loaded.
    """
    import torch
    from hydra.utils import instantiate

    model = instantiate(cfg.model, cfg_data_module=cfg.data_module,
                        processor=processor, cache_dir=None, _recursive_=False)
    chain = []
    history_attached = False

    def _sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 22), b""):
                h.update(c)
        return h.hexdigest()

    def attach_history():
        nonlocal history_attached
        if history_attached:
            return
        fusion = model.attach_history(use_traj=True, use_img=True, mode="pool",
                                      d=128, n_q=8, n_traj=len(C.TRAJ_LAGS_S),
                                      n_img=len(C.IMG_LAGS_S), n_traj_feat=7)
        n_zero = sum(1 for p in fusion.parameters() if float(p.abs().sum()) == 0.0)
        if verbose:
            print(f"[engine] history branch attached (zero-init residual); "
                  f"all-zero tensors in fusion={n_zero}", flush=True)
        chain.append({"role": "history_branch", "init": "newly initialised, zero residual",
                      "loaded_from_old_h3pool": False})
        history_attached = True

    # A stage-2 checkpoint already contains history parameters.  Attach the
    # history branch BEFORE loading it; otherwise its 41 history keys are
    # reported as unexpected.  For a stage-1/N2 init it is equally safe: the
    # missing history keys stay zero-initialised.
    if stage == 2 and init_ckpt:
        attach_history()

    if init_ckpt:
        sd = torch.load(init_ckpt, map_location="cpu")
        state = sd.get("full_state") or sd.get("trainable_state") or sd.get("model") or sd
        state = {k: v.float() if torch.is_tensor(v) else v for k, v in state.items()}
        miss, unexp = model.load_state_dict(state, strict=False)
        if verbose:
            print(f"[engine] stage{stage} init from {init_ckpt} "
                  f"missing={len(miss)} unexpected={len(unexp)}", flush=True)
        if unexp:
            raise RuntimeError(f"unexpected keys loading {init_ckpt}: {unexp[:5]}")
        chain.append({"role": ("n2_v02_best" if stage == 2 else "stage1_init"),
                      "path": init_ckpt, "sha256": _sha(init_ckpt),
                      "len_missing": len(miss), "len_unexpected": len(unexp)})
    else:
        base = torch.load(C.BASE_CKPT, map_location="cpu")
        r = model.load_state_dict(base)
        if r.missing_keys or r.unexpected_keys:
            raise RuntimeError(f"base mismatch missing={r.missing_keys[:5]} "
                               f"unexpected={r.unexpected_keys[:5]}")
        chain.append({"role": "official_base", "path": C.BASE_CKPT, "sha256": _sha(C.BASE_CKPT)})
        if verbose:
            print(f"[engine] base loaded STRICT from {C.BASE_CKPT}", flush=True)

        if stage == 1 and os.path.exists(C.N2_CKPT):
            inc = torch.load(C.N2_CKPT, map_location="cpu")
            r2 = model.load_state_dict(inc["model"], strict=False)
            chain.append({"role": "old_n2_increment", "path": C.N2_CKPT,
                          "sha256": _sha(C.N2_CKPT),
                          "len_unexpected": len(r2.unexpected_keys),
                          "len_missing": len(r2.missing_keys)})
            if verbose:
                print(f"[engine] old N2 increment loaded {C.N2_CKPT} "
                      f"missing={len(r2.missing_keys)} unexpected={len(r2.unexpected_keys)}",
                      flush=True)

    if stage == 2:
        attach_history()

    model._init_chain = chain
    return model


def set_trainable(model):
    """Head + LoRA always; history modules only when they exist.

    FROZEN by omission: vision tower, projector, and the language base weights.
    """
    head, lora_p, new = [], [], []
    for _, p in model.named_parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if n.startswith("adaptors.driving") and ".history." not in n:
            p.requires_grad = True
            head.append(p)
        elif "lora_" in n:
            p.requires_grad = True
            lora_p.append(p)
        elif ".history." in n:
            p.requires_grad = True
            new.append(p)
    return head, lora_p, new


# ----------------------------------------------------------------------- loss
def mask_loss_dict(ld, route_mask, wp_mask):
    """Replace the adaptor's all-ones validity tensors with our per-point masks.

    The adaptor returns (per_element_loss, validity) pairs; summarise_losses
    normalises by the validity count, so zeroing invalid points is enough.
    """
    for key, m in (("route_loss", route_mask), ("speed_wps_loss", wp_mask)):
        if key in ld:
            loss, valid = ld[key]
            ld[key] = (loss, valid * m.to(valid.dtype))
    return ld


def compute_total(model, example, hist, route_mask, wp_mask):
    import torch
    from simlingo_training.models.utils import summarise_losses
    if hist is not None:
        hist = model.encode_history(hist)
        ad = model.adaptors(example, hist=hist)
    else:
        ad = model.adaptors(example)
    feats, logits = model.forward_model(example.driving_input, ad)
    ld = model.adaptors.compute_loss(feats, logits, ad, example)
    ld = mask_loss_dict(ld, route_mask, wp_mask)
    losses_only = {k: v for k, v in ld.items() if k.endswith("loss")}
    return summarise_losses(losses_only, weights=WEIGHTS), ld


# ------------------------------------------------------------------ devices
def move_batch(b):
    import torch
    di = b.driving_input
    di = di._replace(camera_images=di.camera_images.cuda().to(torch.bfloat16),
                     image_sizes=di.image_sizes.cuda(),
                     vehicle_speed=di.vehicle_speed.cuda().float(),
                     target_point=di.target_point.cuda().float(),
                     prompt=di.prompt._replace(
                         phrase_ids=di.prompt.phrase_ids.cuda(),
                         phrase_valid=di.prompt.phrase_valid.cuda(),
                         phrase_mask=di.prompt.phrase_mask.cuda(),
                         loss_masking=di.prompt.loss_masking.cuda()),
                     prompt_inference=di.prompt_inference._replace(
                         phrase_ids=di.prompt_inference.phrase_ids.cuda(),
                         phrase_valid=di.prompt_inference.phrase_valid.cuda(),
                         phrase_mask=di.prompt_inference.phrase_mask.cuda(),
                         loss_masking=di.prompt_inference.loss_masking.cuda()))
    lb = b.driving_label
    return b._replace(driving_input=di,
                      driving_label=lb._replace(waypoints=lb.waypoints.cuda().float(),
                                                path=lb.path.cuda().float()))


def hist_to_cuda(hist, use_img=True):
    import torch
    if hist is None:
        return None
    out = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in hist.items()}
    if use_img and "img_pixels" in out:
        out["img_pixels"] = out["img_pixels"].cuda().to(torch.bfloat16)
    return out


# ---------------------------------------------------------------- checkpoints
def save_checkpoint(path, model, opt=None, scaler=None, meta=None, full=True):
    """Write a self-contained checkpoint (full state, fp16) plus resume state."""
    import torch
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sd = model.state_dict()
    blob = {
        "full_state": ({k: v.detach().to("cpu", torch.float16) for k, v in sd.items()}
                       if full else None),
        "trainable_state": {k: v.detach().cpu() for k, v in sd.items()
                            if ("adaptors.driving" in k) or ("lora_" in k)
                            or (".history." in k)},
        "meta": meta or {},
        "init_chain": getattr(model, "_init_chain", []),
    }
    if opt is not None:
        blob["optimizer"] = opt.state_dict()
    if scaler is not None:
        blob["scaler"] = scaler.state_dict()
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint_state(path):
    import torch
    blob = torch.load(path, map_location="cpu")
    state = blob.get("full_state") or blob.get("model") or blob.get("trainable_state")
    return state, blob
