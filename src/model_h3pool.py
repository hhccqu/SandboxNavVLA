#!/usr/bin/env python3
"""History encoders + query modulation for TASK-20261111-003.

Design constraints (from the task sheet):
  * SimLingo backbone, navigation conditioning, output definitions (20 route
    points + 10 time waypoints), losses and post-processing stay FIXED.
  * The current frame still goes through the original VLM path (T == 1 stays
    untouched, no extra visual tokens are inserted into the language sequence).
  * History is fused at low cost: past frames are encoded by the *frozen* vision
    encoder under no_grad, compressed to a few tokens, and the resulting memory
    modulates the 30 learned driving queries that enter the LLM.
  * The modulation is a zero-initialised residual, so with the fusion module
    attached but history disabled/empty the model is numerically identical to
    the baseline. That is asserted in verify_baseline_equiv.py.
"""
import math

import torch
import torch.nn as nn

from simlingo_training.models.adaptors.adaptors import DrivingAdaptor
from simlingo_training.models.driving import DrivingModel

FUSION_MODES = ("attn", "pool")


class HistTrajEncoder(nn.Module):
    """GRU over past ego states expressed in the current ego frame."""

    def __init__(self, in_dim=7, d=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(in_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.gru = nn.GRU(d, d, num_layers=1, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, feats, valid):
        x = self.inp(feats)
        x = x * valid.unsqueeze(-1).to(x.dtype)
        h, _ = self.gru(x)
        h = self.norm(h)
        return h * valid.unsqueeze(-1).to(h.dtype)


class HistImgCompressor(nn.Module):
    """Per-frame attention pooling of frozen InternViT tokens into n_q summary tokens."""

    def __init__(self, in_dim=896, d=128, n_q=8, n_heads=4):
        super().__init__()
        self.n_q = n_q
        self.ln = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d)
        self.q = nn.Parameter(0.02 * torch.randn(1, n_q, d))
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        self.ln2 = nn.LayerNorm(d)

    def forward(self, tok, valid):
        x = self.proj(self.ln(tok))
        q = self.q.expand(x.size(0), -1, -1)
        out, _ = self.attn(q, x, x, need_weights=False)
        out = self.ln1(out)
        out = self.ln2(out + self.ff(out))
        return out * valid.view(-1, 1, 1).to(out.dtype)


class HistFusion(nn.Module):
    """Turns history into a residual added to the driving query embeddings."""

    def __init__(self, hidden, d=128, use_traj=True, use_img=True, mode="attn",
                 n_q=8, n_traj=5, n_img=2, n_traj_feat=7, n_heads=4):
        super().__init__()
        assert mode in FUSION_MODES, mode
        assert use_traj or use_img
        self.use_traj, self.use_img, self.mode = bool(use_traj), bool(use_img), mode
        self.n_traj, self.n_img, self.n_q, self.hidden = n_traj, n_img, n_q, hidden
        self.d = d

        self.traj_enc = HistTrajEncoder(n_traj_feat, d) if use_traj else None
        self.img_comp = HistImgCompressor(hidden, d, n_q, n_heads) if use_img else None

        self.traj_lag_emb = nn.Parameter(0.02 * torch.randn(1, n_traj, d))
        self.img_lag_emb = nn.Parameter(0.02 * torch.randn(1, n_img, d))
        self.type_emb = nn.Parameter(torch.zeros(2, 1, d))

        self.q_proj = nn.Linear(hidden, d, bias=False)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.out_proj = nn.Linear(d, hidden)
        self.pool_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, hidden))

        for m in (self.out_proj, self.pool_mlp[2]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.last_gate = None  # diagnostic: mean |residual| / |query| (filled on forward)

    def _memory(self, hist):
        tokens, masks = [], []
        B = None
        if self.use_traj:
            vt = hist["traj_valid"].bool()
            B = vt.size(0)
            tok = self.traj_enc(hist["traj_feats"].float(), vt) + self.traj_lag_emb
            tokens.append(tok + self.type_emb[0])
            masks.append(vt)
        if self.use_img:
            vi = hist["img_valid"].bool()
            B = vi.size(0)
            feat = hist["img_feats"].float()
            k = vi.size(1)
            ntok = feat.size(1) // k
            tok = self.img_comp(feat.reshape(B * k, ntok, feat.size(-1)), vi.reshape(-1))
            tok = tok.reshape(B, k, self.n_q, self.d) + self.img_lag_emb.unsqueeze(2)
            tok = tok.reshape(B, k * self.n_q, self.d) + self.type_emb[1]
            tokens.append(tok)
            masks.append(vi.unsqueeze(-1).expand(B, k, self.n_q).reshape(B, -1))
        return torch.cat(tokens, 1), torch.cat(masks, 1)

    def forward(self, queries, hist):
        mem, mask = self._memory(hist)
        any_valid = mask.any(1)
        safe = mask.clone()
        if (~any_valid).any():
            safe[~any_valid, 0] = True
        if self.mode == "attn":
            q = self.q_proj(queries.float())
            out, _ = self.attn(q, mem, mem, key_padding_mask=~safe, need_weights=False)
            res = self.out_proj(out)
        else:
            w = safe.float().unsqueeze(-1)
            pooled = (mem * w).sum(1) / w.sum(1).clamp_min(1.0)
            res = self.pool_mlp(pooled).unsqueeze(1).expand(-1, queries.size(1), -1)
        res = res * any_valid.view(-1, 1, 1).to(res.dtype)
        with torch.no_grad():
            self.last_gate = float((res.float().norm(dim=-1).mean() /
                                    queries.float().norm(dim=-1).mean().clamp_min(1e-6)))
        return res.to(queries.dtype)


class TemporalDrivingAdaptor(DrivingAdaptor):
    """DrivingAdaptor whose query embeddings can be modulated by history."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.history = None

    def attach_history(self, fusion):
        self.history = fusion

    def forward(self, driving_example, hist=None, **kw):
        out = super().forward(driving_example, **kw)
        if self.history is not None and hist is not None and hist.get("use_history", True):
            with torch.autocast(device_type="cuda", enabled=False):
                out["inputs"] = out["inputs"] + self.history(out["inputs"], hist)
        return out


class TemporalDrivingModel(DrivingModel):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        old = self.adaptors.driving
        new = TemporalDrivingAdaptor(
            self.language_model.hidden_size,
            speed_wps_mode=self.speed_wps_mode,
            predict_route_as_wps=self.predict_route_as_wps,
        )
        new.load_state_dict(old.state_dict())
        self.adaptors.driving = new

    def attach_history(self, use_traj, use_img, mode="attn", d=128, n_q=8,
                       n_traj=5, n_img=2, n_traj_feat=7):
        fusion = HistFusion(self.language_model.hidden_size, d=d,
                            use_traj=use_traj, use_img=use_img, mode=mode,
                            n_q=n_q, n_traj=n_traj, n_img=n_img,
                            n_traj_feat=n_traj_feat)
        self.adaptors.driving.attach_history(fusion)
        n = sum(p.numel() for p in fusion.parameters())
        print(f"[tmpl] attached history fusion use_traj={use_traj} use_img={use_img} "
              f"mode={mode} new_trainable_params={n:,}", flush=True)
        return fusion

    # ---- frozen visual encoding of past frames ------------------------------
    def encode_history(self, hist):
        """Add `img_feats` to the hist dict by running the frozen vision encoder.

        The encoder is frozen in this training recipe (only adaptors.driving and
        LoRA are trainable), so this is done under no_grad. Nothing is cached
        across steps, so a stale cache can never be reused by accident.
        """
        if hist is None or "img_pixels" not in hist or hist["img_pixels"] is None:
            return hist
        px = hist["img_pixels"]
        B, K, T, C, H, W = px.shape
        flat = px.reshape(B * K * T, C, H, W)
        enc = self.vision_model.image_encoder.model
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                f = enc.extract_feature(flat)
        f = f.float().reshape(flat.size(0), -1, f.shape[-1])
        out = dict(hist)
        out["img_feats"] = f.reshape(B, K * T * f.size(1), f.size(-1))
        return out

    def adaptors_with_history(self, example, hist):
        hist = self.encode_history(hist)
        return self.adaptors(example, hist=hist)
