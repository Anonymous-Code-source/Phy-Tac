# -*- coding: utf-8 -*-
"""
Latent Diffusion for Xsense (Plan B + Route A: Concat + FiLM + Cross-Attn)
- Frozen VAE encoder (shared for input & contact_depth); train diffusion only
- Conditions: input_diff (zin), contact_depth (zcdp), mass/texture tokens
- v-pred by default, AMP-enabled
- VAL: MAE/RMSE/SSIM/PSNR/LPIPS (+ save comparison grid)
- Logging: tqdm single-line bars, TensorBoard, CSV
- Best checkpoint by crit = (100-PSNR) + 100*LPIPS
- Improved training loss: Min-SNR weighting + optional Huber, pixel loss cosine ramp

Run:
    python train_ldm_xsense.py
or:
    python train_ldm_xsense.py --cfg ldm_xsense.yaml
"""
import os, re, math, argparse, random, csv, importlib
from time import time
from typing import Dict, Tuple, Optional

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from tqdm import tqdm
from torchvision.utils import make_grid, save_image

# External (your repo)
from src.xsense_dataset import XsenseTactileDataset
from src.mt_embedder_cmd import MassTextureCmdEmbedder

# -------------------- Small utils --------------------
def exists(x): return x is not None
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def tlog(msg: str):
    try: tqdm.write(msg)
    except Exception: print(msg)

# -------- AMP compatibility (new/old torch) --------
from contextlib import contextmanager
def make_grad_scaler(amp_enabled: bool):
    try:
        return torch.amp.GradScaler(enabled=amp_enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)

@contextmanager
def autocast_fp16(amp_enabled: bool):
    try:
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            yield
    except Exception:
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            yield

# -------- Optional LPIPS (auto-detect) --------
try:
    import lpips  # pip install lpips
    _HAS_LPIPS = True
except Exception:
    lpips = None
    _HAS_LPIPS = False

_LPIPS_NET = None
def get_lpips(device):
    global _LPIPS_NET
    if not _HAS_LPIPS:
        return None
    if _LPIPS_NET is None:
        _LPIPS_NET = lpips.LPIPS(net='alex').to(device).eval()
        for p in _LPIPS_NET.parameters(): p.requires_grad = False
    return _LPIPS_NET

# -------------------- Spatial alignment --------------------
def center_crop_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    H, W = x.shape[-2:]
    if H == h and W == w: return x
    top = max((H - h) // 2, 0); left = max((W - w) // 2, 0)
    return x[..., top:top + h, left:left + w]

def match_spatial_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    H, W = x.shape[-2:]; Hr, Wr = ref.shape[-2:]
    if H == Hr and W == Wr: return x
    if H >= Hr and W >= Wr: return center_crop_to(x, Hr, Wr)
    pad_h = max(Hr - H, 0); pad_w = max(Wr - W, 0)
    pad = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)  # (left,right,top,bottom)
    return F.pad(x, pad, mode="replicate")

# -------------------- Load Frozen VAE --------------------
def load_frozen_vae(vae_cfg, device):
    import_path = vae_cfg.get("import_path", "train_vae_xsense_SD")
    try:
        vae_mod = importlib.import_module(import_path)
        VAEClass = getattr(vae_mod, "AutoencoderKL_SD")
    except Exception as e:
        raise ImportError(f"无法从 {import_path} 导入 AutoencoderKL_SD：{e}")

    vae = VAEClass(
        in_ch=vae_cfg.get("input_channels", 3),
        out_ch=vae_cfg.get("output_channels", 3),
        base_ch=vae_cfg.get("base_ch", 64),
        ch_mult=vae_cfg.get("ch_mult", [1, 2, 4, 4]),
        num_res_blocks=vae_cfg.get("num_res_blocks", 2),
        z_ch=vae_cfg.get("z_channels", 4),
        attn_mid=vae_cfg.get("attn_mid", True),
    )

    # load weights (old/new torch compatible)
    try:
        state = torch.load(vae_cfg["ckpt_path"], map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(vae_cfg["ckpt_path"], map_location="cpu")

    missing, unexpected = vae.load_state_dict(state, strict=True)
    if missing or unexpected:
        tlog(f"[VAE] state_dict keys mismatch -> missing:{missing} unexpected:{unexpected}")

    vae.to(device).eval()
    for p in vae.parameters(): p.requires_grad = False

    # wrap encode/decode to return tensors directly
    _enc = vae.encode; _dec = vae.decode
    def _encode_wrap(x):
        out = _enc(x)
        if isinstance(out, (tuple, list)): return out[0]
        if isinstance(out, dict):
            return out.get("z", next(iter(out.values())))
        return out
    def _decode_wrap(z):
        out = _dec(z)
        if isinstance(out, (tuple, list)): return out[0]
        if isinstance(out, dict):
            return out.get("x", next(iter(out.values())))
        return out
    vae.encode = _encode_wrap
    vae.decode = _decode_wrap

    vae.in_ch = vae_cfg.get("input_channels", 3)
    vae.z_channels = vae_cfg.get("z_channels", 4)
    return vae

# -------------------- Embeddings --------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        denom = max(half - 1, 1)
        freqs = torch.exp(-math.log(10000) * torch.arange(0, max(half, 1), device=t.device).float() / denom)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1: emb = F.pad(emb, (0, 1))
        return self.mlp(emb)

class CrossAttention(nn.Module):
    def __init__(self, dim: int, cond_dim: int, num_heads: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.h = num_heads; self.d = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(cond_dim, dim, bias=False)
        self.v = nn.Linear(cond_dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).view(B, N, self.h, self.d).transpose(1, 2)
        k = self.k(cond_tokens).view(B, -1, self.h, self.d).transpose(1, 2)
        v = self.v(cond_tokens).view(B, -1, self.h, self.d).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.d), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.o(out)

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, film_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.use_film = film_dim is not None
        if self.use_film:
            self.film = nn.Linear(film_dim, out_ch * 2)
        self.need_skip = (in_ch != out_ch)
        if self.need_skip:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x, t_emb, film_emb=None):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.norm2(h) + self.t_proj(t_emb)[:, :, None, None]
        if self.use_film and film_emb is not None:
            gamma, beta = self.film(film_emb).chunk(2, dim=-1)
            h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.conv2(self.act(h))
        if self.need_skip: x = self.skip(x)
        return x + h

# -------------------- UNet (Concat + FiLM + X-Attn) --------------------
class UNetLDM(nn.Module):
    def __init__(
        self,
        zc: int = 4,
        base: int = 128,
        mult: Tuple[int, ...] = (1, 2, 4),
        n_res: int = 2,
        t_dim: int = 256,
        cond_dim: int = 256,
        heads: int = 4,
        attn_res: Tuple[int, ...] = (1, 2),
        film: bool = True,
        v_pred: bool = True,
    ):
        super().__init__()
        self.v_pred = v_pred
        self.tmlp = SinusoidalTimeEmbedding(t_dim)

        in_ch = zc * 3
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        # Down
        self.downs = nn.ModuleList()
        feats = []
        ch = base; attn_set = set(attn_res)
        for i, m in enumerate(mult):
            out = base * m
            blocks = nn.ModuleList([ResBlock(ch if j == 0 else out, out, t_dim, cond_dim if film else None)
                                    for j in range(n_res)])
            attn = nn.ModuleList([CrossAttention(out, cond_dim, heads)]) if i in attn_set else None
            down = nn.Conv2d(out, out, 3, stride=2, padding=1) if i < len(mult) - 1 else nn.Identity()
            self.downs.append(nn.ModuleDict(dict(blocks=blocks, attn=attn, down=down)))
            feats.append(out); ch = out

        # Mid
        self.mid1 = ResBlock(ch, ch, t_dim, cond_dim if film else None)
        self.mid_attn = CrossAttention(ch, cond_dim, heads)
        self.mid2 = ResBlock(ch, ch, t_dim, cond_dim if film else None)

        # Up
        self.ups = nn.ModuleList()
        prev_ch = ch
        for i, out in reversed(list(enumerate(feats))):
            up = nn.ConvTranspose2d(prev_ch, out, 4, stride=2, padding=1) if i > 0 else nn.Identity()
            in_ch_after_up = out if i > 0 else prev_ch
            first_in_ch = in_ch_after_up + out
            blocks = nn.ModuleList(
                [ResBlock(first_in_ch, out, t_dim, cond_dim if film else None)] +
                [ResBlock(out, out, t_dim, cond_dim if film else None) for _ in range(n_res - 1)]
            )
            attn = nn.ModuleList([CrossAttention(out, cond_dim, heads)]) if i in attn_set else None
            self.ups.append(nn.ModuleDict(dict(blocks=blocks, attn=attn, up=up)))
            prev_ch = out

        self.out_norm = nn.GroupNorm(32, prev_ch)
        self.out_conv = nn.Conv2d(prev_ch, zc, 3, padding=1)

    def forward(self, z_noisy, z_in, z_cdp, t, cond_tokens):
        B, C, H, W = z_noisy.shape
        x = torch.cat([z_noisy, z_in, z_cdp], dim=1)
        t_emb = self.tmlp(t)
        film_emb = cond_tokens.mean(dim=1)

        # Down
        skips = []
        x = self.in_conv(x)
        for m in self.downs:
            for blk in m["blocks"]:
                x = blk(x, t_emb, film_emb)
            if m["attn"] is not None:
                Bc, Cc, Hc, Wc = x.shape
                xf = x.permute(0,2,3,1).reshape(Bc, Hc*Wc, Cc)
                xf = xf + m["attn"][0](xf, cond_tokens)
                x = xf.reshape(Bc,Hc,Wc,Cc).permute(0,3,1,2).contiguous()
            skips.append(x)
            x = m["down"](x)

        # Mid
        x = self.mid1(x, t_emb, film_emb)
        Bm,Cm,Hm,Wm = x.shape
        xf = x.permute(0,2,3,1).reshape(Bm,Hm*Wm,Cm)
        xf = xf + self.mid_attn(xf, cond_tokens)
        x = xf.reshape(Bm,Hm,Wm,Cm).permute(0,3,1,2).contiguous()
        x = self.mid2(x, t_emb, film_emb)

        # Up
        for m in self.ups:
            x = m["up"](x)
            skip = skips.pop()
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            for blk in m["blocks"]:
                x = blk(x, t_emb, film_emb)
            if m["attn"] is not None:
                Bc, Cc, Hc, Wc = x.shape
                xf = x.permute(0,2,3,1).reshape(Bc,Hc*Wc,Cc)
                xf = xf + m["attn"][0](xf, cond_tokens)
                x = xf.reshape(Bc,Hc,Wc,Cc).permute(0,3,1,2).contiguous()

        x = F.silu(self.out_norm(x))
        return self.out_conv(x)

# -------------------- Scheduler --------------------
class NoiseScheduler:
    def __init__(self, T=1000, kind="cosine"):
        self.T = T
        if kind == "cosine":
            s = 0.008
            steps = torch.arange(T + 1, dtype=torch.float32)
            ac = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
            self.alphas_cumprod = (ac / ac[0]).clamp(min=1e-6)
        else:
            betas = torch.linspace(1e-4, 0.02, T)
            alphas = 1.0 - betas
            self.alphas_cumprod = torch.cumprod(alphas, dim=0)
    def to(self, d): self.alphas_cumprod = self.alphas_cumprod.to(d); return self
    def sample_t(self, b, d): return torch.randint(1, self.T, (b,), device=d, dtype=torch.long)
    def q_sample(self, z0, t, eps):
        a_bar = self.alphas_cumprod[t]
        a_prev = self.alphas_cumprod[t - 1]
        alpha = (a_bar / a_prev).clamp(min=1e-6)
        sigma = (1 - alpha).sqrt()
        zt = alpha.sqrt().view(-1,1,1,1) * z0 + sigma.view(-1,1,1,1) * eps
        return zt, alpha, sigma, a_bar

# -------------------- EMA --------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.m = model
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.requires_grad}
        self.decay = decay
        self.backup = None
    @torch.no_grad()
    def update(self):
        for k, v in self.m.state_dict().items():
            if k in self.shadow and v.requires_grad:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
    @torch.no_grad()
    def store(self): self.backup = {k: v.detach().clone() for k, v in self.m.state_dict().items()}
    @torch.no_grad()
    def copy_to(self):
        sd = self.m.state_dict(); sd.update(self.shadow)
        self.m.load_state_dict(sd, strict=False)
    @torch.no_grad()
    def restore(self):
        if self.backup is not None:
            sd = self.m.state_dict(); sd.update(self.backup)
            self.m.load_state_dict(sd, strict=False)

# -------------------- Metrics & Visualization --------------------
def psnr_from_mse(mse): return 10.0 * math.log10(1.0 / max(mse, 1e-12))

def ssim_torch(x, y, C1=0.01**2, C2=0.03**2):
    """
    simplified SSIM on [0,1], NCHW
    """
    x = x.float(); y = y.float()
    def _avg3(z):
        z = F.pad(z, (1,1,1,1), mode="replicate")
        return F.avg_pool2d(z, kernel_size=3, stride=1, padding=0)
    mu_x = _avg3(x); mu_y = _avg3(y)
    sigma_x = _avg3(x*x) - mu_x*mu_x
    sigma_y = _avg3(y*y) - mu_y*mu_y
    sigma_xy = _avg3(x*y) - mu_x*mu_y
    ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / (
        (mu_x*mu_x + mu_y*mu_y + C1)*(sigma_x + sigma_y + C2) + 1e-12
    )
    return ssim_map.mean().item()

@torch.no_grad()
def compute_batch_metrics(x_pred_01: torch.Tensor, x_tgt_01: torch.Tensor, device, lpips_net=None):
    """
    Inputs in [0,1], NCHW. Returns sums for global aggregation (true MAE/RMSE) +
    sample-averaged PSNR/SSIM/LPIPS weighted by B.
    """
    B, C, H, W = x_pred_01.shape
    mae_sum = F.l1_loss(x_pred_01, x_tgt_01, reduction="sum").item()
    mse_sum = F.mse_loss(x_pred_01, x_tgt_01, reduction="sum").item()
    pix = B * C * H * W

    mse_mean = mse_sum / pix
    psnr = psnr_from_mse(mse_mean)
    ssim = ssim_torch(x_pred_01, x_tgt_01)

    lpips_val = None
    if lpips_net is not None:
        def to3(x):
            if x.size(1) == 3: return x
            if x.size(1) > 3:  return x[:, :3]
            return x.repeat(1,3,1,1)
        xh_m11 = (to3(x_pred_01)*2.0 - 1.0).clamp(-1,1)
        xt_m11 = (to3(x_tgt_01)*2.0 - 1.0).clamp(-1,1)
        try:
            lp = lpips_net(xh_m11, xt_m11, normalize=False)
            lpips_val = lp.mean().item()
        except Exception:
            lpips_val = None

    return {
        "mae_sum": mae_sum,
        "mse_sum": mse_sum,
        "pix": pix,
        "psnr_sum": psnr * B,
        "ssim_sum": ssim * B,
        "lpips_sum": (lpips_val * B) if (lpips_val is not None) else None,
        "b": B
    }

@torch.no_grad()
def make_val_grid(batch, x_pred, vae_in_ch, n=8):
    dev = x_pred.device
    n = min(n, batch["vae_in_pixel"].size(0))
    x_in  = batch["vae_in_pixel"][:n].to(dev, non_blocking=True).clone()
    x_tgt = batch["vae_tgt_pixel"][:n].to(dev, non_blocking=True).clone()
    x_hat = x_pred[:n].to(dev, non_blocking=True).clone()
    x_cdp = batch["cond_contact_pixel"][:n].to(dev, non_blocking=True).clone()

    x_hat = match_spatial_to(x_hat, x_tgt)
    to01 = lambda x: ((x.clamp(-1,1) + 1) * 0.5).clamp(0,1)
    xin, xtg, xht, xcd = map(to01, (x_in, x_tgt, x_hat, x_cdp))
    if xcd.size(1) == 1: xcd = xcd.repeat(1,3,1,1)
    def to3(x): return x if x.size(1)==3 else (x[:, :3] if x.size(1)>3 else x.repeat(1,3,1,1))
    xin, xtg, xht = map(to3, (xin, xtg, xht))
    row = torch.cat([xin, xcd, xtg, xht], dim=3)
    return make_grid(row, nrow=1)

def save_val_images(grid, save_dir, step, prefix="val_cmp"):
    ensure_dir(save_dir)
    save_path = os.path.join(save_dir, f"{prefix}_step{step}.png")
    save_image(grid.detach().cpu(), save_path)
    return save_path

# -------------------- loss helpers --------------------
def huber_loss(input: torch.Tensor, target: torch.Tensor, delta: float = 0.01, reduction: str = "mean"):
    err = input - target
    abs_e = err.abs()
    quad = torch.clamp(abs_e, max=delta)
    lin  = abs_e - quad
    loss = 0.5 * (quad ** 2) / delta + lin
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss

def min_snr_weight(a_bar: torch.Tensor, min_snr_gamma: float, v_prediction: bool) -> torch.Tensor:
    """
    a_bar: ᾱ_t, shape [B]
    对 ε-pred：weight = min(gamma, SNR)/SNR
    对 v-pred：给温和权重：sqrt(min(gamma,SNR)/SNR)
    """
    snr = (a_bar / (1.0 - a_bar)).clamp(min=1e-8)
    w = torch.minimum(torch.full_like(snr, min_snr_gamma), snr) / snr
    if v_prediction:
        w = w.sqrt()
    return w  # shape [B]

def cosine_ramp(step: int, warmup: int, max_steps: int) -> float:
    if step < warmup: return 0.0
    if step >= max_steps: return 1.0
    t = (step - warmup) / max(1, (max_steps - warmup))
    return 0.5 - 0.5 * math.cos(math.pi * t)

# -------------------- EVALUATE (with MAE/RMSE/SSIM/PSNR/LPIPS + CRIT) --------------------
@torch.no_grad()
def evaluate(dl_va, vae, unet, embedder, sched, v_pred, emb_cfg, device,
             amp=True, vis_dir=None, vis_n=8, writer=None, step=0):
    unet.eval(); embedder.eval()

    loss_sum = 0.0; n_img = 0
    mae_sum = 0.0; mse_sum = 0.0; pix_sum = 0
    psnr_sum = 0.0; ssim_sum = 0.0
    lpips_sum = 0.0; lpips_n = 0

    first_grid_saved = False
    last_saved_path = None
    lpips_net = get_lpips(device)

    itr = tqdm(dl_va, dynamic_ncols=True, desc=f"val@{step}", smoothing=0.1, leave=False)
    for bi, batch in enumerate(itr):
        xin  = batch["vae_in_pixel"].to(device)
        xtgt = batch["vae_tgt_pixel"].to(device)
        xcdp = batch["cond_contact_pixel"].to(device)

        z0   = vae.encode(xtgt)
        zin  = vae.encode(xin)
        if xcdp.shape[1] == 1 and vae.in_ch == 3: xcdp = xcdp.repeat(1,3,1,1)
        zcdp = vae.encode(xcdp)

        B = z0.size(0)
        t = sched.sample_t(B, device)
        eps = torch.randn_like(z0)

        a_bar = sched.alphas_cumprod[t]
        a = a_bar.sqrt().view(-1,1,1,1)
        s = (1.0 - a_bar).sqrt().view(-1,1,1,1)
        zt = a * z0 + s * eps

        mass = batch["mass_value"].to(device).view(B)
        tex  = batch["texture_id"].to(device).view(B)
        cond_tok, _ = embedder(mass, tex, None, cond_drop_prob=0.0)

        with autocast_fp16(amp):
            pred = unet(zt, zin, zcdp, t, cond_tok)
            if v_pred:
                v_target = a * eps - s * z0
                loss = F.mse_loss(pred, v_target)
                z0_hat = a * zt - s * pred
            else:
                loss = F.mse_loss(pred, eps)
                z0_hat = (zt - s * pred) / (a + 1e-8)

        x_hat = vae.decode(z0_hat).clamp(-1,1)
        x_hat = match_spatial_to(x_hat, xtgt)

        xh = ((x_hat + 1) * 0.5).clamp(0,1)
        xt = ((xtgt.clamp(-1,1) + 1) * 0.5).clamp(0,1)

        bm = compute_batch_metrics(xh, xt, device, lpips_net=lpips_net)

        loss_sum += loss.item() * B
        mae_sum  += bm["mae_sum"]
        mse_sum  += bm["mse_sum"]
        pix_sum  += bm["pix"]
        psnr_sum += bm["psnr_sum"]
        ssim_sum += bm["ssim_sum"]
        if bm["lpips_sum"] is not None:
            lpips_sum += bm["lpips_sum"]; lpips_n += bm["b"]
        n_img    += B

        itr.set_postfix_str(f"loss={loss_sum/max(n_img,1):.4f}")

        if (not first_grid_saved) and vis_dir:
            grid = make_val_grid(batch, x_hat, vae.in_ch, n=min(vis_n, B))
            last_saved_path = save_val_images(grid, vis_dir, step, prefix="val_cmp")
            first_grid_saved = True

    val_loss = loss_sum / max(n_img, 1)
    val_mae  = mae_sum / max(pix_sum, 1)
    val_rmse = math.sqrt(max(mse_sum / max(pix_sum, 1), 0.0))
    val_psnr = psnr_sum / max(n_img, 1)
    val_ssim = ssim_sum / max(n_img, 1)
    val_lpips = (lpips_sum / max(lpips_n, 1)) if lpips_n > 0 else None

    # 组合准则：crit = (100-PSNR) + 100*LPIPS（LPIPS 缺失时取 0.2 兜底）
    lpips_eff = val_lpips if (val_lpips is not None) else 0.2
    val_crit = (100.0 - float(val_psnr)) + 100.0 * float(lpips_eff)

    return {
        "val_loss":  val_loss,
        "val_mae":   val_mae,
        "val_rmse":  val_rmse,
        "val_psnr":  val_psnr,
        "val_ssim":  val_ssim,
        "val_lpips": val_lpips,
        "val_crit":  val_crit,
        "val_vis_path": last_saved_path
    }

# -------------------- Checkpoint GC --------------------
def gc_ckpts(out_dir: str, keep_n: int, prefix: str = "ckpt_", suffix: str = ".pt"):
    """
    只删除形如 ckpt_<step>.pt 的历史权重，保留最近 keep_n 个；忽略 ckpt_best.pt 等非数字后缀。
    """
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    numbered = []
    for f in os.listdir(out_dir):
        m = pattern.match(f)
        if m:
            step_num = int(m.group(1))
            numbered.append((step_num, f))
    if not numbered:
        return
    numbered.sort(key=lambda x: x[0])
    to_delete = [f for _, f in numbered[:-keep_n]]
    for f in to_delete:
        try:
            os.remove(os.path.join(out_dir, f))
        except Exception as e:
            tlog(f"[warn] failed to remove {f}: {e}")

# -------------------- Dataset builder --------------------
def make_xsense_dataset(meta_path: str, data_cfg: Dict):
    img_size = data_cfg.get("img_size", None)
    if img_size is None:
        raise ValueError("data.img_size 未在配置中指定，请在 ldm_xsense.yaml 的 data 段添加 img_size。")
    root_prefix = data_cfg.get("root_prefix", "")
    return XsenseTactileDataset(meta_path, image_size=img_size, root_prefix=root_prefix)

# -------------------- MAIN --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="ldm_xsense.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg, "r"))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.get("seed", 42))

    # ---- expand ${work_dir} and env vars ----
    def expand_path(p: str) -> str:
        if not isinstance(p, str): return p
        s = p
        if "${work_dir}" in s:
            s = s.replace("${work_dir}", cfg.get("work_dir", ""))
        s = os.path.expandvars(s)
        return s

    # expand & ensure dirs
    cfg["train"]["out_dir"] = expand_path(cfg["train"]["out_dir"])
    cfg["train"]["vis_dir"] = expand_path(cfg["train"]["vis_dir"])
    cfg["log"]["tb_dir"]    = expand_path(cfg["log"]["tb_dir"])
    cfg["log"]["csv_path"]  = expand_path(cfg["log"]["csv_path"])
    cfg["vae"]["ckpt_path"] = expand_path(cfg["vae"]["ckpt_path"])
    cfg["data"]["train_meta"] = expand_path(cfg["data"]["train_meta"])
    cfg["data"]["val_meta"]   = expand_path(cfg["data"]["val_meta"])

    out_dir = ensure_dir(cfg["train"]["out_dir"])
    vis_dir = ensure_dir(cfg["train"]["vis_dir"])
    tb_dir  = ensure_dir(cfg["log"]["tb_dir"])
    ensure_dir(os.path.dirname(cfg["log"]["csv_path"]))

    # Logging
    log_cfg = cfg.get("log", {})
    use_tqdm = log_cfg.get("use_tqdm", True)
    use_tb   = log_cfg.get("use_tb", True) and (SummaryWriter is not None)
    writer = SummaryWriter(tb_dir) if use_tb else None

    # Data
    data_cfg = cfg["data"]
    ds_tr = make_xsense_dataset(data_cfg["train_meta"], data_cfg)
    ds_va = make_xsense_dataset(data_cfg["val_meta"], data_cfg)
    dl_tr = DataLoader(ds_tr, batch_size=data_cfg["batch_size"], shuffle=True,
                       num_workers=data_cfg["num_workers"], pin_memory=data_cfg["pin_memory"], drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=data_cfg["batch_size"], shuffle=False,
                       num_workers=data_cfg["num_workers"], pin_memory=data_cfg["pin_memory"], drop_last=False)

    # VAE
    vae = load_frozen_vae(cfg["vae"], device)

    # Embedder
    emb_cfg = cfg["embedder"]
    embedder = MassTextureCmdEmbedder(
        num_textures=emb_cfg["num_textures"],
        embed_dim=emb_cfg["embed_dim"],
        seq_len=emb_cfg["seq_len"],
        use_cmd=emb_cfg["use_cmd"],
        mass_scale=emb_cfg["mass_scale"],
        mass_log1p=emb_cfg["mass_log1p"],
    ).to(device)

    # UNet
    mcfg = cfg["model"]
    unet = UNetLDM(
        zc=cfg["vae"]["z_channels"],
        base=mcfg["base_channels"],
        mult=tuple(mcfg["channel_mult"]),
        n_res=mcfg["num_res_blocks"],
        t_dim=emb_cfg["embed_dim"],
        cond_dim=emb_cfg["embed_dim"],
        heads=mcfg["num_heads"],
        attn_res=tuple(mcfg["attn_resolutions"]),
        film=mcfg["film"],
        v_pred=mcfg["v_prediction"]
    ).to(device)

    # Optim/EMA/Sched
    params = list(unet.parameters()) + [p for p in embedder.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = make_grad_scaler(cfg["train"]["amp"])
    ema = EMA(unet, decay=cfg["train"]["ema_decay"])
    sched = NoiseScheduler(T=mcfg["timesteps"], kind=mcfg["beta_schedule"]).to(device)

    # Resume
    step = 0
    # best_val 用于存 crit（越小越好）
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        unet.load_state_dict(ckpt["unet"], strict=False)
        embedder.load_state_dict(ckpt["embedder"], strict=False)
        optim.load_state_dict(ckpt["optim"])
        step = ckpt.get("step", 0)
        if "ema" in ckpt: ema.shadow = ckpt["ema"]
        if "best_val" in ckpt: best_val = ckpt["best_val"]
        tlog(f"[resume] from {args.resume} @ step={step}")

    # Train loop
    tr = cfg["train"]
    loss_cfg = cfg.get("loss_cfg", {})
    log_every, val_every, save_every = tr["log_every"], tr["val_every"], tr["save_every"]
    max_steps = tr["max_steps"]
    p_uncond_img = mcfg["p_uncond_image"]
    v_pred = mcfg["v_prediction"]

    while step < max_steps:
        itr = tqdm(dl_tr, dynamic_ncols=True, desc=f"train@{step}", smoothing=0.1, leave=False) if use_tqdm else dl_tr
        for batch in itr:
            start_t = time()
            unet.train(); embedder.train()

            xin  = batch["vae_in_pixel"].to(device)
            xtgt = batch["vae_tgt_pixel"].to(device)
            xcdp = batch["cond_contact_pixel"].to(device)

            with torch.no_grad():
                z0   = vae.encode(xtgt)
                zin  = vae.encode(xin)
                if xcdp.shape[1] == 1 and vae.in_ch == 3:
                    xcdp = xcdp.repeat(1,3,1,1)
                zcdp = vae.encode(xcdp)

            B = z0.size(0)
            t = sched.sample_t(B, device)
            eps = torch.randn_like(z0)
            zt, alpha, sigma, a_bar = sched.q_sample(z0, t, eps)

            mass = batch["mass_value"].to(device).view(B)
            tex  = batch["texture_id"].to(device).view(B)
            cond_tok, _ = embedder(mass, tex, None, cond_drop_prob=emb_cfg["cond_drop_prob"])

            if random.random() < p_uncond_img:
                zin_, zcdp_ = torch.zeros_like(zin), torch.zeros_like(zcdp)
            else:
                zin_, zcdp_ = zin, zcdp

            with autocast_fp16(tr["amp"]):
                pred_c = unet(zt, zin_, zcdp_, t, cond_tok)

                # ---- latent target ----
                if v_pred:
                    v_target = a_bar.sqrt().view(-1,1,1,1)*eps - (1.0 - a_bar).sqrt().view(-1,1,1,1)*z0
                    diff_target = v_target
                else:
                    diff_target = eps

                # ---- Min-SNR weighting + Huber/MSE 选择 ----
                lat_cfg = loss_cfg.get("latent", {})
                use_snr = bool(lat_cfg.get("snr_weighting", True))
                gamma   = float(lat_cfg.get("min_snr_gamma", 5.0))
                ltype   = str(lat_cfg.get("type", "mse")).lower()
                huber_d = float(lat_cfg.get("huber_delta", 0.01))

                if use_snr:
                    w = min_snr_weight(a_bar, gamma, v_pred).view(-1,1,1,1)  # [B,1,1,1]
                else:
                    w = 1.0

                if ltype == "huber":
                    loss_latent = (w * huber_loss(pred_c, diff_target, delta=huber_d, reduction="none")).mean()
                else:
                    loss_latent = (w * (pred_c - diff_target) ** 2).mean()

                loss = loss_latent

                # ---- pixel-space perceptual loss (MAE + LPIPS) with cosine schedule ----
                pix_cfg = loss_cfg.get("pixel", {})
                if bool(pix_cfg.get("enable", True)):
                    # 为了像素损失，构建 z0_hat
                    if v_pred:
                        z0_hat = a_bar.sqrt().view(-1,1,1,1) * zt - (1.0 - a_bar).sqrt().view(-1,1,1,1) * pred_c
                    else:
                        z0_hat = (zt - (1.0 - a_bar).sqrt().view(-1,1,1,1) * pred_c) / (a_bar.sqrt().view(-1,1,1,1) + 1e-8)

                    x_hat = vae.decode(z0_hat).clamp(-1, 1)
                    x_hat = match_spatial_to(x_hat, xtgt)
                    xh = ((x_hat + 1) * 0.5).clamp(0, 1)
                    xt = ((xtgt.clamp(-1, 1) + 1) * 0.5).clamp(0, 1)

                    base_lambda = float(pix_cfg.get("lambda_img", 0.2))
                    sched_type  = str(pix_cfg.get("schedule", "cosine"))
                    warm        = int(pix_cfg.get("warmup_steps", 2000))
                    maxs        = int(pix_cfg.get("max_steps", 20000))
                    if sched_type == "cosine":
                        ramp = cosine_ramp(step, warm, maxs)
                    else:
                        ramp = 1.0

                    lam = base_lambda * ramp
                    if lam > 0:
                        loss_img = F.l1_loss(xh, xt) * float(pix_cfg.get("w_mae", 1.0))
                        if _HAS_LPIPS and float(pix_cfg.get("w_lpips", 0.5)) > 0:
                            net_lp = get_lpips(device)
                            def to3(z): return z if z.size(1)==3 else (z[:, :3] if z.size(1)>3 else z.repeat(1,3,1,1))
                            lp_xh = (to3(xh)*2-1).clamp(-1,1); lp_xt = (to3(xt)*2-1).clamp(-1,1)
                            loss_img = loss_img + float(pix_cfg.get("w_lpips", 0.5)) * net_lp(lp_xh, lp_xt, normalize=False).mean()
                        loss = loss + lam * loss_img

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, tr["grad_clip"])
            scaler.step(optim); scaler.update()
            ema.update()

            if use_tqdm:
                itr.set_postfix_str(f"loss={loss.item():.4f}")

            if writer and (step % log_every == 0):
                dt = max(time() - start_t, 1e-9)
                it_per_s = 1.0 / dt
                gpu_mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
                lr = optim.param_groups[0]["lr"]
                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/it_per_s", it_per_s, step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/gpu_mem_MB", gpu_mem, step)

            if (step % val_every == 0) and (step > 0):
                ema.store(); ema.copy_to()
                metrics = evaluate(
                    dl_va, vae, unet, embedder, sched, v_pred, emb_cfg, device,
                    amp=tr["amp"], vis_dir=tr["vis_dir"], vis_n=tr["vis_n"],
                    writer=writer, step=step
                )
                ema.restore()

                val_loss  = metrics["val_loss"]
                val_mae   = metrics["val_mae"]
                val_rmse  = metrics["val_rmse"]
                val_psnr  = metrics["val_psnr"]
                val_ssim  = metrics["val_ssim"]
                val_lpips = metrics["val_lpips"]
                val_crit  = metrics["val_crit"]

                msg = (f"[val] step={step} loss={val_loss:.4f} "
                       f"MAE={val_mae:.4f} RMSE={val_rmse:.4f} "
                       f"PSNR={val_psnr:.2f} SSIM={val_ssim:.3f}")
                if val_lpips is not None:
                    msg += f" LPIPS={val_lpips:.4f}"
                msg += f" CRIT={val_crit:.3f}"
                tlog(msg)

                if writer:
                    writer.add_scalar("val/loss",  val_loss, step)
                    writer.add_scalar("val/mae",   val_mae,  step)
                    writer.add_scalar("val/rmse",  val_rmse, step)
                    writer.add_scalar("val/psnr",  val_psnr, step)
                    writer.add_scalar("val/ssim",  val_ssim, step)
                    writer.add_scalar("val/crit_psnr_lpips", val_crit, step)
                    if val_lpips is not None:
                        writer.add_scalar("val/lpips", val_lpips, step)

                # 追加到 CSV
                try:
                    csv_path = cfg["log"].get("csv_path", os.path.join(cfg["train"]["out_dir"], "log.csv"))
                    ensure_dir(os.path.dirname(csv_path))
                    write_header = (not os.path.isfile(csv_path))
                    with open(csv_path, "a", newline="") as f:
                        w = csv.writer(f)
                        if write_header:
                            w.writerow(["step","loss","mae","rmse","psnr","ssim","lpips","crit","vis_path"])
                        w.writerow([
                            step, f"{val_loss:.6f}", f"{val_mae:.6f}", f"{val_rmse:.6f}",
                            f"{val_psnr:.4f}", f"{val_ssim:.6f}",
                            (f"{val_lpips:.6f}" if val_lpips is not None else ""),
                            f"{val_crit:.6f}",
                            metrics.get("val_vis_path","")
                        ])
                except Exception as e:
                    tlog(f"[warn] csv write failed: {e}")

                # 用联合准则来挑 best（越小越好）
                if step == val_every and best_val == float("inf"):
                    best_val = val_crit  # 初始化
                if val_crit < best_val:
                    best_val = val_crit
                    best_path = os.path.join(out_dir, "ckpt_best.pt")
                    torch.save({
                        "unet": unet.state_dict(),
                        "embedder": embedder.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                        "ema": ema.shadow,
                        "cfg": cfg,
                        "best_val": best_val
                    }, best_path)
                    tlog(f"[save] best -> {best_path} (crit={best_val:.3f})")

            if (step % save_every == 0) and (step > 0):
                path = os.path.join(out_dir, f"ckpt_{step}.pt")
                torch.save({
                    "unet": unet.state_dict(),
                    "embedder": embedder.state_dict(),
                    "optim": optim.state_dict(),
                    "step": step,
                    "ema": ema.shadow,
                    "cfg": cfg,
                    "best_val": best_val
                }, path)
                gc_ckpts(out_dir, tr["keep_n_checkpoints"])
                tlog(f"[save] {path}")

            step += 1
            if step >= max_steps: break

    final_path = os.path.join(out_dir, "ckpt_final.pt")
    torch.save({
        "unet": unet.state_dict(),
        "embedder": embedder.state_dict(),
        "ema": ema.shadow,
        "cfg": cfg,
        "best_val": best_val
    }, final_path)
    tlog(f"[done] saved {final_path}")

if __name__ == "__main__":
    main()
