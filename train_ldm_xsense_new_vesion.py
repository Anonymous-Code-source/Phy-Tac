# -*- coding: utf-8 -*-
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

# =====  =====
from src.xsense_dataset import XsenseTactileDataset
from src.mt_embedder_cmd import MassTextureCmdEmbedder

# -------------------- Small utils --------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def tlog(msg: str):
    try: tqdm.write(msg)
    except Exception: print(msg)

# -------- AMP  --------
from contextlib import contextmanager
def make_grad_scaler(amp_enabled: bool):
    try:
        return torch.amp.GradScaler(enabled=amp_enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)

@contextmanager
def autocast_fp16(amp_enabled: bool, device_type: str):
    use_amp = bool(amp_enabled) and (device_type == 'cuda')
    try:
        with torch.amp.autocast('cuda', enabled=use_amp):
            yield
    except Exception:
        with torch.cuda.amp.autocast(enabled=use_amp):
            yield


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


def load_frozen_vae(vae_cfg, device):
    import_path = vae_cfg.get("import_path", "train_vae_xsense_SD")
    try:
        vae_mod = importlib.import_module(import_path)
        VAEClass = getattr(vae_mod, "AutoencoderKL_SD")
    except Exception as e:
        raise ImportError(f" {import_path}  AutoencoderKL_SD：{e}")

    vae = VAEClass(
        in_ch=vae_cfg.get("input_channels", 3),
        out_ch=vae_cfg.get("output_channels", 3),
        base_ch=vae_cfg.get("base_ch", 64),
        ch_mult=vae_cfg.get("ch_mult", [1, 2, 4, 4]),
        num_res_blocks=vae_cfg.get("num_res_blocks", 2),
        z_ch=vae_cfg.get("z_channels", 4),
        attn_mid=vae_cfg.get("attn_mid", True),
    )

    # 
    try:
        state = torch.load(vae_cfg["ckpt_path"], map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(vae_cfg["ckpt_path"], map_location="cpu")

    missing, unexpected = vae.load_state_dict(state, strict=True)
    if missing or unexpected:
        tlog(f"[VAE] state_dict keys mismatch -> missing:{missing} unexpected:{unexpected}")

    vae.to(device).eval()
    for p in vae.parameters(): p.requires_grad = False

    # 
    _enc = vae.encode; _dec = vae.decode
    def _encode_wrap(x):
        out = _enc(x)
        if isinstance(out, (tuple, list)): return out[0]
        if isinstance(out, dict): return out.get("z", next(iter(out.values())))
        return out
    def _decode_wrap(z):
        out = _dec(z)
        if isinstance(out, (tuple, list)): return out[0]
        if isinstance(out, dict): return out.get("x", next(iter(out.values())))
        return out
    vae.encode = _encode_wrap
    vae.decode = _decode_wrap

    vae.in_ch = vae_cfg.get("input_channels", 3)
    vae.z_channels = vae_cfg.get("z_channels", 4)
    return vae

# --------------------  --------------------
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

def _safe_gn(ch: int):
    g = 32 if ch >= 32 else max(1, ch)
    while g > 1 and (ch % g) != 0: g //= 2
    return nn.GroupNorm(g, ch)

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, film_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = _safe_gn(in_ch)
        self.norm2 = _safe_gn(out_ch)
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

# -------------------- UNet (Concat + FiLM + Cross-Attn) --------------------
class UNetLDM(nn.Module):
    def __init__(self, zc=4, base=128, mult=(1, 2, 4), n_res=2, t_dim=256,
                 cond_dim=256, heads=4, attn_res=(1,2), film=True, v_pred=True):
        super().__init__()
        self.v_pred = v_pred
        self.tmlp = SinusoidalTimeEmbedding(t_dim)

        in_ch = zc * 3
        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        # Down
        self.downs = nn.ModuleList()
        feats = []; ch = base; attn_set = set(attn_res)
        for i, m in enumerate(mult):
            out = base * m
            blocks = nn.ModuleList([ResBlock(ch if j==0 else out, out, t_dim, cond_dim if film else None)
                                    for j in range(n_res)])
            attn = nn.ModuleList([CrossAttention(out, cond_dim, heads)]) if i in attn_set else None
            down = nn.Conv2d(out, out, 3, stride=2, padding=1) if i < len(mult)-1 else nn.Identity()
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

# --------------------  --------------------
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

class EMA:
    def __init__(self, model, decay=0.9999):
        self.m = model
        self.decay = decay
        self.shadow = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self):
        for (n, p) in self.m.named_parameters():
            if not p.requires_grad: continue
            self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def load_to(self, model):
        sd = model.state_dict()
        for n in sd.keys():
            if n in self.shadow:
                sd[n].copy_(self.shadow[n])

# --------------------  --------------------
def psnr_from_mse(mse): return 10.0 * math.log10(1.0 / max(mse, 1e-12))

def ssim_torch(x, y, C1=0.01**2, C2=0.03**2):
    x = x.float(); y = y.float()
    def _avg3(z):
        z = F.pad(z, (1,1,1,1), mode="replicate")
        return F.avg_pool2d(z, 3, 1, 0)
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
        "mae_sum": mae_sum, "mse_sum": mse_sum, "pix": pix,
        "psnr_sum": psnr * B, "ssim_sum": ssim * B,
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

    def to01(x): return ((x.clamp(-1,1) + 1) * 0.5).clamp(0,1)
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

# --------------------  --------------------
@torch.no_grad()
def ddim_sample(unet, vae, sched: NoiseScheduler, zin, zcdp, cond_tok, steps=200, device=None):
    device = device or zin.device
    B, C, H, W = zin.shape
    zt = torch.randn(B, vae.z_channels, H, W, device=device)
    ts_full = torch.linspace(sched.T-1, 0, steps, device=device).long()
    a_bar_full = sched.alphas_cumprod.to(device)

    for i, t in enumerate(ts_full):
        t_batch = torch.full((B,), int(t.item()), device=device, dtype=torch.long)
        v = unet(zt, zin, zcdp, t_batch, cond_tok)
        a_bar = a_bar_full[t]
        eps = torch.sqrt(1 - a_bar) * v + torch.sqrt(a_bar) * zt
        x0  = (zt - torch.sqrt(1 - a_bar) * eps) / torch.sqrt(a_bar + 1e-8)
        ab_prev = a_bar_full[ts_full[i+1]] if i+1 < len(ts_full) else torch.tensor(1.0, device=device)
        zt = torch.sqrt(ab_prev+1e-8) * x0 + torch.sqrt(1 - ab_prev + 1e-8) * eps

    x_out = vae.decode(zt).clamp(-1, 1)
    return x_out

# --------------------  --------------------
def make_xsense_dataset(meta_path: str, data_cfg: Dict):
    img_size = data_cfg.get("img_size", None)
    if img_size is None:
        raise ValueError("data.img_size。")
    root_prefix = data_cfg.get("root_prefix", "")
    return XsenseTactileDataset(meta_path, image_size=img_size, root_prefix=root_prefix)

@torch.no_grad()
def evaluate(dl_va, vae, unet, embedder, sched, v_pred, emb_cfg, device,
             amp=True, vis_dir=None, vis_n=8, writer=None, step=0, loss_cfg=None):
    unet.eval(); embedder.eval()

    loss_sum = 0.0; n_img = 0
    mae_sum = 0.0; mse_sum = 0.0; pix_sum = 0
    psnr_sum = 0.0; ssim_sum = 0.0
    lpips_sum = 0.0; lpips_n = 0

    first_grid_saved = False
    last_saved_path = None
    lpips_net = get_lpips(device)

    itr = tqdm(dl_va, dynamic_ncols=True, desc=f"val@{step}", smoothing=0.1, leave=False)
    for batch in itr:
        B = batch["vae_tgt_pixel"].size(0)
        xin  = batch["vae_in_pixel"].to(device)
        xtgt = batch["vae_tgt_pixel"].to(device)
        xcdp = batch["cond_contact_pixel"].to(device)
        with torch.no_grad():
            z0   = vae.encode(xtgt)
            zin  = vae.encode(xin)
            if xcdp.shape[1] == 1 and vae.in_ch == 3: xcdp = xcdp.repeat(1,3,1,1)
            zcdp = vae.encode(xcdp)

        t = sched.sample_t(B, device)
        eps = torch.randn_like(z0)
        zt, alpha, sigma, a_bar = sched.q_sample(z0, t, eps)

        mass = batch["mass_value"].to(device).view(B)
        tex  = batch["texture_id"].to(device).view(B)
        cond_tok, _ = embedder(mass, tex, None, cond_drop_prob=emb_cfg.get("cond_drop_prob", 0.0))

        with autocast_fp16(amp, device.type):
            pred = unet(zt, zin, zcdp, t, cond_tok)
            if v_pred:
                diff_target = a_bar.sqrt().view(-1,1,1,1) * eps - (1.0 - a_bar).sqrt().view(-1,1,1,1) * z0
            else:
                diff_target = eps

            lat_cfg = (loss_cfg or {}).get("latent", {})
            use_snr = bool(lat_cfg.get("snr_weighting", True))
            gamma   = float(lat_cfg.get("min_snr_gamma", 2.0))
            ltype   = str(lat_cfg.get("type", "huber")).lower()
            huber_d = float(lat_cfg.get("huber_delta", 0.05))

            snr = (a_bar / (1 - a_bar + 1e-8)).clamp(min=1e-8)
            w = (torch.minimum(torch.full_like(snr, gamma), snr) / snr).sqrt().view(-1,1,1,1) if (use_snr and v_pred) else \
                (torch.minimum(torch.full_like(snr, gamma), snr) / snr).view(-1,1,1,1) if use_snr else 1.0

            if ltype == "huber":
                loss_latent = (w * (torch.where((pred - diff_target).abs() <= huber_d,
                                                0.5 * (pred - diff_target) ** 2 / huber_d,
                                                (pred - diff_target).abs() - 0.5 * huber_d))).mean()
            else:
                loss_latent = (w * (pred - diff_target) ** 2).mean()
            loss = loss_latent

            # pixel loss (MAE+LPIPS) with same schedule
            pix_cfg = (loss_cfg or {}).get("pixel", {})
            if bool(pix_cfg.get("enable", True)):
                if v_pred:
                    z0_hat = a_bar.sqrt().view(-1,1,1,1) * zt - (1.0 - a_bar).sqrt().view(-1,1,1,1) * pred
                else:
                    z0_hat = (zt - (1.0 - a_bar).sqrt().view(-1,1,1,1) * pred) / (a_bar.sqrt().view(-1,1,1,1) + 1e-8)
                x_hat = vae.decode(z0_hat).clamp(-1, 1)

                xh = ((x_hat + 1) * 0.5).clamp(0, 1)
                xt = ((xtgt.clamp(-1, 1) + 1) * 0.5).clamp(0, 1)

                base_lambda = float(pix_cfg.get("lambda_img", 0.4))
                warm        = int(pix_cfg.get("warmup_steps", 800))
                maxs        = int(pix_cfg.get("max_steps", 20000))
                tprog = 1.0 if step >= maxs else (0.0 if step <= warm else 0.5*(1 - math.cos(math.pi * (step-warm)/max(1,(maxs-warm)))))
                lam = base_lambda * tprog
                if lam > 0:
                    loss_img = F.l1_loss(xh, xt) * float(pix_cfg.get("w_mae", 1.5))
                    if _HAS_LPIPS and float(pix_cfg.get("w_lpips", 0.2)) > 0:
                        net_lp = get_lpips(device)
                        def to3(z): return z if z.size(1)==3 else (z[:, :3] if z.size(1)>3 else z.repeat(1,3,1,1))
                        lp_xh = (to3(xh)*2-1).clamp(-1,1); lp_xt = (to3(xt)*2-1).clamp(-1,1)
                        loss_img = loss_img + float(pix_cfg.get("w_lpips", 0.2)) * net_lp(lp_xh, lp_xt, normalize=False).mean()
                    loss = loss + lam * loss_img

        bm = compute_batch_metrics(xh, xt, device, lpips_net=lpips_net)
        loss_sum += loss.item() * B
        mae_sum  += bm["mae_sum"]; mse_sum  += bm["mse_sum"]; pix_sum  += bm["pix"]
        psnr_sum += bm["psnr_sum"]; ssim_sum += bm["ssim_sum"]
        if bm["lpips_sum"] is not None: lpips_sum += bm["lpips_sum"]; lpips_n += bm["b"]
        n_img += B

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

    lpips_eff = val_lpips if (val_lpips is not None) else 0.2
    val_crit = (100.0 - float(val_psnr)) + 100.0 * float(lpips_eff)

    return {
        "val_loss":  val_loss, "val_mae": val_mae, "val_rmse": val_rmse,
        "val_psnr":  val_psnr, "val_ssim": val_ssim, "val_lpips": val_lpips,
        "val_crit":  val_crit, "val_vis_path": last_saved_path
    }

# -------------------- Checkpoint --------------------
def gc_ckpts(out_dir: str, keep_n: int, prefix: str = "ckpt_", suffix: str = ".pt"):
    import re
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
    numbered = []
    for f in os.listdir(out_dir):
        m = pattern.match(f)
        if m: numbered.append((int(m.group(1)), f))
    if not numbered: return
    numbered.sort(key=lambda x: x[0])
    to_delete = [f for _, f in numbered[:-keep_n]]
    for f in to_delete:
        try: os.remove(os.path.join(out_dir, f))
        except Exception as e: tlog(f"[warn] failed to remove {f}: {e}")

# -------------------- MAIN --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="ldm.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.cfg, "r"))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.get("seed", 42))

    # 
    def ex(p: str) -> str:
        if not isinstance(p, str): return p
        s = p.replace("${work_dir}", cfg.get("work_dir", ""))
        return os.path.expandvars(s)

    cfg["train"]["out_dir"] = ex(cfg["train"]["out_dir"])
    cfg["train"]["vis_dir"] = ex(cfg["train"]["vis_dir"])
    cfg["log"]["tb_dir"]    = ex(cfg["log"]["tb_dir"])
    cfg["log"]["csv_path"]  = ex(cfg["log"]["csv_path"])
    cfg["vae"]["ckpt_path"] = ex(cfg["vae"]["ckpt_path"])
    cfg["data"]["train_meta"] = ex(cfg["data"]["train_meta"])
    cfg["data"]["val_meta"]   = ex(cfg["data"]["val_meta"])

    out_dir = ensure_dir(cfg["train"]["out_dir"])
    vis_dir = ensure_dir(cfg["train"]["vis_dir"])
    tb_dir  = ensure_dir(cfg["log"]["tb_dir"])
    ensure_dir(os.path.dirname(cfg["log"]["csv_path"]))

    writer = SummaryWriter(tb_dir) if (cfg["log"]["use_tb"] and SummaryWriter is not None) else None

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
    tr = cfg["train"]; loss_cfg = cfg.get("loss_cfg", {})
    log_every, val_every, save_every = tr["log_every"], tr["val_every"], tr["save_every"]
    max_steps = tr["max_steps"]; p_uncond_img = mcfg["p_uncond_image"]; v_pred = mcfg["v_prediction"]

    while step < max_steps:
        itr = tqdm(dl_tr, dynamic_ncols=True, desc=f"train@{step}", smoothing=0.1, leave=False)
        for batch in itr:
            start_t = time()
            unet.train(); embedder.train()

            xin  = batch["vae_in_pixel"].to(device)
            xtgt = batch["vae_tgt_pixel"].to(device)
            xcdp = batch["cond_contact_pixel"].to(device)

            with torch.no_grad():
                z0   = vae.encode(xtgt)
                zin  = vae.encode(xin)
                if xcdp.shape[1] == 1 and vae.in_ch == 3: xcdp = xcdp.repeat(1,3,1,1)
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

            with autocast_fp16(tr["amp"], device.type):
                pred_c = unet(zt, zin_, zcdp_, t, cond_tok)

                # latent target
                if v_pred:
                    diff_target = a_bar.sqrt().view(-1,1,1,1) * eps - (1.0 - a_bar).sqrt().view(-1,1,1,1) * z0
                else:
                    diff_target = eps

                # Min-SNR + Huber
                lat_cfg = loss_cfg.get("latent", {})
                use_snr = bool(lat_cfg.get("snr_weighting", True))
                gamma   = float(lat_cfg.get("min_snr_gamma", 2.0))
                huber_d = float(lat_cfg.get("huber_delta", 0.05))

                snr = (a_bar / (1 - a_bar + 1e-8)).clamp(min=1e-8)
                w = (torch.minimum(torch.full_like(snr, gamma), snr) / snr).sqrt().view(-1,1,1,1) if (use_snr and v_pred) else \
                    (torch.minimum(torch.full_like(snr, gamma), snr) / snr).view(-1,1,1,1) if use_snr else 1.0

                loss_latent = (w * (torch.where((pred_c - diff_target).abs() <= huber_d,
                                                0.5 * (pred_c - diff_target) ** 2 / huber_d,
                                                (pred_c - diff_target).abs() - 0.5 * huber_d))).mean()
                loss = loss_latent

                # pixel loss (MAE + LPIPS) with cosine ramp
                pix_cfg = loss_cfg.get("pixel", {})
                if bool(pix_cfg.get("enable", True)):
                    if v_pred:
                        z0_hat = a_bar.sqrt().view(-1,1,1,1) * zt - (1.0 - a_bar).sqrt().view(-1,1,1,1) * pred_c
                    else:
                        z0_hat = (zt - (1.0 - a_bar).sqrt().view(-1,1,1,1) * pred_c) / (a_bar.sqrt().view(-1,1,1,1) + 1e-8)
                    x_hat = vae.decode(z0_hat).clamp(-1, 1)

                    xh = ((x_hat + 1) * 0.5).clamp(0, 1)
                    xt = ((xtgt.clamp(-1, 1) + 1) * 0.5).clamp(0, 1)

                    base_lambda = float(pix_cfg.get("lambda_img", 0.4))
                    warm        = int(pix_cfg.get("warmup_steps", 800))
                    maxs        = int(pix_cfg.get("max_steps", 20000))
                    tprog = 1.0 if step >= maxs else (0.0 if step <= warm else 0.5*(1 - math.cos(math.pi * (step-warm)/max(1,(maxs-warm)))))
                    lam = base_lambda * tprog
                    if lam > 0:
                        loss_img = F.l1_loss(xh, xt) * float(pix_cfg.get("w_mae", 1.5))
                        if _HAS_LPIPS and float(pix_cfg.get("w_lpips", 0.2)) > 0:
                            net_lp = get_lpips(device)
                            def to3(z): return z if z.size(1)==3 else (z[:, :3] if z.size(1)>3 else z.repeat(1,3,1,1))
                            lp_xh = (to3(xh)*2-1).clamp(-1,1); lp_xt = (to3(xt)*2-1).clamp(-1,1)
                            loss_img = loss_img + float(pix_cfg.get("w_lpips", 0.2)) * net_lp(lp_xh, lp_xt, normalize=False).mean()
                        loss = loss + lam * loss_img

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, tr["grad_clip"])
            scaler.step(optim); scaler.update()
            ema.update()

            itr.set_postfix_str(f"loss={loss.item():.4f}")

            # \
            if (step % val_every == 0) and (step > 0):
                # 
                bk = {k: v.detach().clone() for k, v in unet.state_dict().items()}
                ema.load_to(unet)
                metrics = evaluate(
                    dl_va, vae, unet, embedder, sched, v_pred, emb_cfg, device,
                    amp=tr["amp"], vis_dir=cfg["train"]["vis_dir"], vis_n=cfg["train"]["vis_n"],
                    writer=writer, step=step, loss_cfg=loss_cfg
                )
                unet.load_state_dict(bk)  # 

                val_loss, val_psnr = metrics["val_loss"], metrics["val_psnr"]
                val_mae,  val_rmse = metrics["val_mae"], metrics["val_rmse"]
                val_ssim, val_lpips, val_crit = metrics["val_ssim"], metrics["val_lpips"], metrics["val_crit"]
                msg = (f"[val] step={step} loss={val_loss:.4f} PSNR={val_psnr:.2f} "
                       f"SSIM={val_ssim:.4f} MAE={val_mae:.4f} RMSE={val_rmse:.4f}")
                if val_lpips is not None: msg += f" LPIPS={val_lpips:.4f}"
                msg += f" CRIT={(val_crit):.3f}"
                tlog(msg)

                # 
                if writer:
                    writer.add_scalar("val/loss", val_loss, step)
                    writer.add_scalar("val/psnr", val_psnr, step)
                    writer.add_scalar("val/ssim", val_ssim, step)
                    writer.add_scalar("val/mae", val_mae, step)
                    writer.add_scalar("val/rmse", val_rmse, step)
                    if val_lpips is not None: writer.add_scalar("val/lpips", val_lpips, step)
                    writer.add_scalar("val/crit", val_crit, step)

                csv_path = cfg["log"].get("csv_path", os.path.join(cfg["train"]["out_dir"], "log.csv"))
                try:
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

                # 
                lpips_eff = val_lpips if (val_lpips is not None) else 0.2
                crit_eff = (100.0 - float(val_psnr)) + 100.0 * float(lpips_eff)
                if crit_eff < best_val:
                    best_val = crit_eff
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

            # 
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
                gc_ckpts(out_dir, cfg["train"]["keep_n_checkpoints"])
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
