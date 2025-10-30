#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json, math, csv
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- TensorBoard ----
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_OK = True
except Exception:
    _TB_OK = False
    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            print("[WARN] tensorboard not found; proceeding without TB logging. "
                  "To enable: pip install tensorboard && then run: tensorboard --logdir <work_dir>")
        def add_scalar(self, *a, **k): pass
        def add_image(self, *a, **k): pass
        def close(self): pass

from torchvision.utils import save_image, make_grid

# ---- LPIPS ----
try:
    import lpips  # pip install lpips
    _LPIPS_OK = True
except Exception:
    _LPIPS_OK = False
    lpips = None

from tqdm import tqdm

# 
try:
    import yaml
except Exception as e:
    raise RuntimeError("please install pyyaml：pip install pyyaml")

CONFIG_PATH = Path("vae.yaml")
DEBUG_DIR_NAME = "debug_nan"

# ----  ----
ROOT = Path(__file__).parent.resolve()
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xsense_dataset import XsenseTactileDataset  # noqa: E402
from mt_embedder_cmd import MassTextureCmdEmbedder  # noqa: E402

# ------------------------------
# Utils
# ------------------------------

def set_seed(seed: int = 2024):
    import random
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def psnr(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    x = (img1.clamp(-1, 1) + 1) / 2
    y = (img2.clamp(-1, 1) + 1) / 2
    mse = F.mse_loss(x, y, reduction='mean')
    if mse == 0:
        return torch.tensor(99.0, device=img1.device)
    return 10 * torch.log10(1.0 / mse)

# ---- SSIM ----
try:
    from torchmetrics.functional.image.ssim import structural_similarity_index_measure as tm_ssim
    _TM_SSIM_OK = True
except Exception:
    _TM_SSIM_OK = False

try:
    from skimage.metrics import structural_similarity as sk_ssim
    _SKIMAGE_SSIM_OK = True
except Exception:
    _SKIMAGE_SSIM_OK = False


def ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    x = (img1.clamp(-1, 1) + 1) / 2
    y = (img2.clamp(-1, 1) + 1) / 2
    if _TM_SSIM_OK:
        return tm_ssim(x, y, data_range=1.0)
    if _SKIMAGE_SSIM_OK:
        vals = []
        xb = x.detach().cpu(); yb = y.detach().cpu()
        for i in range(xb.size(0)):
            xi = xb[i].permute(1,2,0).numpy(); yi = yb[i].permute(1,2,0).numpy()
            vals.append(sk_ssim(xi, yi, channel_axis=2, data_range=1.0))
        return torch.tensor(float(sum(vals)/max(1,len(vals))), device=img1.device, dtype=x.dtype)
    C1, C2 = 0.01**2, 0.03**2
    B, C, H, W = x.shape
    g = torch.tensor([math.exp(-(i-5)**2/(2*1.5**2)) for i in range(11)], device=x.device)
    g = (g / g.sum()).float()
    w2d = g.unsqueeze(1) @ g.unsqueeze(0)
    window = w2d.expand(C,1,11,11).contiguous()
    mu_x = F.conv2d(x, window, padding=5, groups=C)
    mu_y = F.conv2d(y, window, padding=5, groups=C)
    mu_x2, mu_y2 = mu_x*mu_x, mu_y*mu_y
    mu_xy = mu_x*mu_y
    sigma_x2 = F.conv2d(x*x, window, padding=5, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y*y, window, padding=5, groups=C) - mu_y2
    sigma_xy = F.conv2d(x*y, window, padding=5, groups=C) - mu_xy
    ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2))/((mu_x2+mu_y2+C1)*(sigma_x2+sigma_y2+C2))
    return ssim_map.mean()

# ------------------------------
# Stable Diffusion
# ------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, gn_groups: int = 32):
        super().__init__()
        self.in_ch = in_ch; self.out_ch = out_ch
        self.norm1 = nn.GroupNorm(min(gn_groups, in_ch), in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(gn_groups, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (in_ch != out_ch)
        if self.skip:
            self.conv_skip = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        if self.skip:
            x = self.conv_skip(x)
        return x + h

class AttnBlock(nn.Module):
    """"""
    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        h_ = self.norm(x)
        q = self.q(h_).reshape(b, c, h*w).transpose(1,2)   # (b, hw, c)
        k = self.k(h_).reshape(b, c, h*w)                  # (b, c, hw)
        v = self.v(h_).reshape(b, c, h*w).transpose(1,2)   # (b, hw, c)
        attn = torch.softmax(torch.bmm(q, k) / math.sqrt(c), dim=-1)  # (b, hw, hw)
        h_attn = torch.bmm(attn, v).transpose(1,2).reshape(b, c, h, w)
        return x + self.proj(h_attn)

class Down(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x):
        return self.op(x)

class Up(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x):
        return self.op(x)

class EncoderSD(nn.Module):
    def __init__(self, in_ch: int, base_ch: int, ch_mult: List[int], num_res_blocks: int, z_ch: int, attn_mid: bool):
        super().__init__()
        chs = [base_ch * m for m in ch_mult]
        self.stem = nn.Conv2d(in_ch, chs[0], 3, padding=1)
        blocks = []
        in_c = chs[0]
        for i, c in enumerate(chs):
            blocks.append(ResBlock(in_c, c))
            in_c = c
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(in_c, in_c))
            if i < len(chs) - 1:
                blocks.append(Down(in_c))
        self.down = nn.Sequential(*blocks)
        mid_c = chs[-1]
        mid = [ResBlock(mid_c, mid_c)]
        if attn_mid:
            mid.append(AttnBlock(mid_c))
        mid.append(ResBlock(mid_c, mid_c))
        self.mid = nn.Sequential(*mid)
        self.to_mu = nn.Conv2d(mid_c, z_ch, 3, padding=1)
        self.to_logvar = nn.Conv2d(mid_c, z_ch, 3, padding=1)
    def forward(self, x):
        h = self.stem(x)
        h = self.down(h)
        h = self.mid(h)
        return self.to_mu(h), self.to_logvar(h)

class DecoderSD(nn.Module):
    def __init__(self, out_ch: int, base_ch: int, ch_mult: List[int], num_res_blocks: int, z_ch: int, attn_mid: bool):
        super().__init__()
        chs = [base_ch * m for m in ch_mult]
        mid_c = chs[-1]
        self.in_conv = nn.Conv2d(z_ch, mid_c, 3, padding=1)
        mid = [ResBlock(mid_c, mid_c)]
        if attn_mid:
            mid.append(AttnBlock(mid_c))
        mid.append(ResBlock(mid_c, mid_c))
        self.mid = nn.Sequential(*mid)
        blocks = []
        in_c = mid_c
        for i in reversed(range(len(chs))):
            c = chs[i]
            blocks.append(ResBlock(in_c, c))
            in_c = c
            for _ in range(num_res_blocks - 1):
                blocks.append(ResBlock(in_c, in_c))
            if i > 0:
                blocks.append(Up(in_c))
        self.up = nn.Sequential(*blocks)
        self.out = nn.Sequential(nn.GroupNorm(32, chs[0]), nn.SiLU(), nn.Conv2d(chs[0], out_ch, 3, padding=1), nn.Tanh())
    def forward(self, z):
        h = self.in_conv(z)
        h = self.mid(h)
        h = self.up(h)
        return self.out(h)

class AutoencoderKL_SD(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base_ch=64, ch_mult: List[int] = [1,2,4], num_res_blocks: int = 2,
                 z_ch: int = 2, attn_mid: bool = True):
        super().__init__()
        self.enc = EncoderSD(in_ch, base_ch, ch_mult, num_res_blocks, z_ch, attn_mid)
        self.dec = DecoderSD(out_ch, base_ch, ch_mult, num_res_blocks, z_ch, attn_mid)
        self.downs = len(ch_mult)  # f = 2**downs
    @staticmethod
    def _safe_reparameterize(mu: torch.Tensor, logvar: torch.Tensor):
        dtype = mu.dtype
        mu32 = mu.float(); lv32 = logvar.float().clamp_(-30.0, 20.0)
        std = torch.exp(0.5 * lv32); eps = torch.randn_like(std)
        z32 = mu32 + eps * std
        return z32.to(dtype), mu32.to(dtype), lv32.to(dtype)
    def encode(self, x):
        mu, logvar = self.enc(x)
        z, mu, logvar = self._safe_reparameterize(mu, logvar)
        return z, mu, logvar
    def decode(self, z):
        return self.dec(z)
    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int):
        _, _, h, w = x.shape
        nh = ((h + multiple - 1) // multiple) * multiple
        nw = ((w + multiple - 1) // multiple) * multiple
        ph, pw = nh - h, nw - w
        if ph == 0 and pw == 0:
            return x, 0, 0
        return F.pad(x, (0, pw, 0, ph)), ph, pw
    @staticmethod
    def _crop_to(x: torch.Tensor, h: int, w: int):
        return x[..., :h, :w]
    def forward(self, x):
        B, C, H, W = x.shape
        x_pad, _, _ = self._pad_to_multiple(x, 2 ** self.downs)  # 
        z, mu, logvar = self.encode(x_pad)
        x_rec_pad = self.decode(z)
        x_rec = self._crop_to(x_rec_pad, H, W)
        return x_rec, mu, logvar

# ------------------------------
# training and val
# ------------------------------

def kld_loss(mu, logvar):
    logvar = logvar.clamp(-30.0, 20.0)
    return 0.5 * torch.mean(torch.exp(logvar) + mu**2 - 1.0 - logvar)


def _dump_debug_batch(work_dir: Path, epoch: int, gstep: int, batch: Dict[str, torch.Tensor], x: torch.Tensor,
                      x_rec: torch.Tensor|None, mu: torch.Tensor|None, logvar: torch.Tensor|None):
    try:
        dd = work_dir / DEBUG_DIR_NAME / f"e{epoch:03d}_gs{gstep}"
        dd.mkdir(parents=True, exist_ok=True)
        def _stats(t: torch.Tensor):
            return dict(shape=list(t.shape), min=float(torch.nanmin(t).item()), max=float(torch.nanmax(t).item()),
                        mean=float(torch.nanmean(t).item()), finite=bool(torch.isfinite(t).all()))
        info = {
            'x_stats': _stats(x.detach()),
            'x_rec_stats': _stats(x_rec.detach()) if x_rec is not None else None,
            'mu_stats': _stats(mu.detach()) if mu is not None else None,
            'logvar_stats': _stats(logvar.detach()) if logvar is not None else None,
            'time': datetime.now().isoformat(timespec='seconds')
        }
        with open(dd / 'stats.json', 'w') as f:
            json.dump(info, f, indent=2)
        xv = (x.clamp(-1,1)+1)/2
        if x_rec is not None:
            xr = (x_rec.clamp(-1,1)+1)/2
            k = min(4, xv.size(0))
            grid = make_grid(torch.cat([xv[:k], xr[:k]], dim=0), nrow=k)
            save_image(grid, dd / 'io_grid.png')
        else:
            save_image(xv[:4], dd / 'x.png')
        stems = batch.get('stem', None)
        if stems is not None:
            with open(dd / 'stems.txt', 'w') as f:
                for s in stems:
                    f.write((s if isinstance(s, str) else str(s)) + '\n')
    except Exception as e:
        print(f"[WARN] debug dump failed: {e}")


def main():
    assert CONFIG_PATH.exists(), f"can not find: {CONFIG_PATH}"
    try:
        with open(CONFIG_PATH, 'r') as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"[ERROR] : {CONFIG_PATH}\n{e}\n\n")
        raise
    best_by = str(cfg.get('best_by', 'auto')).lower()  # 'auto' | 'lpips' | 'loss'
    # ----------   ----------
    save_last = bool(cfg.get('save_last', True))
    export_model_only = bool(cfg.get('export_model_only', True))
    export_best_name = str(cfg.get('export_best_name', 'vae_best_model.pt'))
    export_last_name = str(cfg.get('export_last_name', 'vae_last_model.pt'))

    set_seed(int(cfg.get('seed', 2024)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    work_dir = Path(cfg['work_dir']).expanduser().resolve()
    ckpt_dir = work_dir / 'ckpt'
    (ckpt_dir).mkdir(parents=True, exist_ok=True)
    (work_dir / 'samples').mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(work_dir / 'config_snapshot.json', 'w'), indent=2)

    debug_on_nan = bool(cfg.get('debug_on_nan', True))

    #  
    ds_args = dict(image_size=(cfg['image_h'], cfg['image_w']),
                   root_prefix=cfg['data_root_prefix'], vocab_dir=cfg['work_dir'])
    train_ds = XsenseTactileDataset(meta_file=cfg['train_meta'], **ds_args)
    val_ds   = XsenseTactileDataset(meta_file=cfg['val_meta'],   **ds_args)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False,
                              num_workers=cfg['num_workers'], pin_memory=True, drop_last=False)

    #  
    model = AutoencoderKL_SD(
        in_ch=3,
        out_ch=3,
        base_ch=cfg['base_ch'],
        ch_mult=cfg['ch_mult'],
        num_res_blocks=cfg['num_res_blocks'],
        z_ch=cfg['z_ch'],
        attn_mid=bool(cfg.get('attn_mid', True)),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']),
                            weight_decay=cfg['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg['amp']))
    writer = SummaryWriter(log_dir=str(work_dir))

    # LPIPS 
    if _LPIPS_OK:
        lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
        print('[INFO] LPIPS is ready。')
    else:
        lpips_fn = None
        print('[WARN] not install lpips')

    # ---------------- Train ----------------
    def train_one_epoch(epoch: int, global_step: int):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train E{epoch}")
        for batch in pbar:
            batch = to_device(batch, device)
            x = batch['vae_in_pixel']

            with torch.amp.autocast('cuda', enabled=bool(cfg['amp'])):
                x_rec, mu, logvar = model(x)
                rec_l1 = F.l1_loss(x_rec, x)
                kl_w = min(1.0, epoch / max(1, cfg['kl_warmup_epochs'])) * cfg['kl_weight']
                loss = rec_l1 + kl_w * kld_loss(mu, logvar)
                # ==================   ==================
                fm_w_cfg = float(cfg.get('fm_weight', 0.0))
                use_lpips = (
                    str(cfg.get('fm_kind', 'none')).lower() == 'lpips'
                    and fm_w_cfg > 0.0
                    and (lpips_fn is not None)
                )
                fm_loss = None
                if use_lpips and (global_step % int(cfg.get('fm_every_n_steps', 1)) == 0):
                    fm_warmup_epochs = int(cfg.get('fm_warmup_epochs', 0) or 0)
                    fm_scale = min(1.0, epoch / max(1, fm_warmup_epochs)) if fm_warmup_epochs > 0 else 1.0
                    fm_w = fm_w_cfg * fm_scale
                    fm_loss = lpips_fn(x_rec, x).mean()
                    loss = loss + fm_w * fm_loss
                # ======================================================================

            if not torch.isfinite(loss):
                print('[WARN] non-finite loss; skip this batch')
                if debug_on_nan:
                    _dump_debug_batch(work_dir, epoch, global_step, batch, x, x_rec, mu, logvar)
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            # loss
            if (global_step % cfg['log_interval']) == 0:
                writer.add_scalar('train/loss', float(loss.item()), global_step)
                #  
                try:
                    _ = fm_loss
                    if fm_loss is not None:
                        writer.add_scalar('train/fm_lpips', float(fm_loss.item()), global_step)
                        if 'fm_w' in locals():
                            writer.add_scalar('train/fm_weight_eff', float(fm_w), global_step)
                except NameError:
                    pass
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1
        return global_step

    # ---------------- Val ----------------
    def validate(epoch: int):
        model.eval()
        val_loss = val_psnr = val_ssim = val_mae01 = val_rmse01 = 0.0
        lpips_sum = 0.0; lpips_batches = 0
        n_batches = 0
        last_batch_imgs = None

        vis_dir = work_dir / 'val_samples' / f'epoch{epoch:03d}'
        vis_dir.mkdir(parents=True, exist_ok=True)
        save_k = int(cfg.get('val_save_n', 8))
        saved = 0

        pbar = tqdm(val_loader, desc=f"Val   E{epoch}", leave=True)
        with torch.no_grad():
            for batch in pbar:
                batch = to_device(batch, device)
                x = batch['vae_in_pixel']
                x_rec, mu, logvar = model(x)
                rec_l1 = F.l1_loss(x_rec, x)
                loss = rec_l1 + cfg['kl_weight'] * kld_loss(mu, logvar)

                x01  = (x.clamp(-1,1) + 1) / 2
                xr01 = (x_rec.clamp(-1,1) + 1) / 2
                _mae = F.l1_loss(xr01, x01).item()
                _mse = F.mse_loss(xr01, x01).item()
                _rmse = math.sqrt(_mse)
                _psnr = psnr(x_rec, x).item()
                _ssim = float(ssim(x_rec, x).item())
                if lpips_fn is not None:
                    _lpips = float(lpips_fn(x_rec, x).mean().item())
                    lpips_sum += _lpips
                    lpips_batches += 1

                val_loss += float(loss.item())
                val_mae01 += _mae; val_rmse01 += _rmse
                val_psnr  += _psnr; val_ssim   += _ssim
                n_batches += 1
                last_batch_imgs = (x, x_rec)

                if saved < save_k:
                    b = min(x.size(0), save_k - saved)
                    for i in range(b):
                        pair = torch.stack([x01[i], xr01[i]], dim=0)
                        save_image(make_grid(pair, nrow=2), vis_dir / f'sample_{saved+i:03d}.png')
                    saved += b

                n = max(1, n_batches)
                postfix = {
                    'MAE': f"{val_mae01/n:.4f}",
                    'RMSE': f"{val_rmse01/n:.4f}",
                    'PSNR': f"{val_psnr/n:.2f}",
                    'SSIM': f"{val_ssim/n:.4f}",
                }
                if lpips_batches > 0:
                    postfix['LPIPS'] = f"{lpips_sum/lpips_batches:.4f}"
                pbar.set_postfix(postfix)

        n = max(1, n_batches)
        val_loss  /= n
        val_mae01 /= n
        val_rmse01/= n
        val_psnr  /= n
        val_ssim  /= n

        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/mae',  val_mae01, epoch)
        writer.add_scalar('val/rmse', val_rmse01, epoch)
        writer.add_scalar('val/psnr', val_psnr, epoch)
        writer.add_scalar('val/ssim', val_ssim, epoch)
        lpips_avg = None
        if lpips_batches > 0:
            lpips_avg = lpips_sum / lpips_batches
            writer.add_scalar('val/lpips', lpips_avg, epoch)

        #  
        csv_path = work_dir / 'val_metrics.csv'
        fields = ['epoch','timestamp','loss','mae','rmse','psnr','ssim','lpips']
        row = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'loss': round(val_loss, 6),
            'mae': round(val_mae01, 6),
            'rmse': round(val_rmse01, 6),
            'psnr': round(val_psnr, 4),
            'ssim': round(val_ssim, 6),
            'lpips': round(lpips_sum / lpips_batches, 6) if lpips_batches > 0 else ''
        }
        file_exists = csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                w.writeheader()
            w.writerow(row)

        if last_batch_imgs is not None:
            x, x_rec = last_batch_imgs
            x_vis  = (x.clamp(-1,1) + 1) / 2
            xr_vis = (x_rec.clamp(-1,1) + 1) / 2
            grid = make_grid(torch.cat([x_vis, xr_vis], dim=0), nrow=x_vis.shape[0])
            save_image(grid, work_dir / 'samples' / f'epoch{epoch:03d}.png')
            writer.add_image('val/sample_grid', grid, epoch)

        return val_loss, lpips_avg

    #  
    best_metric = float('inf'); global_step = 0
    for epoch in range(1, int(cfg['epochs']) + 1):
        global_step = train_one_epoch(epoch, global_step)
        val_loss, lpips_avg = validate(epoch)

        #  
        use_lpips = False
        if best_by == 'lpips':
            use_lpips = (lpips_avg is not None)
            # if not use_lpips:
            #     print("[WARN] ")
        elif best_by == 'auto':
            use_lpips = (lpips_avg is not None)
        else:  # 'loss'
            use_lpips = False

        current_metric = lpips_avg if use_lpips else val_loss
        metric_name = 'LPIPS' if use_lpips else 'val_loss'

        # -----------  -----------
        state = {
            'epoch': epoch,
            'global_step': global_step,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
            'scaler': scaler.state_dict(),
            'cfg': cfg,
            'best_val': best_metric
        }

        #  
        if current_metric < best_metric:
            best_metric = current_metric
            torch.save(state, ckpt_dir / 'best.pt')  # 
            if export_model_only:
                torch.save(model.state_dict(), ckpt_dir / export_best_name)  # 
            print(f"[INFO] New best ({metric_name}) @ epoch {epoch}: {best_metric:.6f}")

        #  
        if bool(cfg.get('save_last', True)):
            torch.save(state, ckpt_dir / 'last.pt')  #  
            if export_model_only:
                torch.save(model.state_dict(), ckpt_dir / export_last_name)  #  

        #  
        if (epoch % int(cfg['save_every'])) == 0:
            torch.save(state, ckpt_dir / f'epoch{epoch:03d}.pt')

    writer.close()
    print("Training finished. Best val loss:", best_metric)

if __name__ == '__main__':
    main()