
#!/usr/bin/env python3
"""
infer_vae_scores.py

Load the trained VAE (as defined in train_vae_xsense_SD_new_loss.py),
encode two images to latent codes, then aggregate into scalars d_test and d_target.

Usage:
  python infer_vae_scores.py --config infer_vae_scores.yaml \
         --x_test path/to/test.jpg --x_target path/to/target.jpg
"""
import argparse
import importlib.util
import sys
import os
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from PIL import Image

# ----------------------------
# Utilities
# ----------------------------
def load_module_from_path(py_path: Path, module_name: str):
    """Dynamically import a module from a given file path."""
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def load_yaml(yaml_path: Path) -> dict:
    try:
        import yaml
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required to read the config. Please install it via `pip install pyyaml`."
        ) from e
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> FloatTensor in [-1, 1], shape (C,H,W)."""
    x = torch.from_numpy(
        (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
         .view(img.size[1], img.size[0], len(img.getbands()))
         .numpy()).copy()
    )
    x = x.float() / 255.0
    x = x.permute(2, 0, 1)  # C,H,W
    x = x * 2 - 1
    return x

def load_image_as_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return pil_to_tensor(img)

# ----------------------------
# Weighting schemes
# ----------------------------
def score_channel_weighted(z: torch.Tensor, channel_weights: torch.Tensor) -> torch.Tensor:
    """
    z: (B, C, H, W)
    channel_weights: (C,)
    returns: (B,)
    """
    z_mean = z.mean(dim=(2, 3))  # (B, C)
    w = channel_weights / (channel_weights.abs().sum() + 1e-8)
    return (z_mean * w.unsqueeze(0)).sum(dim=1)

def score_spatial_weighted(z: torch.Tensor, spatial_weights: torch.Tensor) -> torch.Tensor:
    """
    z: (B, C, H, W)
    spatial_weights: (H, W)
    returns: (B,)
    """
    z_mean_ch = z.mean(dim=1)  # (B, H, W)
    W = spatial_weights / (spatial_weights.abs().sum() + 1e-8)
    return (z_mean_ch * W.unsqueeze(0)).sum(dim=(1, 2))

def make_center_weight(h: int, w: int, sigma_scale: float = 0.1, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    A soft center prior over spatial positions, default sigma depends on area.
    """
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    sigma2 = sigma_scale * (h * w)
    return torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (sigma2 + 1e-8))

# ----------------------------
# Model loading
# ----------------------------
def load_vae_from_cfg(cfg: dict, device: torch.device):
    """
    Instantiate AutoencoderKL_SD using architecture in cfg and load weights.
    """
    # Import the training module where AutoencoderKL_SD is defined
    train_py = Path(cfg["program"]["train_script_path"])
    if not train_py.exists():
        # fallback: try relative to current working directory
        candi = Path.cwd() / train_py.name
        if candi.exists():
            train_py = candi
        else:
            raise FileNotFoundError(f"Training script not found at: {cfg['program']['train_script_path']}")

    train_mod = load_module_from_path(train_py, "train_vae_xsense_SD_new_loss")
    AutoencoderKL_SD = getattr(train_mod, "AutoencoderKL_SD")

    arch = cfg["model"]["arch"]
    vae = AutoencoderKL_SD(
        in_ch=arch.get("in_ch", 3),
        out_ch=arch.get("out_ch", 3),
        base_ch=arch.get("base_ch", 64),
        ch_mult=arch.get("ch_mult", [1, 2, 4]),
        num_res_blocks=arch.get("num_res_blocks", 2),
        z_ch=arch.get("z_ch", 2),
        attn_mid=arch.get("attn_mid", True),
    ).to(device).eval()

    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    candidates = [
        ckpt_dir / "vae_best_model.pt",
        ckpt_dir / "vae_last_model.pt",
        ckpt_dir / "best.pt",
        ckpt_dir / "last.pt",
    ]
    state_dict = None
    for p in candidates:
        if p.exists():
            obj = torch.load(p, map_location=device)
            if isinstance(obj, dict) and "model" in obj and any(k.startswith("encoder") or k.startswith("decoder") for k in obj["model"].keys()):
                state_dict = obj["model"]
            else:
                state_dict = obj
            break
    if state_dict is None:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}. Tried: {', '.join(map(str, candidates))}")
    vae.load_state_dict(state_dict, strict=True)
    return vae

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="YAML config for inference")
    ap.add_argument("--x_test", type=str, required=True, help="Path to test image")
    ap.add_argument("--x_target", type=str, required=True, help="Path to target image")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))

    # Device
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    torch.set_grad_enabled(False)

    # Load model
    vae = load_vae_from_cfg(cfg, device)

    # Load images
    xt = load_image_as_tensor(Path(args.x_test)).unsqueeze(0).to(device)     # (1,3,H,W)
    xg = load_image_as_tensor(Path(args.x_target)).unsqueeze(0).to(device)   # (1,3,H,W)

    # Encode
    z_test, _, _ = vae.encode(xt)
    z_targ, _, _ = vae.encode(xg)

    # Choose weighting
    method = cfg["scoring"]["method"]
    if method == "channel":
        C = z_test.shape[1]
        if cfg["scoring"].get("channel_weights_path"):
            w = torch.load(cfg["scoring"]["channel_weights_path"], map_location=device).to(device)
            if w.ndim > 1:
                w = w.view(-1)
        else:
            init = cfg["scoring"].get("channel_weights_init", "ones")
            if init == "ones":
                w = torch.ones(C, device=device)
            elif init == "zeros":
                w = torch.zeros(C, device=device)
            elif init == "randn":
                w = torch.randn(C, device=device)
            else:
                raise ValueError(f"Unknown channel_weights_init: {init}")
        d_test = score_channel_weighted(z_test, w)[0].item()
        d_target = score_channel_weighted(z_targ, w)[0].item()

    elif method == "spatial":
        H, W = z_test.shape[2:]
        if cfg["scoring"].get("spatial_weights_path"):
            Wmap = torch.load(cfg["scoring"]["spatial_weights_path"], map_location=device).to(device)
        else:
            # build default center prior
            sigma_scale = float(cfg["scoring"].get("spatial_sigma_scale", 0.1))
            Wmap = make_center_weight(H, W, sigma_scale=sigma_scale, device=device)
        d_test = score_spatial_weighted(z_test, Wmap)[0].item()
        d_target = score_spatial_weighted(z_targ, Wmap)[0].item()

    elif method == "hybrid":
        # first apply channel weights, then spatial weights
        C = z_test.shape[1]
        H, W = z_test.shape[2:]
        # channel
        if cfg["scoring"].get("channel_weights_path"):
            w = torch.load(cfg["scoring"]["channel_weights_path"], map_location=device).to(device)
            if w.ndim > 1:
                w = w.view(-1)
        else:
            init = cfg["scoring"].get("channel_weights_init", "ones")
            if init == "ones":
                w = torch.ones(C, device=device)
            elif init == "zeros":
                w = torch.zeros(C, device=device)
            elif init == "randn":
                w = torch.randn(C, device=device)
            else:
                raise ValueError(f"Unknown channel_weights_init: {init}")
        # spatial
        if cfg["scoring"].get("spatial_weights_path"):
            Wmap = torch.load(cfg["scoring"]["spatial_weights_path"], map_location=device).to(device)
        else:
            sigma_scale = float(cfg["scoring"].get("spatial_sigma_scale", 0.1))
            Wmap = make_center_weight(H, W, sigma_scale=sigma_scale, device=device)

        def hybrid_score(z):
            # (B,C,H,W) -> apply channel weights then spatial weights
            w_norm = w / (w.abs().sum() + 1e-8)
            z_c = (z * w_norm.view(1, -1, 1, 1)).sum(dim=1)  # (B,H,W)
            Wn = Wmap / (Wmap.abs().sum() + 1e-8)
            return (z_c * Wn).sum(dim=(1, 2))

        d_test = hybrid_score(z_test)[0].item()
        d_target = hybrid_score(z_targ)[0].item()

    else:
        raise ValueError(f"Unknown scoring method: {method}")

    print(f"d_test: {d_test:.6f}")
    print(f"d_target: {d_target:.6f}")

    # Optional: save to text
    out_txt = Path(cfg["paths"].get("output_path", "vae_scores.txt"))
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"d_test: {d_test:.6f}\n")
        f.write(f"d_target: {d_target:.6f}\n")
    print(f"Saved scores to: {out_txt}")

if __name__ == "__main__":
    main()
