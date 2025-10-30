# -*- coding: utf-8 -*-
"""
Xsense 数据集读取器（RGB 版，统一到旧流程）
----------------------------------------
- 读取 prepare_xsense_with_cmd.py 产出的 JSONL
- VAE 输入：       input_diff      -> RGB [3,H,W] in [-1,1]
- VAE 监督/扩散目标：target_diff   -> RGB [3,H,W] in [-1,1]
- 扩散图像条件：   contact_depth   -> RGB [3,H,W] in [-1,1]
- 数值/离散条件：  mass(转为克,g)、texture(映射为 id)

返回字典键：
  "vae_in_pixel"       Tensor [-1,1] [3,H,W]
  "vae_tgt_pixel"      Tensor [-1,1] [3,H,W]
  "cond_contact_pixel" Tensor [-1,1] [3,H,W]
  "mass_value"         Tensor [1] float32 (g)
  "texture_id"         Tensor [1] long
"""

import os
import re
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ---------------------------- 工具与转换 ----------------------------

def _norm_hw(image_size):
    """支持 int 或 [H,W]"""
    if isinstance(image_size, int):
        return int(image_size), int(image_size)
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return int(image_size[0]), int(image_size[1])
    raise ValueError(f"Invalid image_size: {image_size}")


def _read_rgb_01(path: str, size_hw: Tuple[int, int]) -> torch.Tensor:
    """读取为 RGB，并调整到 [0,1] 的 CHW 张量（3,H,W）"""
    H, W = size_hw
    im = Image.open(path).convert("RGB")
    if im.size != (W, H):
        im = im.resize((W, H), Image.BICUBIC)
    x01 = T.ToTensor()(im)  # [3,H,W] in [0,1]
    return x01


def _to_raw(x01: torch.Tensor) -> torch.Tensor:
    """[0,1] -> [-1,1]"""
    return x01 * 2.0 - 1.0


_mass_pat = re.compile(r'^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)')


def parse_mass_to_grams(s: str) -> float:
    """尽量把各种单位解析为克（g），解析失败返回 0.0"""
    if not s:
        return 0.0
    m = _mass_pat.match(str(s).strip())
    if not m:
        return 0.0
    v = float(m.group(1))
    u = (m.group(2) or "g").lower()
    if u in ("g", "gram", "grams", ""):
        return v
    if u in ("kg", "kilogram", "kilograms"):
        return v * 1000.0
    if u in ("mg",):
        return v / 1000.0
    if u in ("lb", "lbs", "pound", "pounds"):
        return v * 453.59237
    if u in ("oz", "ounce", "ounces"):
        return v * 28.349523125
    if u in ("jin",):
        return v * 500.0
    if u in ("liang",):
        return v * 500.0 / 16.0
    try:
        return float(s)
    except Exception:
        return 0.0


def build_or_load_texture_vocab(jsonl_path: str, out_dir: str):
    """在 out_dir 生成/复用 texture_vocab.json（{"<unk>":0, "rough":1, ...}）"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vf = out / "texture_vocab.json"
    if vf.exists():
        return json.loads(vf.read_text(encoding="utf-8"))
    tex_set = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                t = str(o.get("texture", "")).strip().lower()
                if t:
                    tex_set.add(t)
            except Exception:
                pass
    mapping = {"<unk>": 0}
    for i, t in enumerate(sorted(tex_set), start=1):
        mapping[t] = i
    vf.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return mapping


# ---------------------------- 数据集实现 ----------------------------

class XsenseTactileDataset(Dataset):
    """
    读取 prepare_xsense_with_cmd.py 产出的 JSONL（键：input_diff, target_diff, contact_depth, mass, texture）

    __getitem__ 返回：
      {
        "vae_in_pixel":       Tensor[-1,1] [3,H,W],  # input_diff (RGB)
        "vae_tgt_pixel":      Tensor[-1,1] [3,H,W],  # target_diff (RGB)
        "cond_contact_pixel": Tensor[-1,1] [3,H,W],  # contact_depth (RGB)
        "mass_value":         Tensor[1] float32 (g),
        "texture_id":         Tensor[1] long,
      }
    """

    def __init__(self, meta_file: str, image_size, root_prefix: str = "", vocab_dir: str = None):
        super().__init__()
        self.meta: List[Dict[str, Any]] = []
        with open(meta_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.meta.append(json.loads(line))
        self.N = len(self.meta)
        self.root = root_prefix
        self.size = _norm_hw(image_size)

        # 纹理词表（优先复用）
        self.vocab_dir = vocab_dir or (Path(meta_file).parent.as_posix())
        self.tex_vocab = build_or_load_texture_vocab(meta_file, self.vocab_dir)

    def __len__(self):
        return self.N

    def _abs(self, p: str):
        return p if os.path.isabs(p) else os.path.join(self.root, p)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        m = self.meta[idx]

        # 统一按 RGB 处理（与旧版 dataset_tactile.py 一致）
        x_in  = _read_rgb_01(self._abs(m["input_diff"]),    self.size)  # [3,H,W] in [0,1]
        x_tgt = _read_rgb_01(self._abs(m["target_diff"]),   self.size)  # [3,H,W] in [0,1]
        x_cdp = _read_rgb_01(self._abs(m["contact_depth"]), self.size)  # [3,H,W] in [0,1]

        vae_in_pixel       = _to_raw(x_in)   # [-1,1]
        vae_tgt_pixel      = _to_raw(x_tgt)  # [-1,1]
        cond_contact_pixel = _to_raw(x_cdp)  # [-1,1]

        # mass / texture
        mass_value = torch.tensor(parse_mass_to_grams(m.get("mass", "0")), dtype=torch.float32)
        tex = str(m.get("texture", "")).strip().lower()
        texture_id = torch.tensor(self.tex_vocab.get(tex, 0), dtype=torch.long)

        return {
            "vae_in_pixel":       vae_in_pixel,
            "vae_tgt_pixel":      vae_tgt_pixel,
            "cond_contact_pixel": cond_contact_pixel,
            "mass_value":         mass_value,
            "texture_id":         texture_id,
        }
