# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

class MassTextureCmdEmbedder(nn.Module):

    def __init__(
        self,
        num_textures: int,
        embed_dim: int = 256,
        seq_len: int = 2,
        use_cmd: bool = True,
        mass_log1p: bool = True,
        mass_scale: float = 1.0,   
        cmd_scale: float = 1.0,    
        cmd_log1p: bool = False,
        mlp_hidden: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.embed_dim = int(embed_dim)
        self.use_cmd = bool(use_cmd)
        self.mass_log1p = bool(mass_log1p)
        self.mass_scale = float(mass_scale)
        self.cmd_scale  = float(cmd_scale)
        self.cmd_log1p  = bool(cmd_log1p)

        h = mlp_hidden or embed_dim
        act = nn.SiLU()

 
        self.tex_emb = nn.Embedding(num_textures, embed_dim)


        self.mass_mlp = nn.Sequential(
            nn.Linear(1, h), act, nn.Linear(h, embed_dim)
        )


        if self.use_cmd:
            self.cmd_mlp = nn.Sequential(
                nn.Linear(1, h), act, nn.Linear(h, embed_dim)
            )
        else:
            self.register_parameter("cmd_mlp", None)


        in_fuse = embed_dim * (3 if self.use_cmd else 2)
        self.fuse = nn.Sequential(
            nn.Linear(in_fuse, h), act, nn.Dropout(dropout), nn.Linear(h, embed_dim)
        )


        self.register_buffer("null_token", torch.zeros(embed_dim), persistent=False)


        self.mass_norm = nn.LayerNorm(embed_dim)
        self.cmd_norm  = nn.LayerNorm(embed_dim) if self.use_cmd else None
        self.tex_norm  = nn.LayerNorm(embed_dim)
        self.out_norm  = nn.LayerNorm(embed_dim)

    @staticmethod
    def _to_2d(x) -> torch.Tensor:
        if torch.is_tensor(x):
            x = x.view(-1, 1)
        else:
            x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
        return x

    def _prep_scalar(self, x: torch.Tensor, scale: float, use_log1p: bool) -> torch.Tensor:
        x = x * scale
        if use_log1p:
            x = torch.log1p(torch.clamp(x, min=0)) 
        return x

    def forward(
        self,
        mass_value,              # Tensor/ndarray/list/scalar, 
        texture_id,              # Tensor/ndarray/list/scalar (long)
        cmd_mm=None,             # Tensor/..., 
        cond_drop_prob: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.null_token.device
        dtype  = self.null_token.dtype


        m = self._to_2d(mass_value).to(device=device, dtype=dtype)
        m = self._prep_scalar(m, self.mass_scale, self.mass_log1p)
        m_feat = self.mass_mlp(m)                      # [B, D]
        m_feat = self.mass_norm(m_feat)

        if not torch.is_tensor(texture_id):
            t = torch.tensor(texture_id, dtype=torch.long, device=device)
        else:
            t = texture_id.to(device=device, dtype=torch.long)
        tex_feat = self.tex_emb(t)                     # [B, D]
        tex_feat = self.tex_norm(tex_feat)

        feats = [m_feat, tex_feat]

        if self.use_cmd:
            if cmd_mm is None:
                c = torch.zeros((m.shape[0], 1), device=device, dtype=dtype)
            else:
                c = self._to_2d(cmd_mm).to(device=device, dtype=dtype)
            c = self._prep_scalar(c, self.cmd_scale, self.cmd_log1p)
            c_feat = self.cmd_mlp(c)                  # [B, D]
            c_feat = self.cmd_norm(c_feat)
            feats.append(c_feat)


        emb = self.fuse(torch.cat(feats, dim=-1))      # [B, D]
        emb = self.out_norm(emb)

        cond = emb.unsqueeze(1).repeat(1, self.seq_len, 1)    # [B, T, D]

        B = emb.size(0)
        uncond = self.null_token.unsqueeze(0).unsqueeze(1).repeat(B, self.seq_len, 1)

        if self.training and cond_drop_prob > 0:
            mask = torch.rand(B, device=device) < float(cond_drop_prob)
            if mask.any():
                cond[mask] = uncond[mask]

        return cond, uncond


# ----------------------------  ----------------------------
if __name__ == "__main__":
    B = 4
    mt = MassTextureCmdEmbedder(num_textures=128, embed_dim=256, seq_len=2, use_cmd=True).to("cpu")
    mass = torch.rand(B) * 500.0   # g
    tex  = torch.randint(0, 128, (B,))
    cmd  = torch.rand(B) * 30.0    # mm
    cond, uncond = mt(mass, tex, cmd, cond_drop_prob=0.1)
    print(cond.shape, uncond.shape)  # => torch.Size([4, 2, 256]) torch.Size([4, 2, 256])
