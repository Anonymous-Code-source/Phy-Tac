# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

class MassTextureCmdEmbedder(nn.Module):
    """
    将 mass_value (float, g), texture_id (int), cmd_mm (float, mm) 融合为 cross-attn 条件：
      - texture:  Embedding(num_textures, D)
      - mass:     MLP(1 -> D)
      - cmd_mm:   MLP(1 -> D)
      - fuse:     Linear(3D -> D) + SiLU + Linear(D -> D)
      - 输出:     条件序列 cond [B, T, D] 与无条件序列 uncond [B, T, D]
    训练期可用 cond_drop_prob 做条件丢弃（类似 classifier-free guidance）。

    用法：
      cond, uncond = mt_embedder(mass_value, texture_id, cmd_mm, cond_drop_prob=0.1)
      # 将 cond 作为 encoder_hidden_states 喂给 U-Net；推理时可按 CFG 线性组合。

    备注：
      - 若当前实验不使用 cmd_mm，可在上游传入全 0 张量或启用 use_cmd=False 初始化关闭该分支。
      - mass/cmd 的数值尺度可选做对数/标准化以更稳健（见 __init__ 参数）。
    """
    def __init__(
        self,
        num_textures: int,
        embed_dim: int = 256,
        seq_len: int = 2,
        use_cmd: bool = True,
        mass_log1p: bool = True,
        mass_scale: float = 1.0,   # 例如把克转千克可设 1/1000
        cmd_scale: float = 1.0,    # 例如毫米转厘米可设 0.1
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

        # texture 分支
        self.tex_emb = nn.Embedding(num_textures, embed_dim)

        # mass 分支（1 -> D）
        self.mass_mlp = nn.Sequential(
            nn.Linear(1, h), act, nn.Linear(h, embed_dim)
        )

        # cmd 分支（1 -> D），可关闭
        if self.use_cmd:
            self.cmd_mlp = nn.Sequential(
                nn.Linear(1, h), act, nn.Linear(h, embed_dim)
            )
        else:
            self.register_parameter("cmd_mlp", None)

        # 融合
        in_fuse = embed_dim * (3 if self.use_cmd else 2)
        self.fuse = nn.Sequential(
            nn.Linear(in_fuse, h), act, nn.Dropout(dropout), nn.Linear(h, embed_dim)
        )

        # null token（无条件向量）
        self.register_buffer("null_token", torch.zeros(embed_dim), persistent=False)

        # 轻量归一化层（可提高稳定性；保持可关闭）
        self.mass_norm = nn.LayerNorm(embed_dim)
        self.cmd_norm  = nn.LayerNorm(embed_dim) if self.use_cmd else None
        self.tex_norm  = nn.LayerNorm(embed_dim)
        self.out_norm  = nn.LayerNorm(embed_dim)

    @staticmethod
    def _to_2d(x) -> torch.Tensor:
        """将标量输入拉成 [B,1] 的 float32/float16/bfloat16 张量"""
        if torch.is_tensor(x):
            x = x.view(-1, 1)
        else:
            x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
        return x

    def _prep_scalar(self, x: torch.Tensor, scale: float, use_log1p: bool) -> torch.Tensor:
        x = x * scale
        if use_log1p:
            x = torch.log1p(torch.clamp(x, min=0))  # 非负假设；如需支持负值可改为 torch.sign(x)*torch.log1p(x.abs())
        return x

    def forward(
        self,
        mass_value,              # Tensor/ndarray/list/scalar, 单位: g
        texture_id,              # Tensor/ndarray/list/scalar (long)
        cmd_mm=None,             # Tensor/..., 单位: mm；若 use_cmd=False 可忽略
        cond_drop_prob: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
          cond   [B, T, D] 条件序列
          uncond [B, T, D] 无条件序列（同形状，用于 CFG）
        """
        device = self.null_token.device
        dtype  = self.null_token.dtype

        # --- mass 分支 ---
        m = self._to_2d(mass_value).to(device=device, dtype=dtype)
        m = self._prep_scalar(m, self.mass_scale, self.mass_log1p)
        m_feat = self.mass_mlp(m)                      # [B, D]
        m_feat = self.mass_norm(m_feat)

        # --- texture 分支 ---
        if not torch.is_tensor(texture_id):
            t = torch.tensor(texture_id, dtype=torch.long, device=device)
        else:
            t = texture_id.to(device=device, dtype=torch.long)
        tex_feat = self.tex_emb(t)                     # [B, D]
        tex_feat = self.tex_norm(tex_feat)

        feats = [m_feat, tex_feat]

        # --- cmd 分支（可选） ---
        if self.use_cmd:
            if cmd_mm is None:
                # 若未提供，则以 0 占位
                c = torch.zeros((m.shape[0], 1), device=device, dtype=dtype)
            else:
                c = self._to_2d(cmd_mm).to(device=device, dtype=dtype)
            c = self._prep_scalar(c, self.cmd_scale, self.cmd_log1p)
            c_feat = self.cmd_mlp(c)                  # [B, D]
            c_feat = self.cmd_norm(c_feat)
            feats.append(c_feat)

        # --- 融合 ---
        emb = self.fuse(torch.cat(feats, dim=-1))      # [B, D]
        emb = self.out_norm(emb)

        # 展开为 cross-attn 所需的序列（长度 T）
        cond = emb.unsqueeze(1).repeat(1, self.seq_len, 1)    # [B, T, D]

        # 无条件向量（全 0 作为 null token）
        B = emb.size(0)
        uncond = self.null_token.unsqueeze(0).unsqueeze(1).repeat(B, self.seq_len, 1)

        # 训练期条件丢弃（CFG）
        if self.training and cond_drop_prob > 0:
            mask = torch.rand(B, device=device) < float(cond_drop_prob)
            if mask.any():
                cond[mask] = uncond[mask]

        return cond, uncond


# ---------------------------- 简单自测 ----------------------------
if __name__ == "__main__":
    B = 4
    mt = MassTextureCmdEmbedder(num_textures=128, embed_dim=256, seq_len=2, use_cmd=True).to("cpu")
    mass = torch.rand(B) * 500.0   # g
    tex  = torch.randint(0, 128, (B,))
    cmd  = torch.rand(B) * 30.0    # mm
    cond, uncond = mt(mass, tex, cmd, cond_drop_prob=0.1)
    print(cond.shape, uncond.shape)  # => torch.Size([4, 2, 256]) torch.Size([4, 2, 256])
