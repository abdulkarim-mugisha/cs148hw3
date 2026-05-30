"""Vision Transformer — §2.

You implement: PatchEmbeddings, ViT.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from basics.model import Block


# ---------------------------------------------------------------------------
# RoPE-aware attention blocks (§6 ablations)
# These live here because basics/model.py must not be modified.
# ---------------------------------------------------------------------------

class _RoPEHead(nn.Module):
    """Single attention head that applies either 1D or 2D RoPE to Q and K."""

    def __init__(self, d_model: int, head_dim: int, is_decoder: bool = False,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.is_decoder = is_decoder
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = None  # set by owning block after construction

    def forward(self, x: torch.Tensor, positions_1d=None,
                x_coords=None, y_coords=None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x)  # (B, T, head_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.rope is not None:
            # Reshape to (B, 1, T, head_dim) for RoPE interface
            q_r = q.unsqueeze(1)
            k_r = k.unsqueeze(1)
            if x_coords is not None:
                q_r = self.rope(q_r, x_coords, y_coords)
                k_r = self.rope(k_r, x_coords, y_coords)
            else:
                q_r = self.rope(q_r, positions_1d)
                k_r = self.rope(k_r, positions_1d)
            q = q_r.squeeze(1)
            k = k_r.squeeze(1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return attn @ v


class _RoPEMultiHeadAttn(nn.Module):
    """Multi-head attention using _RoPEHead, sharing a single RoPE module."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0,
                 rope=None) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        head_dim = d_model // num_heads
        self.heads = nn.ModuleList([
            _RoPEHead(d_model, head_dim, is_decoder=False, dropout=dropout)
            for _ in range(num_heads)
        ])
        for h in self.heads:
            h.rope = rope  # shared RoPE module (buffers are shared, no grad)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions_1d=None, x_coords=None, y_coords=None):
        out = torch.cat(
            [h(x, positions_1d, x_coords, y_coords) for h in self.heads],
            dim=-1,
        )
        return self.dropout(self.out_proj(out))


class _RoPEBlock(nn.Module):
    """Pre-LayerNorm Transformer block with RoPE attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0,
                 rope=None) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _RoPEMultiHeadAttn(d_model, num_heads, dropout, rope)
        self.ln2 = nn.LayerNorm(d_model)
        d_ff = 4 * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, positions_1d=None, x_coords=None, y_coords=None):
        x = x + self.attn(self.ln1(x), positions_1d, x_coords, y_coords)
        x = x + self.mlp(self.ln2(x))
        return x


class PatchEmbeddings(nn.Module):
    """Split an image into non-overlapping patches and project each to d_model.

    Implemented with a strided Conv2d whose kernel size and stride both equal
    `patch_size`.

    Args:
        img_size:   Input image side length (assumed square). Must be divisible
                    by patch_size.
        patch_size: Side length of each patch in pixels.
        d_model:    Output embedding dimension per patch.

    Forward:
        x: (B, 3, img_size, img_size) float tensor.
        returns: (B, num_patches, d_model) where num_patches = (img_size // patch_size) ** 2.
    """

    def __init__(self, img_size: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        # After conv: (B, d_model, H/P, W/P)
        x = self.proj(x)
        # Flatten spatial dims and transpose to (B, N, d_model)
        B, d, h, w = x.shape
        x = x.flatten(2)          # (B, d_model, N)
        x = x.transpose(1, 2)     # (B, N, d_model)
        return x


class ViT(nn.Module):
    """Vision Transformer.

    Pipeline:
      1. Patchify with `PatchEmbeddings`.
      2. Prepend a learnable [CLS] token.
      3. Add positional information (learned PE, 1D RoPE, or 2D RoPE).
      4. Pass through `num_blocks` Transformer blocks (is_decoder=False).
      5. Final LayerNorm.
      6. Return CLS token (B, d_model), or all tokens if return_all_tokens=True.

    Args:
        img_size, patch_size, d_model, num_heads, num_blocks, dropout
        pos_enc: "learned" (default) | "rope1d" | "rope2d"
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0.1,
        pos_enc: str = "learned",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pos_enc = pos_enc
        self.patch_embed = PatchEmbeddings(img_size, patch_size, d_model)
        self.num_patches = self.patch_embed.num_patches
        self.grid_size = img_size // patch_size  # patches per row/col
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        head_dim = d_model // num_heads

        if pos_enc == "learned":
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, d_model)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.blocks = nn.ModuleList([
                Block(
                    d_model=d_model, num_heads=num_heads,
                    block_size=self.num_patches + 1,
                    is_decoder=False, dropout=dropout,
                )
                for _ in range(num_blocks)
            ])
            self.rope = None

        elif pos_enc == "rope1d":
            from basics.rope import RoPE1D
            # max_seq_len large enough for extrapolation (96x96 / P=8 = 144 patches)
            self.rope = RoPE1D(head_dim=head_dim, max_seq_len=512)
            self.blocks = nn.ModuleList([
                _RoPEBlock(d_model, num_heads, dropout, rope=self.rope)
                for _ in range(num_blocks)
            ])
            self.pos_embed = None

        elif pos_enc == "rope2d":
            from basics.rope import RoPE2D
            # grid_size large enough for extrapolation (96/8=12 > 8)
            self.rope = RoPE2D(head_dim=head_dim, grid_size=32)
            self.blocks = nn.ModuleList([
                _RoPEBlock(d_model, num_heads, dropout, rope=self.rope)
                for _ in range(num_blocks)
            ])
            self.pos_embed = None

        else:
            raise ValueError(f"Unknown pos_enc: {pos_enc!r}")

        self.norm = nn.LayerNorm(d_model)

    def _get_rope_positions(self, num_patches_total: int):
        """Build 1D or 2D position tensors for a given number of patches."""
        device = self.cls_token.device
        N = num_patches_total  # excludes CLS

        if self.pos_enc == "rope1d":
            # CLS = pos 0, patches = pos 1..N
            positions = torch.arange(N + 1, device=device)  # (N+1,)
            return {"positions_1d": positions}

        else:  # rope2d
            # Infer grid size from N (assumes square grid)
            g = int(N ** 0.5)
            assert g * g == N, f"Expected square patch grid, got N={N}"
            # CLS gets (0, 0); patches get their (col, row) coords (1-indexed to avoid collision)
            xs = torch.zeros(N + 1, dtype=torch.long, device=device)
            ys = torch.zeros(N + 1, dtype=torch.long, device=device)
            patch_idx = torch.arange(N, device=device)
            xs[1:] = (patch_idx % g) + 1   # col 1..g
            ys[1:] = (patch_idx // g) + 1  # row 1..g
            return {"x_coords": xs, "y_coords": ys}

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        B, _, H, W = x.shape
        x = self.patch_embed(x)          # (B, N, d_model)
        N = x.shape[1]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)   # (B, N+1, d_model)

        if self.pos_enc == "learned":
            # Interpolate learned PE if sequence length differs (extrapolation test)
            if x.shape[1] != self.pos_embed.shape[1]:
                x = x + self._interpolate_pos_embed(x.shape[1])
            else:
                x = x + self.pos_embed

            for block in self.blocks:
                x = block(x)

        else:
            pos = self._get_rope_positions(N)
            for block in self.blocks:
                x = block(x, **pos)

        x = self.norm(x)
        if return_all_tokens:
            return x
        return x[:, 0]

    def _interpolate_pos_embed(self, new_seq_len: int) -> torch.Tensor:
        """Bilinearly interpolate learned patch PE to a different grid size."""
        pe = self.pos_embed  # (1, N_train+1, d_model)
        cls_pe = pe[:, :1, :]          # keep CLS PE as-is
        patch_pe = pe[:, 1:, :]        # (1, N_train, d_model)

        N_train = patch_pe.shape[1]
        g_train = int(N_train ** 0.5)
        N_new = new_seq_len - 1
        g_new = int(N_new ** 0.5)

        # Reshape to 2D spatial grid and interpolate
        patch_pe_2d = patch_pe.reshape(1, g_train, g_train, -1).permute(0, 3, 1, 2)
        patch_pe_2d = F.interpolate(
            patch_pe_2d.float(), size=(g_new, g_new), mode="bilinear", align_corners=False
        ).to(patch_pe.dtype)
        patch_pe_new = patch_pe_2d.permute(0, 2, 3, 1).reshape(1, N_new, -1)
        return torch.cat([cls_pe, patch_pe_new], dim=1)
