"""Vision-Language Projector — §5.

You implement: VisionLanguageProjector.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisionLanguageProjector(nn.Module):
    """2-layer MLP that maps image features into the decoder's embedding space.

    Architecture:
        Linear(d_image, expansion * d_image) -> GELU -> Linear(expansion * d_image, d_decoder)

    A single linear layer would only perform an affine transformation, limiting
    the adapter's ability to bridge the semantic gap between the frozen vision
    encoder and frozen language decoder. The hidden nonlinearity allows the
    projector to learn a richer, task-adaptive mapping without updating either
    frozen model.

    Must handle both:
      - A single pooled image vector:  input (B, d_image)         -> output (B, 1, d_decoder)
      - A sequence of patch vectors:   input (B, N_vis, d_image)  -> output (B, N_vis, d_decoder)

    Args:
        d_image:   Image-encoder embedding dim (your ViT's d_model).
        d_decoder: Decoder embedding dim (e.g., 960 for SmolLM2-360M).
        expansion: MLP hidden expansion factor (4 by default, à la LLaVA).
    """

    def __init__(self, d_image: int, d_decoder: int, expansion: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_image, expansion * d_image)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(expansion * d_image, d_decoder)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        # Handle (B, d_image) by unsqueezing to (B, 1, d_image)
        squeeze = False
        if image_features.dim() == 2:
            image_features = image_features.unsqueeze(1)
            squeeze = True
        # (B, N_vis, d_image) -> (B, N_vis, d_decoder)
        out = self.fc2(self.act(self.fc1(image_features)))
        return out
