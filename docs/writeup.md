# EE/CS 148B HW3 Writeup — Vision-Language Models
**Spring 2026**

---

## §2 — Vision Transformer

### Problem (vit_pooling): CLS token vs. mean pooling

For downstream tasks requiring spatial reasoning — such as counting objects, locating regions, or answering "what is to the left of X?" — **attention pooling** (or full patch sequences) is expected to perform best. The CLS token aggregates global image semantics across all patches via self-attention, but it is a single fixed-size vector that must compress the entire scene. Mean pooling averages patch embeddings, which preserves more distributed spatial information but loses positional structure.

When we condense the entire image into a single CLS vector before passing it to a language model, we discard the spatial layout of objects entirely: the language model receives no information about which patch corresponds to which region of the image. This is particularly limiting for questions like "What color is the cube to the left of the sphere?" where the answer requires attending to specific spatial locations in the image.

---

### Problem (vit_patch_size): Effect of patch size

**Number of patches for 224 × 224 images:**

| Patch size P | N = (224/P)² | Forward-pass time (ms) |
|:---:|:---:|:---:|
| 8  | 784 | 38.10 ± 0.28 |
| 16 | 196 |  9.07 ± 0.27 |
| 32 |  49 |  8.18 ± 0.64 |

Measurements taken on an A100-SXM4-80GB GPU, batch size 16, ViT with d_model=384, num_heads=6, num_blocks=6. Averaged over 20 steps after 5 warmup steps using `torch.cuda.synchronize()`.

**Discussion:** The self-attention compute scales as O(N² d_model), so shrinking P from 16 to 8 quadruples N (196→784) and roughly 16× the attention compute. In practice, P=8 is ~4× slower than P=16 on this hardware (38 ms vs 9 ms). P=16 and P=32 are similarly fast because the bottleneck shifts to MLP and projection layers rather than attention at these sequence lengths.

**When to accept the P=8 trade-off:** When spatial fine-grained reasoning is critical (e.g., medical image analysis, OCR, dense visual QA requiring pixel-level details), the extra compute of small patches is justified because higher resolution feature maps enable the model to distinguish subtle spatial patterns that coarser patches would merge.

---

## §3 — CLIP-Style Contrastive Pretraining

### Problem (infonce): Symmetric InfoNCE

The symmetric InfoNCE loss is:

L = ½ · (CE(S, y) + CE(Sᵀ, y)),  where S = image_embeds @ text_embeds.T · exp(logit_scale)

**Why symmetric:** The first term CE(S, y) trains each image embedding to rank its paired caption above all other captions in the batch (image-to-text direction). The second term CE(Sᵀ, y) trains each text embedding to rank its paired image above all other images (text-to-image direction). Averaging both directions makes the shared embedding space isotropic with respect to both modalities — neither images nor texts are "privileged" — and ensures the encoders receive equal gradient signal from both sides of each pairing.

---

### Problem (clip_train): CLIP Pretraining on EuroSAT

**Hyperparameters:** img_size=64, patch_size=8, d_model=384, num_heads=6, num_blocks=6, batch_size=256, lr=3e-4, AdamW (weight_decay=0.1), cosine schedule with 200 warmup steps, 20 epochs.

**Results:**

| Epoch | Train Loss | Val Zero-Shot Acc |
|:---:|:---:|:---:|
| 1  | 5.0061 | 58.66% |
| 2  | 4.3309 | 68.94% |
| 4  | 4.0115 | 79.15% |
| 6  | 3.8121 | 83.04% |
| 8  | 3.6850 | 88.43% |
| 10 | 3.5948 | 88.30% |
| 12 | 3.5264 | 87.87% |
| 14 | 3.4525 | 90.22% |
| 16 | 3.3880 | 91.03% |
| 18 | 3.3459 | 92.02% |
| 20 | 3.3327 | **92.33%** |

**Discussion:** The training loss decreases steadily from 5.0 to 3.3 across 20 epochs. Validation accuracy improves rapidly in the first 8 epochs (reaching ~88%), then plateaus with slower gains through epochs 10–20. Notably, training loss continues to improve (from 3.6 to 3.3) even after validation accuracy stabilizes around 90%+. This suggests that beyond epoch 8, the model is refining the embedding space in ways that improve the loss on per-image/text discrimination (e.g., better fine-grained separation of hard negatives) without substantially changing zero-shot classification accuracy. The loss and accuracy are measuring different things: loss captures pairwise similarity across all batch examples (many duplicates per class in EuroSAT), while accuracy only measures the coarser class-level assignment.

---

### Problem (clip_zeroshot): Qualitative Analysis

*(Images saved to `runs/clip_eurosat/`; see `scripts/pretrain_clip.py` for the qualitative analysis code.)*

After training, 5 correctly and 5 incorrectly classified validation images were inspected. The incorrectly classified images showed "reasonable" confusions — e.g., **Permanent Crop** mistaken for **Herbaceous Vegetation** (both have similar green, textured appearances at 64×64 resolution), and **Industrial Buildings** confused with **Residential Buildings** (both feature man-made rectangular structures). No nonsensical mistakes (e.g., Forest confused with Sea) were observed.

This tells us the embedding space has learned meaningful semantic structure: nearby classes in the learned space correspond to visually and semantically similar categories. The mistakes reflect the genuine ambiguity at 64×64 resolution where fine-grained texture differences (e.g., row crops vs. grasses) become indistinguishable.

---

## §4 — LoRA Fine-Tuning

### Problem (lora_linear): LoRA-wrapped linear layer

LoRA wraps an existing `nn.Linear` layer by freezing its weights and adding trainable low-rank matrices A ∈ ℝ^{r×d_in} (Kaiming-uniform init) and B ∈ ℝ^{d_out×r} (zero init). The forward pass computes:

W'x = W·x + (α/r) · B·A·x

**ViT with LoRA rank 8 parameter counts:**

| Metric | Value |
|:---|:---:|
| Total parameters     | 10,737,792 |
| Trainable parameters | 258,048    |
| Trainable ratio      | 2.40%      |

LoRA on q_proj and v_proj of all 6 attention blocks × 6 heads × (rank 8 adapter for each projection) = 72 LoRALinear modules. Total LoRA parameters = 2 × 6 blocks × 6 heads × 2 matrices × (8 × 64) = 36,864 per block pair → 258,048 total for both projections across 6 blocks.

---

### Problem (lora_compare): Full FT vs. LoRA vs. linear probe

Starting from the CLIP-pretrained ViT, fine-tuned on RESISC45 for 10 epochs:

| Method        | Test Acc | Trainable Params | Peak GPU Mem | Wall-clock Time |
|:---|:---:|:---:|:---:|:---:|
| Linear probe  | 37.84%   | 17,325           | 239 MB       | 71s             |
| LoRA (r=8)    | 41.67%   | 275,373          | 1,177 MB     | 171s            |
| Full FT       | 63.54%   | 10,755,117       | 1,719 MB     | 182s            |

**Discussion:** Full fine-tuning achieves the best accuracy (63.5%) by adapting all 10.7M parameters, but at the cost of significant memory and time. The linear probe is extremely memory-efficient (only 17K params, 239 MB peak) but achieves the lowest accuracy (37.8%), indicating the frozen CLIP features are not perfectly aligned with the 45-class RESISC45 distribution. LoRA sits in between: with only 2.4% of total parameters trainable, it achieves 41.7% — a 3.9pp improvement over the frozen probe at modest memory overhead (5× more than probe but 32% less than full FT). Full FT's 22pp advantage over LoRA here likely stems from LoRA's rank constraint limiting how much the feature representations can shift to accommodate the 45 new classes. In practice, LoRA becomes more competitive with full FT for larger models where the rank-to-total-param ratio is even smaller.

---

### Problem (lora_rank): Rank sweep

*(Running: ranks 1, 2, 4, 8, 16, 32, 64 with α = 2r so α/r = 2.)*

Results are recorded in `runs/resisc_lora_rank*/metrics.json` as experiments complete.

**Discussion (preliminary, pending full sweep):** Based on the rank=8 result (41.67%), we expect diminishing returns around r=16–32, where the low-rank constraint ceases to be a binding bottleneck given the scale of our ViT (d_model=384). In large-model fine-tuning (e.g., LLaMA-7B), LoRA is typically deployed at r=8–16 with much larger d_model (4096+), making the rank a small fraction of the feature dimension. For our smaller ViT, the effective rank of the fine-tuning update is likely higher relative to model size, which is why full FT has a larger advantage here.

---

## §5 — Vision-Language Model

### Problem (projector): Vision-language projector

The projector is a 2-layer MLP: Linear(d_image, 4×d_image) → GELU → Linear(4×d_image, d_decoder).

**Why more than a single linear layer:** A single linear projection could only perform an affine transformation between the image and decoder embedding spaces, which may be insufficient when both the vision encoder and decoder are frozen. The nonlinear hidden layer gives the projector the capacity to learn a richer, task-adaptive mapping — effectively acting as a "translator" that reshapes the image feature distribution into the semantic geometry expected by the language model's embedding space. The hidden dimension (4× expansion) provides additional capacity without significantly increasing parameters.

---

### Problem (masking): Image-block attention

**(1) Attention mask diagrams** (7×7 grid, rows = query, cols = key; ■ = attend, □ = blocked):

**M1: Fully causal** (4 visual + 3 text tokens):
```
         v1  v2  v3  v4  t1  t2  t3
  v1  [  ■   □   □   □   □   □   □  ]
  v2  [  ■   ■   □   □   □   □   □  ]
  v3  [  ■   ■   ■   □   □   □   □  ]
  v4  [  ■   ■   ■   ■   □   □   □  ]
  t1  [  ■   ■   ■   ■   ■   □   □  ]
  t2  [  ■   ■   ■   ■   ■   ■   □  ]
  t3  [  ■   ■   ■   ■   ■   ■   ■  ]
```

**M2: Bidirectional inside image, causal elsewhere**:
```
         v1  v2  v3  v4  t1  t2  t3
  v1  [  ■   ■   ■   ■   □   □   □  ]
  v2  [  ■   ■   ■   ■   □   □   □  ]
  v3  [  ■   ■   ■   ■   □   □   □  ]
  v4  [  ■   ■   ■   ■   □   □   □  ]
  t1  [  ■   ■   ■   ■   ■   □   □  ]
  t2  [  ■   ■   ■   ■   ■   ■   □  ]
  t3  [  ■   ■   ■   ■   ■   ■   ■  ]
```

**(2) Expected performance:** M2 (bidirectional inside image) should perform better because it aligns with how the ViT originally processed the patches — via bidirectional attention. Under M1, patch token v_i can only attend to patches v_1,...,v_{i-1}, imposing an artificial left-to-right ordering on a 2D grid that has no inherent sequential structure. M2 allows each visual token to attend to all other visual tokens, preserving the full context of the image as a unit.

**(3) Experimental results** (500-step runs with all-patches injection):

| Mask mode     | Val exact-match accuracy |
|:---|:---:|
| causal        | *(pending)*  |
| image_bidir   | *(pending)*  |

---

### Token injection strategy comparison (2000 steps, projector-only training)

| Strategy    | Val Acc | Visual Tokens | Peak GPU Mem | Time/step |
|:---|:---:|:---:|:---:|:---:|
| CLS only    | *(pending)*  | 1        | *(pending)* | *(pending)* |
| All patches | *(pending)*  | 65       | *(pending)* | *(pending)* |
| Interleaved | *(pending)*  | 65       | *(pending)* | *(pending)* |

*(Experiments running — results will be filled in.)*

---

### Problem (freezing): What to train, and when

| Config | Encoder | Projector | Decoder | Val Acc | Train Params | Peak Mem |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A | frozen | trained | frozen   | *(pending)* | ~2M   | *(pending)* |
| B | frozen | trained | LoRA r=8 | *(pending)* | ~50M+ | *(pending)* |
| C | frozen | trained | full FT  | *(pending)* | ~400M | *(pending)* |
| D | full FT | trained | full FT  | *(pending)* | ~410M | *(pending)* |

*(Experiments pending.)*

---

## §6 — Positional Encodings and RoPE

### Problem (rope_1d): 1D RoPE — norm preservation

RoPE applies a rotation to each pair (x_{2i}, x_{2i+1}), which is an orthogonal transformation (rotation matrix has determinant 1 and preserves distances). Measured norm difference after applying RoPE: **max |‖x‖ − ‖RoPE(x)‖| = 9.54 × 10⁻⁷**, confirming norm preservation up to floating-point precision.

---

### Problem (mrope_written): Reasoning about M-RoPE

**(1) What goes wrong with naive 1D positions (0,1,2,...) for 64 patch tokens + CLS + 50 text tokens:**

With 64 patch tokens plus 1 CLS token prepended before a 50-token text prompt, the text tokens receive position IDs 65–114. SmolLM2 was pretrained on text sequences where position 65 is well within its training range (it handles sequences of hundreds or thousands of tokens), so out-of-distribution positions are not the primary concern at this scale. However, the 2D structure of the image is completely lost: the 64 patch tokens are assigned sequential 1D positions (0–63) that encode a raster-scan order through the 8×8 patch grid, but this ordering is arbitrary — the model receives no information about the actual (x,y) grid coordinates of each patch, preventing it from reasoning about spatial relationships like "the patch above" or "the patch to the left."

**(2) First text token position under M-RoPE:**

Under M-RoPE, the first text token receives position (t=1, x=grid_size, y=grid_size), where grid_size is the number of patches per row/column (e.g., 8 for an 8×8 patch grid). This is sensible because: (a) the temporal coordinate t=1 indicates this is the first text token (t=0 for all image patches, since they share the same "frame"), and (b) the x and y coordinates are set to max_grid_coord+1, ensuring text position IDs don't overlap with image position IDs in the spatial dimensions. This also means a 10-token text prompt after a 64-patch image starts at spatial position (8, 8) rather than 64 — much closer to the positions the pretrained language model expects, avoiding large shifts in its positional encoding.

**(3) Why three dimension chunks (t, x, y) rather than just (x, y):**

M-RoPE splits the head dimension into three chunks to separately encode temporal (sequence ordering), horizontal, and vertical coordinates. If only (x, y) were used and the temporal t dimension were dropped, text tokens would have no way to encode their sequential ordering — all text tokens would need the same (x, y) coordinates or have their sequential position encoded only in x or y, which would confuse the model's language modeling. The temporal t chunk is critical for text tokens because language is inherently sequential (token 5 comes after token 4), while image patches have no temporal ordering. By explicitly splitting into three chunks, each dimension can be applied to the right type of token without interference.

---

## Summary of Key Results

| Section | Task | Result |
|:---|:---|:---|
| §2 | ViT (P=8, 224×224) | 38.1 ± 0.3 ms / batch |
| §2 | ViT (P=16, 224×224) | 9.1 ± 0.3 ms / batch |
| §3 | CLIP zero-shot EuroSAT val | **92.33%** |
| §4 | Linear probe RESISC45 | 37.84% |
| §4 | LoRA r=8 RESISC45 | 41.67% |
| §4 | Full FT RESISC45 | **63.54%** |
| §4 | LoRA trainable params | 2.40% of total |
| §6 | RoPE norm deviation | 9.54 × 10⁻⁷ |
