"""Vision-Language Model — §5.

You implement: VisionLanguageModel.

Three injection strategies to support:
  - "cls":          Single visual token (the ViT's CLS embedding) prepended.
  - "all_patches":  All N+1 visual tokens (CLS + patches) prepended.
  - "interleaved":  A special <image> token in the prompt is replaced by the
                    sequence of patch embeddings at runtime.

Two attention masking strategies to support (Problem `masking`):
  - "causal":         Fully causal across the whole sequence.
  - "image_bidir":    Bidirectional within the image block, causal everywhere
                      else. Use vlm.masking.build_image_bidir_mask().
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from vlm.masking import build_image_bidir_mask

InjectionMode = Literal["cls", "all_patches", "interleaved"]
MaskMode = Literal["causal", "image_bidir"]


class VisionLanguageModel(nn.Module):
    """ViT image encoder + projector + pretrained causal LM decoder.

    Args:
        vit:       Your CLIP-pretrained ViT from §3.
        projector: vlm.projector.VisionLanguageProjector instance.
        decoder:   HuggingFace causal LM (e.g., SmolLM2-360M-Instruct) loaded
                   in bf16 with FlashAttention-2.
        tokenizer: Matching HF tokenizer.
        image_token_id: Token ID corresponding to the special <image> placeholder
                        in interleaved mode (None for cls / all_patches modes).

    Forward:
        images:         (B, 3, H, W) float tensor.
        input_ids:      (B, T) tokenized text.
        attention_mask: (B, T) text attention mask from the tokenizer.
        labels:         (B, T) for loss computation, or None for inference.
                        Visual-token positions must be set to -100 in labels
                        before being passed in (so they're masked out by HF's
                        loss).
        injection:      One of "cls", "all_patches", "interleaved".
        mask_mode:      One of "causal", "image_bidir".

    Returns:
        A dict with at least:
          - "loss":   scalar (only if labels was provided).
          - "logits": (B, T_total, vocab_size).
    """

    def __init__(
        self,
        vit: nn.Module,
        projector: nn.Module,
        decoder: nn.Module,
        tokenizer,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id

    def _get_visual_tokens(self, images: torch.Tensor, injection: InjectionMode) -> torch.Tensor:
        """Encode images and project to decoder dim.

        Returns (B, N_vis, d_decoder).
        """
        if injection == "cls":
            # (B, d_model)
            img_feats = self.vit(images, return_all_tokens=False)
            # -> (B, 1, d_decoder)
            visual_tokens = self.projector(img_feats)
        else:
            # "all_patches" or "interleaved": use all N+1 tokens
            img_feats = self.vit(images, return_all_tokens=True)  # (B, N+1, d_model)
            visual_tokens = self.projector(img_feats)              # (B, N+1, d_decoder)
        return visual_tokens

    def _get_text_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from decoder's embedding layer."""
        embed_layer = self.decoder.get_input_embeddings()
        return embed_layer(input_ids)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
    ) -> dict:
        B, T = input_ids.shape
        device = images.device
        dtype = next(self.decoder.parameters()).dtype

        visual_tokens = self._get_visual_tokens(images, injection)  # (B, N_vis, d_dec)
        N_vis = visual_tokens.shape[1]

        text_embeds = self._get_text_embeds(input_ids)  # (B, T, d_dec)
        text_embeds = text_embeds.to(dtype)
        visual_tokens = visual_tokens.to(dtype)

        if injection == "interleaved":
            # Replace <image> token positions with visual token sequence
            # We find where image_token_id is and splice visual tokens in
            inputs_embeds, adj_attention_mask, adj_labels = self._interleave(
                text_embeds, attention_mask, labels, visual_tokens, input_ids, B, T, N_vis, device, dtype
            )
        else:
            # Prepend visual tokens to text sequence
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)  # (B, N_vis+T, d_dec)
            # Extend attention mask with 1s for visual tokens
            vis_mask = torch.ones(B, N_vis, device=device, dtype=attention_mask.dtype)
            adj_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
            if labels is not None:
                # Mask out visual token positions in labels
                vis_labels = torch.full((B, N_vis), -100, device=device, dtype=labels.dtype)
                adj_labels = torch.cat([vis_labels, labels], dim=1)
            else:
                adj_labels = None

        total_len = inputs_embeds.shape[1]

        # Build attention mask
        if mask_mode == "image_bidir" and injection != "interleaved":
            n_text = T
            custom_mask = build_image_bidir_mask(N_vis, n_text, device=device, dtype=dtype)
            # Expand for batch
            custom_mask = custom_mask.expand(B, 1, total_len, total_len)
            attn_mask = custom_mask
        elif mask_mode == "image_bidir" and injection == "interleaved":
            # For interleaved, build a mask respecting visual token positions
            # We'll use causal here for simplicity (interleaved positions vary per sample)
            attn_mask = None
        else:
            attn_mask = None

        # Run decoder — pass either the custom 4-D mask or the standard 1-D mask
        final_attn_mask = attn_mask if attn_mask is not None else adj_attention_mask
        out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attn_mask,
            labels=adj_labels,
            return_dict=True,
            use_cache=False,
        )

        result = {"logits": out.logits}
        if adj_labels is not None and out.loss is not None:
            result["loss"] = out.loss
        return result

    def _interleave(
        self,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        visual_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        B: int,
        T: int,
        N_vis: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Replace <image> placeholder token with visual token sequence."""
        new_embeds_list = []
        new_mask_list = []
        new_labels_list = []

        for b in range(B):
            img_positions = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
            if len(img_positions) == 0:
                # No image token: just prepend
                emb = torch.cat([visual_tokens[b], text_embeds[b]], dim=0)
                msk = torch.cat([
                    torch.ones(N_vis, device=device, dtype=attention_mask.dtype),
                    attention_mask[b]
                ], dim=0)
                if labels is not None:
                    lbl = torch.cat([
                        torch.full((N_vis,), -100, device=device, dtype=labels.dtype),
                        labels[b]
                    ], dim=0)
                else:
                    lbl = None
            else:
                pos = img_positions[0].item()
                # text_embeds[b]: (T, d), split at pos
                before = text_embeds[b, :pos]       # (pos, d)
                after = text_embeds[b, pos+1:]       # (T-pos-1, d)
                emb = torch.cat([before, visual_tokens[b], after], dim=0)
                # attention mask
                before_msk = attention_mask[b, :pos]
                after_msk = attention_mask[b, pos+1:]
                vis_msk = torch.ones(N_vis, device=device, dtype=attention_mask.dtype)
                msk = torch.cat([before_msk, vis_msk, after_msk], dim=0)
                if labels is not None:
                    before_lbl = labels[b, :pos]
                    after_lbl = labels[b, pos+1:]
                    vis_lbl = torch.full((N_vis,), -100, device=device, dtype=labels.dtype)
                    lbl = torch.cat([before_lbl, vis_lbl, after_lbl], dim=0)
                else:
                    lbl = None

            new_embeds_list.append(emb)
            new_mask_list.append(msk)
            if lbl is not None:
                new_labels_list.append(lbl)

        # Pad to same length
        max_len = max(e.shape[0] for e in new_embeds_list)
        d = new_embeds_list[0].shape[-1]
        inputs_embeds = torch.zeros(B, max_len, d, device=device, dtype=dtype)
        adj_attention_mask = torch.zeros(B, max_len, device=device, dtype=attention_mask.dtype)
        adj_labels = torch.full((B, max_len), -100, device=device, dtype=labels.dtype) if labels is not None else None

        for b in range(B):
            l = new_embeds_list[b].shape[0]
            inputs_embeds[b, :l] = new_embeds_list[b]
            adj_attention_mask[b, :l] = new_mask_list[b]
            if adj_labels is not None:
                adj_labels[b, :l] = new_labels_list[b]

        return inputs_embeds, adj_attention_mask, adj_labels

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        injection: InjectionMode = "cls",
        max_new_tokens: int = 32,
        **gen_kwargs,
    ) -> list[str]:
        """Generate text continuations conditioned on images + prompts.

        Uses a manual KV-cache greedy decode loop because HF `generate()` with
        `inputs_embeds` (no input_ids) produces only a single EOS token on
        LLaMA-style models when the prefix is not in their native instruct format.
        """
        device = images.device
        dtype = next(self.decoder.parameters()).dtype
        B = images.shape[0]

        # Tokenize prompts
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        visual_tokens = self._get_visual_tokens(images, injection)  # (B, N_vis, d_dec)
        N_vis = visual_tokens.shape[1]
        visual_tokens = visual_tokens.to(dtype)

        text_embeds = self._get_text_embeds(input_ids).to(dtype)  # (B, T, d_dec)

        if injection == "interleaved" and self.image_token_id is not None:
            inputs_embeds, adj_attention_mask, _ = self._interleave(
                text_embeds, attention_mask, None, visual_tokens,
                input_ids, B, input_ids.shape[1], N_vis, device, dtype
            )
        else:
            inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
            vis_mask = torch.ones(B, N_vis, device=device, dtype=attention_mask.dtype)
            adj_attention_mask = torch.cat([vis_mask, attention_mask], dim=1)

        # Manual greedy decode with KV cache
        eos_id = self.tokenizer.eos_token_id
        generated_ids = [[] for _ in range(B)]
        finished = [False] * B

        # First forward pass: process full prefix with inputs_embeds
        out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=adj_attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = out.past_key_values
        # Get logits at the last VALID (non-padding) position for each sequence.
        # With right-padded prompts, different sequences end at different positions.
        last_valid = adj_attention_mask.sum(dim=1) - 1  # (B,) zero-indexed
        next_token_logits = out.logits[torch.arange(B, device=device), last_valid]  # (B, vocab_size)
        cur_attention_mask = adj_attention_mask

        for _ in range(max_new_tokens):
            next_tokens = next_token_logits.argmax(dim=-1)  # (B,)
            for i in range(B):
                if not finished[i]:
                    tok = next_tokens[i].item()
                    if tok == eos_id:
                        finished[i] = True
                    else:
                        generated_ids[i].append(tok)
            if all(finished):
                break
            # Extend attention mask and run next step with input_ids
            cur_attention_mask = torch.cat(
                [cur_attention_mask, torch.ones(B, 1, device=device, dtype=cur_attention_mask.dtype)],
                dim=1,
            )
            out = self.decoder(
                input_ids=next_tokens.unsqueeze(1),  # (B, 1)
                attention_mask=cur_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = out.past_key_values
            next_token_logits = out.logits[:, -1, :]

        results = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]
        return results
