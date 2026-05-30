"""§5 — VLM training on CLEVR.

Usage:
    python scripts/train_vlm.py --config configs/vlm_clevr.yaml \
        --injection all_patches --mask-mode image_bidir \
        --freeze-config A --pretrained-vit runs/clip_eurosat/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pretrained-vit", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument(
        "--injection",
        choices=["cls", "all_patches", "interleaved"],
        default="all_patches",
    )
    p.add_argument(
        "--mask-mode",
        choices=["causal", "image_bidir"],
        default="causal",
    )
    p.add_argument(
        "--freeze-config",
        choices=["A", "B", "C", "D"],
        default="A",
        help="Per writeup §5.6: A=projector only, B=+decoder LoRA, "
             "C=+full decoder, D=all three.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_clevr_prompt(question: str, tokenizer) -> str:
    """Format a CLEVR question as a VQA prompt."""
    return f"Question: {question}\nAnswer:"


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = (
            Path("runs") / f"vlm_{args.injection}_{args.mask_mode}_{args.freeze_config}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # -------------------------------------------------------------------------
    # 1. Build CLEVR loaders
    # -------------------------------------------------------------------------
    from vlm.data import build_clevr_loaders
    train_dl, val_dl = build_clevr_loaders(
        img_size=64,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    # -------------------------------------------------------------------------
    # 2. Load CLIP-pretrained ViT
    # -------------------------------------------------------------------------
    from basics.vit import ViT
    ckpt = torch.load(args.pretrained_vit, map_location="cpu")
    vit_cfg = ckpt["vit_cfg"]
    vit = ViT(**vit_cfg)
    vit.load_state_dict(ckpt["vit_state_dict"])
    vit = vit.to(device)

    # -------------------------------------------------------------------------
    # 3. Load SmolLM2-360M decoder
    # -------------------------------------------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    decoder_cfg = cfg["decoder"]
    tokenizer = AutoTokenizer.from_pretrained(decoder_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add <image> token for interleaved mode
    image_token_id = None
    if args.injection == "interleaved":
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    torch_dtype = getattr(torch, decoder_cfg["torch_dtype"])
    try:
        decoder = AutoModelForCausalLM.from_pretrained(
            decoder_cfg["model_name"],
            torch_dtype=torch_dtype,
            attn_implementation=decoder_cfg["attn_implementation"],
        )
    except Exception:
        # Fall back without flash attention
        decoder = AutoModelForCausalLM.from_pretrained(
            decoder_cfg["model_name"],
            torch_dtype=torch_dtype,
        )

    if args.injection == "interleaved":
        decoder.resize_token_embeddings(len(tokenizer))

    decoder = decoder.to(device)

    # -------------------------------------------------------------------------
    # 4. Build projector and VLM
    # -------------------------------------------------------------------------
    from vlm.projector import VisionLanguageProjector
    from vlm.model import VisionLanguageModel

    d_decoder = decoder.config.hidden_size
    projector = VisionLanguageProjector(
        d_image=vit_cfg["d_model"],
        d_decoder=d_decoder,
        expansion=cfg["projector"]["expansion"],
    ).to(device)

    vlm = VisionLanguageModel(vit, projector, decoder, tokenizer, image_token_id)

    # -------------------------------------------------------------------------
    # 5. Apply freeze configuration
    # -------------------------------------------------------------------------
    # First freeze everything
    for p in vlm.parameters():
        p.requires_grad = False

    # Projector is always trained
    for p in projector.parameters():
        p.requires_grad = True

    if args.freeze_config == "A":
        pass  # only projector trained

    elif args.freeze_config == "B":
        # Decoder LoRA (wrap SmolLM2 q_proj/v_proj with LoRALinear)
        from basics.lora import LoRALinear
        for name, module in decoder.named_modules():
            if hasattr(module, "q_proj") and isinstance(module.q_proj, torch.nn.Linear):
                module.q_proj = LoRALinear(module.q_proj, rank=8, alpha=16.0).to(device)
            if hasattr(module, "v_proj") and isinstance(module.v_proj, torch.nn.Linear):
                module.v_proj = LoRALinear(module.v_proj, rank=8, alpha=16.0).to(device)

    elif args.freeze_config == "C":
        # Full decoder FT
        for p in decoder.parameters():
            p.requires_grad = True

    elif args.freeze_config == "D":
        # Everything trainable
        for p in vit.parameters():
            p.requires_grad = True
        for p in decoder.parameters():
            p.requires_grad = True

    trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
    print(f"Freeze config {args.freeze_config}: {trainable_params:,} trainable params")

    # -------------------------------------------------------------------------
    # 6. Training setup
    # -------------------------------------------------------------------------
    params = [p for p in vlm.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        params,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )
    num_steps = cfg["train"]["num_steps"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg["optim"]["warmup_steps"], num_steps)

    grad_accum = cfg["train"].get("gradient_accumulation_steps", 1)
    gen_cfg = cfg.get("generation", {})

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # -------------------------------------------------------------------------
    # 7. Training loop
    # -------------------------------------------------------------------------
    from vlm.eval import batch_clevr_accuracy

    best_val_acc = 0.0
    step = 0
    optimizer.zero_grad()
    train_iter = iter(train_dl)

    while step < num_steps:
        vlm.vit.train() if args.freeze_config == "D" else vlm.vit.eval()
        vlm.projector.train()
        vlm.decoder.train() if args.freeze_config in ("C", "D") else vlm.decoder.eval()

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        images = batch["image"].to(device)
        questions = batch["question"]
        answers = batch["answer"]
        B = images.shape[0]

        # Build prompts + target sequences
        prompts = [build_clevr_prompt(q, tokenizer) for q in questions]
        full_texts = [f"{p} {a}" for p, a in zip(prompts, answers)]

        # Tokenize full text
        enc_full = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        enc_prompt = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        input_ids = enc_full["input_ids"].to(device)
        attention_mask = enc_full["attention_mask"].to(device)

        # Build labels: mask everything except answer tokens
        # Find prompt lengths (without padding)
        prompt_lens = enc_prompt["attention_mask"].sum(dim=1)  # (B,)

        labels = input_ids.clone()
        for b in range(B):
            # Mask prompt tokens (not answer)
            labels[b, :prompt_lens[b]] = -100
            # Mask padding
            labels[b, attention_mask[b] == 0] = -100

        out = vlm(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            injection=args.injection,
            mask_mode=args.mask_mode,
        )

        loss = out["loss"] / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        step += 1

        if step % cfg["train"]["log_every"] == 0:
            print(f"Step {step}/{num_steps}  loss={loss.item() * grad_accum:.4f}")

        # -------------------------------------------------------------------------
        # 8. Periodic validation
        # -------------------------------------------------------------------------
        if step % cfg["train"]["eval_every_steps"] == 0:
            vlm.vit.eval()
            vlm.projector.eval()
            vlm.decoder.eval()

            all_preds = []
            all_golds = []
            all_qtypes = []
            eval_count = 0
            max_eval = cfg["train"]["eval_max_examples"]

            with torch.no_grad():
                for val_batch in val_dl:
                    if eval_count >= max_eval:
                        break
                    val_images = val_batch["image"].to(device)
                    val_questions = val_batch["question"]
                    val_answers = val_batch["answer"]
                    val_qtypes = val_batch["q_type"]

                    val_prompts = [build_clevr_prompt(q, tokenizer) for q in val_questions]
                    preds = vlm.generate(
                        val_images, val_prompts, injection=args.injection,
                        max_new_tokens=gen_cfg.get("max_new_tokens", 32),
                    )
                    # CLEVR answers are always single words — extract first word
                    import re
                    clean_preds = []
                    for pred in preds:
                        tokens = pred.strip().split()
                        first_word = tokens[0] if tokens else ""
                        first_word = re.sub(r'^[\s\.\!\?\"\']+|[\s\.\!\?\"\']+$', '', first_word)
                        clean_preds.append(first_word)

                    all_preds.extend(clean_preds)
                    all_golds.extend(val_answers)
                    all_qtypes.extend(val_qtypes)
                    eval_count += len(val_answers)

            acc_dict = batch_clevr_accuracy(all_preds, all_golds, all_qtypes)
            val_acc = acc_dict["overall"]
            peak_mem = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0

            print(f"Step {step}  val_acc={val_acc:.4f}  peak_mem={peak_mem/1e6:.0f}MB")

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                # Only save components that are trainable to keep checkpoints small.
                # The frozen decoder can always be reloaded from HuggingFace.
                ckpt_dict = {
                    "step": step,
                    "projector_state_dict": vlm.projector.state_dict(),
                    "vit_cfg": vit_cfg,
                    "injection": args.injection,
                    "mask_mode": args.mask_mode,
                    "freeze_config": args.freeze_config,
                    "val_acc": val_acc,
                    "image_token_id": image_token_id,
                }
                # Include ViT only if it has been unfrozen (config D)
                vit_trainable = any(p.requires_grad for p in vlm.vit.parameters())
                if vit_trainable:
                    ckpt_dict["vit_state_dict"] = vlm.vit.state_dict()
                else:
                    # Store the pretrained path so we can reload it
                    ckpt_dict["pretrained_vit_path"] = str(args.pretrained_vit)
                # Never save the decoder — it is either frozen (configs A/B-frozen-base)
                # or a full 360M-param model that fills the disk (configs C/D).
                # Val accuracies are recorded in metrics.json, which is sufficient for
                # the writeup. The projector weights here are all that's needed for
                # lightweight checkpoint reloading.
                torch.save(ckpt_dict, args.output_dir / "best.pt")

    # -------------------------------------------------------------------------
    # 9. Final metrics
    # -------------------------------------------------------------------------
    peak_mem = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    metrics = {
        "injection": args.injection,
        "mask_mode": args.mask_mode,
        "freeze_config": args.freeze_config,
        "best_val_acc": best_val_acc,
        "trainable_params": trainable_params,
        "peak_gpu_memory_bytes": peak_mem,
        "n_visual_tokens": (vit_cfg["img_size"] // vit_cfg["patch_size"]) ** 2 + 1
            if args.injection != "cls" else 1,
    }
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Saved checkpoint to {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
