"""§5 — Qualitative evaluation of a trained VLM.

Generates predictions on a held-out CLEVR sample and reports per-q_type
accuracy.

Usage:
    python scripts/eval_vlm.py \
        --checkpoint runs/vlm_all_patches_image_bidir_A/best.pt \
        --num-examples 10 --save-images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--num-examples", type=int, default=10,
                   help="Number of examples to dump for qualitative inspection")
    p.add_argument("--max-eval", type=int, default=500,
                   help="Number of examples to use for accuracy computation")
    p.add_argument("--save-images", action="store_true",
                   help="Save the example images alongside the JSON output")
    p.add_argument("--output-dir", type=Path, default=Path("runs/vlm_qualitative"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_clevr_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # -------------------------------------------------------------------------
    # 1. Load checkpoint and reconstruct VLM
    # -------------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vit_cfg = ckpt["vit_cfg"]
    injection = ckpt.get("injection", "all_patches")
    mask_mode = ckpt.get("mask_mode", "causal")
    image_token_id = ckpt.get("image_token_id", None)

    from basics.vit import ViT
    vit = ViT(**vit_cfg)
    # Load ViT weights: prefer inline state_dict, fall back to pretrained_vit_path
    if "vit_state_dict" in ckpt:
        vit.load_state_dict(ckpt["vit_state_dict"])
    else:
        vit_path = ckpt.get("pretrained_vit_path", "runs/clip_eurosat/best.pt")
        vit_ckpt = torch.load(vit_path, map_location="cpu", weights_only=False)
        vit.load_state_dict(vit_ckpt["vit_state_dict"])
    vit = vit.to(device).eval()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if image_token_id is not None:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})

    try:
        decoder = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    except Exception:
        decoder = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-360M-Instruct",
            torch_dtype=torch.bfloat16,
        )

    if image_token_id is not None:
        decoder.resize_token_embeddings(len(tokenizer))

    # Load decoder weights only if saved (configs C/D omit them to save disk)
    if "decoder_state_dict" in ckpt:
        decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder = decoder.to(device).eval()

    from vlm.projector import VisionLanguageProjector
    d_decoder = decoder.config.hidden_size
    projector = VisionLanguageProjector(
        d_image=vit_cfg["d_model"],
        d_decoder=d_decoder,
        expansion=4,
    )
    projector.load_state_dict(ckpt["projector_state_dict"])
    projector = projector.to(device).eval()

    from vlm.model import VisionLanguageModel
    vlm = VisionLanguageModel(vit, projector, decoder, tokenizer, image_token_id).eval()

    # -------------------------------------------------------------------------
    # 2. Load CLEVR data
    # -------------------------------------------------------------------------
    from vlm.data import CLEVRMiniDataset
    dataset = CLEVRMiniDataset(split=args.split, img_size=64)

    from torch.utils.data import DataLoader

    def collate(batch):
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "question": [b["question"] for b in batch],
            "answer": [b["answer"] for b in batch],
            "q_type": [b["q_type"] for b in batch],
            "raw_idx": list(range(len(batch))),
        }

    val_dl = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate)

    # -------------------------------------------------------------------------
    # 3. Run evaluation
    # -------------------------------------------------------------------------
    from vlm.eval import batch_clevr_accuracy

    all_preds = []
    all_golds = []
    all_qtypes = []
    all_questions = []
    eval_count = 0

    with torch.no_grad():
        for batch in val_dl:
            if eval_count >= args.max_eval:
                break
            images = batch["image"].to(device)
            questions = batch["question"]
            answers = batch["answer"]
            qtypes = batch["q_type"]

            prompts = [build_clevr_prompt(q) for q in questions]
            preds = vlm.generate(
                images, prompts, injection=injection,
                max_new_tokens=32,
            )
            import re
            clean_preds = []
            for pred in preds:
                tokens = pred.strip().split()
                first_word = tokens[0] if tokens else ""
                first_word = re.sub(r'^[\s\.\!\?\"\']+|[\s\.\!\?\"\']+$', '', first_word)
                clean_preds.append(first_word)

            all_preds.extend(clean_preds)
            all_golds.extend(answers)
            all_qtypes.extend(qtypes)
            all_questions.extend(questions)
            eval_count += len(answers)

    # -------------------------------------------------------------------------
    # 4. Compute accuracy
    # -------------------------------------------------------------------------
    acc_dict = batch_clevr_accuracy(all_preds, all_golds, all_qtypes)
    print(f"\nAccuracy breakdown:")
    for k, v in acc_dict.items():
        print(f"  {k}: {v:.4f}")

    # -------------------------------------------------------------------------
    # 5. Qualitative dump
    # -------------------------------------------------------------------------
    from vlm.eval import clevr_exact_match

    correct_examples = [(i, q, g, p) for i, (q, g, p) in enumerate(
        zip(all_questions, all_golds, all_preds)) if clevr_exact_match(p, g)]
    incorrect_examples = [(i, q, g, p) for i, (q, g, p) in enumerate(
        zip(all_questions, all_golds, all_preds)) if not clevr_exact_match(p, g)]

    # Pick a balanced mix
    import random
    random.seed(42)
    n_correct = min(args.num_examples // 2, len(correct_examples))
    n_incorrect = min(args.num_examples - n_correct, len(incorrect_examples))
    selected_correct = random.sample(correct_examples, n_correct)
    selected_incorrect = random.sample(incorrect_examples, n_incorrect)
    selected = selected_correct + selected_incorrect

    examples_out = []
    for (idx, question, gold, prediction) in selected:
        entry = {
            "idx": idx,
            "question": question,
            "gold": gold,
            "prediction": prediction,
            "correct": clevr_exact_match(prediction, gold),
        }
        if args.save_images:
            ex = dataset[idx]
            img_path = args.output_dir / f"example_{idx}.png"
            # Convert tensor back to PIL image for saving
            import torchvision.transforms.functional as TF
            import numpy as np
            # Unnormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = ex["image"] * std + mean
            img_tensor = img_tensor.clamp(0, 1)
            img_pil = TF.to_pil_image(img_tensor)
            img_pil.save(img_path)
            entry["image_file"] = str(img_path)
        examples_out.append(entry)

    out_file = args.output_dir / "examples.jsonl"
    with open(out_file, "w") as f:
        for e in examples_out:
            f.write(json.dumps(e) + "\n")

    # -------------------------------------------------------------------------
    # 6. Print summary table
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"{'Question':<45} {'Gold':<10} {'Pred':<10} {'OK'}")
    print(f"{'='*70}")
    for e in examples_out:
        q = e["question"][:43] + ".." if len(e["question"]) > 45 else e["question"]
        print(f"{q:<45} {e['gold']:<10} {e['prediction']:<10} {'✓' if e['correct'] else '✗'}")

    print(f"\nResults saved to {args.output_dir}")

    # Save summary
    with open(args.output_dir / "accuracy.json", "w") as f:
        json.dump(acc_dict, f, indent=2)


if __name__ == "__main__":
    main()
