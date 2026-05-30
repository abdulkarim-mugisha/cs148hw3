"""§4 — Compare full FT, LoRA, and linear probe on RESISC45.

Usage:
    python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \
        --method lora --rank 8 --pretrained runs/clip_eurosat/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", choices=["linear_probe", "lora", "full_ft"], required=True)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (only for --method lora)")
    p.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha (only for --method lora)")
    p.add_argument("--pretrained", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
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


def evaluate(model, head, loader, device):
    model.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            feats = model(images)
            logits = head(feats)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = Path("runs") / f"resisc_{args.method}_rank{args.rank}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # -------------------------------------------------------------------------
    # 1. Build RESISC45 loaders
    # -------------------------------------------------------------------------
    from vlm.data import build_resisc45_loaders
    train_dl, test_dl = build_resisc45_loaders(
        img_size=64,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    # -------------------------------------------------------------------------
    # 2. Load pretrained ViT
    # -------------------------------------------------------------------------
    from basics.vit import ViT
    ckpt = torch.load(args.pretrained, map_location="cpu")
    vit_cfg = ckpt["vit_cfg"]
    vit = ViT(**vit_cfg).to(device)
    vit.load_state_dict(ckpt["vit_state_dict"])

    # -------------------------------------------------------------------------
    # 3. Apply adaptation strategy
    # -------------------------------------------------------------------------
    num_classes = cfg["num_classes"]
    d_model = vit_cfg["d_model"]

    if args.method == "linear_probe":
        # Freeze ViT entirely
        for p in vit.parameters():
            p.requires_grad = False
        head = nn.Linear(d_model, num_classes).to(device)
        lr = cfg["methods"]["linear_probe"].get("lr", cfg["optim"]["lr"])
        params = list(head.parameters())

    elif args.method == "lora":
        from basics.lora import apply_lora_to_attention
        apply_lora_to_attention(vit, args.rank, args.alpha)
        vit = vit.to(device)  # re-move so new LoRA params go to device
        head = nn.Linear(d_model, num_classes).to(device)
        lr = cfg["methods"]["lora"].get("lr", cfg["optim"]["lr"])
        params = [p for p in vit.parameters() if p.requires_grad] + list(head.parameters())

    elif args.method == "full_ft":
        # All ViT params + head
        head = nn.Linear(d_model, num_classes).to(device)
        lr = cfg["methods"]["full_ft"].get("lr", cfg["optim"]["lr"])
        params = list(vit.parameters()) + list(head.parameters())

    # -------------------------------------------------------------------------
    # 4. Train
    # -------------------------------------------------------------------------
    trainable_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"Method: {args.method}  Trainable params: {trainable_params:,}")

    optimizer = optim.AdamW(
        params,
        lr=lr,
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )
    num_epochs = cfg["train"]["num_epochs"]
    steps_per_epoch = len(train_dl)
    total_steps = num_epochs * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg["optim"]["warmup_steps"], total_steps)

    criterion = nn.CrossEntropyLoss()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        vit.train()
        head.train()
        for step, (images, labels) in enumerate(train_dl, 1):
            images, labels = images.to(device), labels.to(device)
            feats = vit(images)
            logits = head(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step % cfg["train"]["log_every"] == 0:
                print(f"Epoch {epoch}/{num_epochs} step {step}  loss={loss.item():.4f}")

        test_acc = evaluate(vit, head, test_dl, device)
        print(f"Epoch {epoch}/{num_epochs}  test_acc={test_acc:.4f}")

    wall_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0

    # -------------------------------------------------------------------------
    # 5. Save metrics
    # -------------------------------------------------------------------------
    metrics = {
        "method": args.method,
        "rank": args.rank,
        "alpha": args.alpha,
        "test_accuracy": test_acc,
        "trainable_params": trainable_params,
        "peak_gpu_memory_bytes": peak_mem,
        "wall_time_seconds": wall_time,
    }
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults:")
    print(f"  Test accuracy:       {test_acc:.4f}")
    print(f"  Trainable params:    {trainable_params:,}")
    print(f"  Peak GPU memory:     {peak_mem / 1e6:.1f} MB")
    print(f"  Wall-clock time:     {wall_time:.1f}s")
    print(f"  Saved to:            {args.output_dir}/metrics.json")


def run_rank_sweep() -> None:
    """Helper to run LoRA rank sweep (§4 rank sweep problem)."""
    import subprocess
    import sys
    ranks = [1, 2, 4, 8, 16, 32, 64]
    results = {}
    for r in ranks:
        print(f"\n=== LoRA rank={r} ===")
        ret = subprocess.run([
            sys.executable, __file__,
            "--config", "configs/lora_resisc.yaml",
            "--method", "lora",
            "--rank", str(r),
            "--alpha", str(2 * r),
            "--pretrained", "runs/clip_eurosat/best.pt",
            "--output-dir", f"runs/resisc_lora_rank{r}",
        ], check=True)
    print("Rank sweep complete.")


if __name__ == "__main__":
    main()
