"""§3 — CLIP-style pretraining on EuroSAT.

Usage:
    python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--output-dir", type=Path, default=Path("runs/clip_eurosat"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    p.add_argument("--pe", choices=["learned", "rope1d", "rope2d"], default="learned",
                   help="Positional encoding type (for §6 ablations)")
    return p.parse_args()


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # -------------------------------------------------------------------------
    # 1. Build data loaders
    # -------------------------------------------------------------------------
    from vlm.data import build_eurosat_loaders, EUROSAT_CLASSES
    train_dl, val_dl, test_dl = build_eurosat_loaders(
        img_size=cfg["vit"]["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    # -------------------------------------------------------------------------
    # 2. Build ViT and frozen text encoder
    # -------------------------------------------------------------------------
    from basics.vit import ViT
    from basics.text_encoder import FrozenTextEncoder

    vit_kwargs = dict(cfg["vit"])
    if args.pe != "learned":
        vit_kwargs["pos_enc"] = args.pe
    vit = ViT(**vit_kwargs).to(device)
    text_encoder = FrozenTextEncoder(cfg["text_encoder"]["model_name"])
    text_encoder.eval()

    # -------------------------------------------------------------------------
    # 3. Build projection heads + logit scale
    # -------------------------------------------------------------------------
    from vlm.clip import ProjectionHeads, init_logit_scale
    import math

    d_text = text_encoder.embedding_dim
    projection_heads = ProjectionHeads(
        d_image=cfg["vit"]["d_model"],
        d_text=d_text,
        d_proj=cfg["projection"]["d_proj"],
    ).to(device)
    logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), device=device))

    # -------------------------------------------------------------------------
    # 4. Optimizer + scheduler
    # -------------------------------------------------------------------------
    params = list(vit.parameters()) + list(projection_heads.parameters()) + [logit_scale]
    optimizer = optim.AdamW(
        params,
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )
    num_epochs = cfg["train"]["num_epochs"]
    steps_per_epoch = len(train_dl)
    total_steps = num_epochs * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, cfg["optim"]["warmup_steps"], total_steps)

    # -------------------------------------------------------------------------
    # 5. Class prompts for zero-shot evaluation
    # -------------------------------------------------------------------------
    class_prompts = [f"a satellite image of {c}" for c in EUROSAT_CLASSES]
    class_indices = list(range(len(EUROSAT_CLASSES)))

    from vlm.clip import clip_loss
    from vlm.eval import zeroshot_classification_accuracy

    # W&B setup
    if args.wandb:
        import wandb
        wandb.init(project="cs148-clip-eurosat", config=cfg)

    best_val_acc = 0.0
    train_losses = []
    val_accs = []

    # -------------------------------------------------------------------------
    # 6. Training loop
    # -------------------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        vit.train()
        projection_heads.train()
        epoch_loss = 0.0
        step = 0

        for batch_idx, (images, captions) in enumerate(train_dl):
            images = images.to(device)
            # Text embeddings (frozen, detach so they can be used in backward)
            with torch.no_grad():
                text_embeds = text_encoder(captions)
                if text_embeds.device != device:
                    text_embeds = text_embeds.to(device)
                text_embeds = text_embeds.float().clone()

            # Image embeddings
            image_feats = vit(images)  # (B, d_model)
            image_proj, text_proj = projection_heads(image_feats, text_embeds)

            loss = clip_loss(image_proj, text_proj, logit_scale)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

            # Clamp logit scale
            logit_scale.data.clamp_(max=math.log(100.0))

            epoch_loss += loss.item()
            step += 1

            if step % cfg["train"]["log_every"] == 0:
                print(f"Epoch {epoch}/{num_epochs} step {step}/{steps_per_epoch}  loss={loss.item():.4f}")

        avg_loss = epoch_loss / step
        train_losses.append(avg_loss)

        # Validation zero-shot accuracy
        val_acc = zeroshot_classification_accuracy(
            vit, projection_heads, text_encoder,
            val_dl, class_prompts, class_indices, device,
        )
        val_accs.append(val_acc)
        print(f"Epoch {epoch}/{num_epochs}  avg_loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        if args.wandb:
            import wandb
            wandb.log({"epoch": epoch, "train_loss": avg_loss, "val_acc": val_acc})

        # Save best checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "vit_state_dict": vit.state_dict(),
                "projection_heads_state_dict": projection_heads.state_dict(),
                "logit_scale": logit_scale.data,
                "vit_cfg": cfg["vit"],
                "train_losses": train_losses,
                "val_accs": val_accs,
            }, args.output_dir / "best.pt")

    # Save checkpoint with pe info
    ckpt = {
        "epoch": num_epochs,
        "vit_state_dict": vit.state_dict(),
        "projection_heads_state_dict": projection_heads.state_dict(),
        "logit_scale": logit_scale.data,
        "vit_cfg": cfg["vit"],
        "pe": args.pe,
        "train_losses": train_losses,
        "val_accs": val_accs,
    }
    torch.save(ckpt, args.output_dir / "final.pt")

    # -------------------------------------------------------------------------
    # 7. Extrapolation evaluation at 96×96
    # -------------------------------------------------------------------------
    print("\n--- Extrapolation eval at 96×96 ---")
    try:
        from vlm.data import build_eurosat_loaders
        _, _, test_dl_96 = build_eurosat_loaders(
            img_size=96,
            batch_size=cfg["train"]["batch_size"],
            num_workers=cfg["train"]["num_workers"],
        )
        acc_96 = zeroshot_classification_accuracy(
            vit, projection_heads, text_encoder,
            test_dl_96, class_prompts, class_indices, device,
        )
        print(f"96×96 zero-shot accuracy: {acc_96:.4f}")
        import json
        with open(args.output_dir / "extrap_acc_96.json", "w") as f:
            json.dump({"acc_96": acc_96, "pe": args.pe}, f)
    except Exception as e:
        print(f"Extrapolation eval failed: {e}")

    # -------------------------------------------------------------------------
    # 8. Plot training curves
    # -------------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(range(1, len(train_losses)+1), train_losses, marker="o")
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.plot(range(1, len(val_accs)+1), val_accs, marker="o", color="green")
        ax2.set_title("Zero-Shot Val Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(args.output_dir / "training_curves.png", dpi=150)
        plt.close()
        print(f"Saved training curves to {args.output_dir}/training_curves.png")
    except Exception as e:
        print(f"Could not save plot: {e}")

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
