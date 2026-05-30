#!/usr/bin/env bash
# Run remaining HW3 experiments sequentially (after current jobs finish).
set -euo pipefail
cd "$(dirname "$0")/.."

PRETRAINED="runs/clip_eurosat/best.pt"

echo "=== LoRA rank sweep (16, 32, 64) ==="
for r in 16 32 64; do
  if [ ! -f "runs/resisc_lora_rank${r}/metrics.json" ]; then
    python3 scripts/finetune_resisc.py \
      --config configs/lora_resisc.yaml --method lora \
      --rank "$r" --alpha $((2*r)) \
      --pretrained "$PRETRAINED" \
      --output-dir "runs/resisc_lora_rank${r}"
  fi
done

echo "=== VLM injection strategies (2000 steps, config A) ==="
for inj in all_patches interleaved; do
  out="runs/vlm_${inj}_causal_A"
  if [ ! -f "${out}/metrics.json" ]; then
    python3 scripts/train_vlm.py --config configs/vlm_clevr.yaml \
      --pretrained-vit "$PRETRAINED" \
      --injection "$inj" --mask-mode causal --freeze-config A \
      --output-dir "$out"
  fi
done

echo "=== VLM masking comparison (500 steps, all_patches) ==="
for mask in causal image_bidir; do
  out="runs/vlm_all_patches_${mask}_500"
  if [ ! -f "${out}/metrics.json" ]; then
    cfg="/tmp/vlm_mask_${mask}_500.yaml"
    sed 's/num_steps: 2000/num_steps: 500/' configs/vlm_clevr.yaml > "$cfg"
    python3 scripts/train_vlm.py --config "$cfg" \
      --pretrained-vit "$PRETRAINED" \
      --injection all_patches --mask-mode "$mask" --freeze-config A \
      --output-dir "$out"
  fi
done

echo "=== VLM freeze configs B, C, D (2000 steps, all_patches) ==="
for cfg in B C D; do
  out="runs/vlm_all_patches_causal_${cfg}"
  if [ ! -f "${out}/metrics.json" ]; then
    python3 scripts/train_vlm.py --config configs/vlm_clevr.yaml \
      --pretrained-vit "$PRETRAINED" \
      --injection all_patches --mask-mode causal --freeze-config "$cfg" \
      --output-dir "$out"
  fi
done

echo "=== Done ==="
