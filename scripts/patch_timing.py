"""§2.4 — Patch size timing experiment."""
import time
import torch
import sys
sys.path.insert(0, '/workspace/generative/cs148hw3')
from basics.vit import ViT

device = torch.device("cuda")
img_size = 224
d_model = 384
num_heads = 6
num_blocks = 6
batch_size = 16

results = {}
for patch_size in [8, 16, 32]:
    num_patches = (img_size // patch_size) ** 2
    model = ViT(img_size=img_size, patch_size=patch_size, d_model=d_model,
                num_heads=num_heads, num_blocks=num_blocks, dropout=0.0).to(device)
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
    
    # Timing
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    
    import statistics
    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times)
    results[patch_size] = (num_patches, mean_t, std_t)
    print(f"P={patch_size:2d}: N={num_patches:4d} patches  time={mean_t:.2f} ± {std_t:.2f} ms")
    del model

print("\nResults table:")
print(f"{'P':>4} | {'N patches':>10} | {'Time (ms)':>15}")
print("-" * 35)
for p, (n, mean_t, std_t) in sorted(results.items()):
    print(f"{p:>4} | {n:>10} | {mean_t:>8.2f} ± {std_t:.2f}")
