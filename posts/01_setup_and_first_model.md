# Post 1: A100 Adaptation & First Model Results

## Summary

Adapted the NanoGPT Slowrun benchmark (unlimited track) to run on a single A100-80GB GPU, trained the first ensemble member, and achieved **val loss 3.372** — beating the baseline single-model score of 3.402.

## The Benchmark

NanoGPT Slowrun trains a language model on 100M FineWeb tokens with no compute/time limit. Lowest validation loss wins. Two tracks:
- **Limited**: 8xH100, 1 hour max. Best: 3.376
- **Unlimited**: No restrictions. Best: 3.264 (8-model ensemble with logit averaging)

## Our Hardware

Single A100-SXM4-80GB, 96 CPUs, 1.7TB RAM. The original code requires 8xH100 with Flash Attention 3 (Hopper-only).

## Key Adaptations

### Flash Attention 2 (instead of FA3)
The original code requires FA3 which only works on Hopper GPUs (H100). We replaced it with FA2 which works on Ampere (A100). The API is identical — `flash_attn_func(q, k, v, causal=True, window_size=...)`.

### PyTorch 2.3 Compatibility
- `F.rms_norm` doesn't exist in PyTorch 2.3. Replaced with manual implementation: `x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)`
- `torch.uint16` serialization unsupported. Fixed data preparation to use `int32`.
- `torch.compile` requires `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` due to protobuf/onnx conflict.

### Single-GPU Training Stability

This was the hardest challenge. Three issues discovered and fixed:

**1. DataLoader Shuffling (NaN at step ~110)**
The ensemble script shuffled individual sequences into new batch compositions each epoch. This created unlucky gradient patterns causing NaN. Fix: only shuffle batch *order*, not sequence assignment — matching the original train.py.

**2. Memory Management (OOM at epoch boundary)**
`batch_size=8` used 79/80 GB, leaving no headroom. The validation phase after epoch 1 triggered OOM. Fix: reduced to `batch_size=4` (65 GB used, 15 GB headroom).

**3. Gradient Explosion (NaN at step ~1140, epoch 6)**
With 1 GPU, gradient accumulation uses 64 micro-steps (vs 8 on 8xH100). The DDP gradient averaging across ranks acts as implicit smoothing that stabilizes training. Without it, rare gradient spikes cause NaN. Fix: NaN-safe optimizer — detect NaN/inf in gradients and skip the optimizer step.

## Training Results: Model 0

Architecture: 2.7B params (1.2B transformer trunk), 30 layers, 1792 dim, 14 heads.
Hyperparams: Muon optimizer, WD=1.6, dropout=0.1, 16 epochs, batch_size=4.

| Epoch | Val Loss | Val BPB | Notes |
|-------|----------|---------|-------|
| 1     | 4.465    | 1.451   | Initial learning |
| 2     | 4.068    | 1.322   | Rapid improvement |
| 3     | 3.993    | 1.298   | Slowing down |
| 4     | 3.953    | 1.285   | |
| 5     | 3.915    | 1.272   | Pre-warmdown best |
| 6     | 3.934    | 1.278   | Overfitting begins |
| 7     | 3.916    | 1.273   | |
| 8     | 3.946    | 1.282   | Warmdown starts (LR decay) |
| 9     | 3.897    | 1.266   | LR decay kicks in |
| 10    | 3.809    | 1.238   | |
| 11    | 3.780    | 1.229   | |
| 12    | 3.685    | 1.198   | Matches baseline epoch count |
| 13    | 3.633    | 1.181   | |
| 14    | 3.543    | 1.151   | |
| 15    | 3.445    | 1.120   | |
| **16**| **3.372**| **1.096**| **Final — beats baseline 3.402** |

Training time: 22.8 hours on 1xA100.

### Key Insight: Warmdown is Everything

The model plateaus around val loss 3.91-3.95 for epochs 5-8, then the warmdown (LR cosine decay from 50% to 100% of training) drives massive improvement:
- Epochs 1-8 (full LR): 4.465 → 3.946 (-0.52)
- Epochs 9-16 (warmdown): 3.897 → 3.372 (-0.53)

Half the total improvement comes from the warmdown phase. Training 16 epochs instead of 12 gives the warmdown more time to work, which is why we beat the 12-epoch baseline.

## Research Background

Based on Kim et al. 2025 "Pre-training under Infinite Compute":
- Larger models with heavy regularization are optimal in the data-limited regime
- Ensemble of K models (logit averaging) follows a power law improvement in K
- For ensemble members: ~2x epochs + ~0.5x weight decay is optimal (though we found WD=0.8 diverges with current LRs — stuck with WD=1.6)
- Data repetition has a half-life of ~16 epochs

## Next Steps

- Training 7 more ensemble members with diverse hyperparameters
- Each model takes ~23 hours on 1xA100
- Ensemble evaluation after every 2 models
- Target: beat 3.264 (current unlimited track leader)

## Files

- `run_ensemble.py` — Main ensemble training script
- `evaluate_ensemble.py` — Standalone ensemble evaluation
- `train.py` — Adapted single-model training (FA2 compatible)
- `unlimited/train.py` — Adapted original ensemble script
