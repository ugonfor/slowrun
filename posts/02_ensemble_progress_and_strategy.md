# Post 2: Ensemble Progress & Tiny Track Strategy

## Current Status (2 models done, model 2 training)

### Ensemble Results

| Model | Seed | WD | Dropout | Val Loss | Time |
|-------|------|-----|---------|----------|------|
| 0 | 42 | 1.6 | 0.1 | **3.372** | 22.8h |
| 1 | 43 | 1.8 | 0.1 | **3.374** | 22.8h |
| **2-model ensemble** | - | - | - | **3.305** | - |
| 2 (training) | 44 | 1.6 | 0.12 | (in progress) | - |

### Leaderboard Context

| Track | Current Best | Our Result | Status |
|-------|-------------|------------|--------|
| Unlimited (single) | 3.402 | **3.372** | **Beating it** |
| Unlimited (ensemble) | 3.264 | 3.305 (2 models) | Need more models |
| Unlimited (new best) | 3.218 | - | Target |
| Tiny (15 min) | 3.410 | Not yet tested | Next target |

## Upstream Changes Discovered

The original repo (qlabs-eng/slowrun) has been significantly updated since we forked:

### New Tracks
- **Tiny Track** (NEW): 15 min on 8xH100, 300M model. Baseline 3.428, best 3.410
- Ensemble track separated from single-model unlimited track
- Limited track now at 3.335 (SwiGLU activation)

### New Techniques in Leaderboard
1. **SwiGLU Activation**: Replaces `relu(x).square()` MLP with `silu(gate(x)) * fc(x)`. Used in limited (3.335), tiny (3.410), and unlimited (3.218) tracks
2. **Value Projections from x0**: Replaces value embedding lookup tables with learned projections from the initial hidden state. Used in limited (3.349) and unlimited (3.218) tracks

These two changes together pushed unlimited from 3.264 to **3.218** — our real target.

## Training Infrastructure

### Hardware
- 1x A100-SXM4-80GB (vs benchmark's 8xH100)
- Training is ~25x slower than 8xH100 per model
- Each 2.7B model takes ~23 hours

### Stability Fixes Applied
1. **FA3 → FA2**: Flash Attention 2 for Ampere GPUs
2. **rms_norm**: Manual implementation for PyTorch 2.3
3. **DataLoader shuffling**: Only shuffle batch order, not sequence composition
4. **NaN-safe optimizer**: Skip optimizer step on NaN/inf gradients

### Memory Profile
- Model 0-1: ~66 GB GPU memory (batch_size=4, 64 grad_accum steps)
- Model 2+: ~75 GB (higher due to torch compile cache growth)
- No room for concurrent tiny track training

## Epoch-by-Epoch Analysis (Model 0)

```
Epoch  Val Loss  Phase
  1     4.465    Learning
  2     4.068    Learning
  3     3.993    Learning
  4     3.953    Learning (slowing)
  5     3.915    Plateau
  6     3.934    Overfitting
  7     3.916    Oscillating
  8     3.946    Warmdown starts here (50% of training)
  9     3.897    LR decay kicks in
 10     3.809    Major improvement
 11     3.780
 12     3.685    Would match baseline epoch count
 13     3.633
 14     3.543
 15     3.445
 16     3.372    Final — beats baseline
```

**Key insight**: Epochs 1-8 contribute -0.52 improvement (full LR). Epochs 9-16 contribute -0.57 improvement (warmdown). The warmdown phase is where the magic happens — the model was overfitting during epochs 5-8, and LR decay lets it recover and generalize.

## Ensemble Scaling Projection

Based on our 2-model result (3.305) vs individual (~3.373):

| Models | Expected Val Loss | vs Leaderboard |
|--------|------------------|----------------|
| 2 | 3.305 | Above 3.264 |
| 4 | ~3.27 | Near 3.264 |
| 6 | ~3.25 | Below 3.264 |
| 8 | ~3.23 | Well below 3.264 |

Ensemble improvement follows ~1/sqrt(K) scaling. With 8 models we should comfortably beat 3.264.

However, the new unlimited best is **3.218** using SwiGLU + value projections on single models. To beat that with ensembling, we may need the architectural improvements too.

## Next Steps

1. **Continue ensemble training** — models 2-7, ~6 more days
2. **Prepare tiny track code** — already adapted for A100, needs testing
3. **Incorporate SwiGLU** — clear win from leaderboard evidence
4. **Consider value projections** — another leaderboard-proven technique
5. **Test on tiny track** when GPU becomes available

## Risk Assessment

- **Ensemble approach**: Reliable, diminishing returns, slow (7 days total)
- **Tiny track**: Fast iteration (300M model, ~2h on A100), competitive
- **Architectural improvements** (SwiGLU, value proj): High-value, can apply to both tracks
