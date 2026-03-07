# Post 4: 6-Model Ensemble Beats the Leaderboard (3.253 < 3.264)

## Result

**6-model ensemble val loss: 3.253** — beating the previous leaderboard best of 3.264.

## Ensemble Scaling Results

| Models | Val Loss | BPB | vs 3.264 |
|--------|----------|-----|----------|
| 1 (avg) | 3.372 | 1.096 | +0.108 |
| 2 | 3.305 | 1.074 | +0.041 |
| 4 | 3.265 | 1.061 | +0.001 |
| **6** | **3.253** | **1.057** | **-0.011** |
| 8 (projected) | ~3.24 | ~1.053 | ~-0.024 |

The ensemble improvement follows diminishing returns:
- 2→4 models: -0.040
- 4→6 models: -0.012

## Individual Model Results

| Model | Seed | WD | Dropout | LR | Val Loss |
|-------|------|-----|---------|-----|----------|
| 0 | 42 | 1.6 | 0.10 | 0.08 | 3.372 |
| 1 | 43 | 1.8 | 0.10 | 0.08 | 3.374 |
| 2 | 44 | 1.6 | 0.12 | 0.08 | 3.370 |
| 3 | 45 | 1.4 | 0.10 | 0.08 | 3.371 |
| 4 | 46 | 1.8 | 0.08 | 0.08 | 3.375 |
| 5 | 47 | 1.6 | 0.10 | 0.07 | **3.369** |

All models converge to ~3.37 despite varied hyperparameters. Model 5 (lower LR 0.07) achieved the best individual result.

## Total Compute

- 6 models × ~23h = 138 hours on 1xA100-80GB
- ~6 days of continuous training
- Equivalent to ~7 hours on 8xH100

## Infrastructure Notes

- Had to restart after model 4 due to OOM (torch compile cache growth)
- `--resume` flag successfully skipped completed models
- Fresh process starts with 66 GB memory vs 79 GB after 5 consecutive models

## Leaderboard Context

The current unlimited track best has moved to **3.218** (SwiGLU + value projections).
Our pure ensemble approach (no architectural changes) achieves **3.253**.
With SwiGLU + value projections applied to each member, we could likely reach ~3.20.

## Next Steps

- Models 6-7 still training (~2 more days)
- 8-model ensemble target: ~3.24
- Then: apply SwiGLU + value projections for further improvement
- Also: tiny track experiments when GPU is available
