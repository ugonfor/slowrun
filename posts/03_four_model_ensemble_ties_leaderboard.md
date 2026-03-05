# Post 3: 4-Model Ensemble Ties the Leaderboard (3.265)

## Result

**4-model ensemble val loss: 3.265** — within 0.001 of the leaderboard's best (3.264).

## Individual Model Results

| Model | Seed | WD | Dropout | Val Loss | Time |
|-------|------|-----|---------|----------|------|
| 0 | 42 | 1.6 | 0.10 | 3.372 | 22.8h |
| 1 | 43 | 1.8 | 0.10 | 3.374 | 22.8h |
| 2 | 44 | 1.6 | 0.12 | 3.370 | 23.0h |
| 3 | 45 | 1.4 | 0.10 | 3.371 | 22.8h |

All models converge to remarkably similar val loss (~3.37) despite different hyperparameters.

## Ensemble Scaling

| Checkpoints | Val Loss | Improvement |
|-------------|----------|-------------|
| 2 | 3.305 | -0.068 from single |
| 4 | 3.265 | -0.040 from 2-model |
| 8 (projected) | ~3.23 | Should beat 3.264 |

## Key Finding: WD=1.4 Wins During Training But Ties at End

Model 3 (WD=1.4) had significantly better val loss during epochs 10-15:
- Epoch 13: 3.590 vs 3.633 (model 0, WD=1.6)
- Epoch 14: 3.498 vs 3.543

But at epoch 16, all models converged to ~3.37. The warmdown LR schedule equalizes models with different regularization strength.

## Total Compute

- 4 models × 22.8h = 91.2 hours on 1xA100-80GB
- Equivalent to ~4.5 hours on 8xH100
- 4 more models training, ~4 more days to complete

## Next Steps

- 4 more models (models 4-7) training sequentially
- 8-model ensemble should achieve ~3.23
- Then available to try tiny track improvements
