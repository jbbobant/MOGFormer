# Change M — Missing-modality robustness (inference-time)

| Condition | macro-F1 | 95% CI | Δ vs full |
|---|---|---|---|
| RNA+CNV+methy | 0.8135 | [0.7812, 0.8457] | +0.0000 |
| RNA+methy (no CNV) | 0.7762 | [0.7583, 0.7941] | -0.0373 |
| RNA+CNV (no methy) | 0.8006 | [0.7705, 0.8307] | -0.0129 |
| RNA only | 0.7367 | [0.6534, 0.8200] | -0.0768 |

**Methylation contribution (RNA+methy − RNA-only): +0.0395 macro-F1.**
