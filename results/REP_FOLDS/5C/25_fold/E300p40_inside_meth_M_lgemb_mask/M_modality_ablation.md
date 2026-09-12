# Change M — Missing-modality robustness (inference-time)

| Condition | macro-F1 | 95% CI | Δ vs full |
|---|---|---|---|
| RNA+CNV+methy | 0.8283 | [0.8119, 0.8447] | +0.0000 |
| RNA+methy (no CNV) | 0.7906 | [0.7757, 0.8055] | -0.0377 |
| RNA+CNV (no methy) | 0.8049 | [0.7815, 0.8284] | -0.0234 |
| RNA only | 0.7237 | [0.6821, 0.7654] | -0.1046 |

**Methylation contribution (RNA+methy − RNA-only): +0.0668 macro-F1.**
