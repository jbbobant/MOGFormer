# Change M — Missing-modality robustness (inference-time)

| Condition | macro-F1 | 95% CI | Δ vs full |
|---|---|---|---|
| RNA+CNV+methy | 0.8267 | [0.8007, 0.8528] | +0.0000 |
| RNA+methy (no CNV) | 0.7827 | [0.7558, 0.8095] | -0.0441 |
| RNA+CNV (no methy) | 0.8067 | [0.7686, 0.8448] | -0.0201 |
| RNA only | 0.7197 | [0.6599, 0.7795] | -0.1071 |

**Methylation contribution (RNA+methy − RNA-only): +0.0630 macro-F1.**
