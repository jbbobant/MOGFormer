# Change M — Missing-modality robustness (inference-time)

| Condition | macro-F1 | 95% CI | Δ vs full |
|---|---|---|---|
| RNA+CNV+methy | 0.8209 | [0.7994, 0.8424] | +0.0000 |
| RNA only | 0.5686 | [0.4316, 0.7056] | -0.2523 |
| CNV only | 0.1397 | [0.0925, 0.1869] | -0.6812 |
| methy only | 0.1132 | [0.0705, 0.1560] | -0.7077 |
| RNA+CNV (no methy) | 0.7617 | [0.7265, 0.7970] | -0.0592 |
| RNA+methy (no CNV) | 0.7561 | [0.7013, 0.8110] | -0.0648 |
| CNV+methy (no RNA) | 0.2094 | [0.1522, 0.2666] | -0.6115 |

**Methylation contribution (RNA+methy − RNA-only): +0.1875 macro-F1.**
