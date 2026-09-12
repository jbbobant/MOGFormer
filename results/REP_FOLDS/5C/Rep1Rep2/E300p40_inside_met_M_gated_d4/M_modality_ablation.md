# Change M — Missing-modality robustness (inference-time)

| Condition | macro-F1 | 95% CI | Δ vs full |
|---|---|---|---|
| RNA+CNV+methy | 0.8154 | [0.7826, 0.8482] | +0.0000 |
| RNA only | 0.6123 | [0.5425, 0.6821] | -0.2031 |
| CNV only | 0.1574 | [0.1051, 0.2097] | -0.6580 |
| methy only | 0.1132 | [0.0589, 0.1675] | -0.7022 |
| RNA+CNV (no methy) | 0.7964 | [0.7616, 0.8312] | -0.0190 |
| RNA+methy (no CNV) | 0.7867 | [0.7308, 0.8425] | -0.0287 |
| CNV+methy (no RNA) | 0.2054 | [0.1515, 0.2593] | -0.6100 |

**Methylation contribution (RNA+methy − RNA-only): +0.1744 macro-F1.**
