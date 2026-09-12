# MOGFormer Phase 4 — Confirmatory partition & verdicts

## Frozen partition (input to Phase 7)
- File: `results/step8_confirmatory_partition.parquet`  (patient_id, cluster_AB, core, consensus_index)
- n = 335 Luminal-A patients | Cluster A = 219, Cluster B = 116 (high CNV burden)
- Core (consensus >= 0.80): 334/335 — 1 ambiguous patient

## K-registration
- K=2 pre-registered AND stability-optimal (min PAC). No amendment; confirmatory K locked before modeling.

## Confound verdict (Step 4)
- PROCEED — clean molecular split. Dominant driver: CNV_burden (biology), effect=0.50.
- No technical confound dominates (TSS permutation-null, Holm p=1.0; platform constant HM450-only).
- Stage/N/M are burden-correlated mediators, not confounds. Age null.
- Untested (documented limitation): grade, ER/PR/HER2, tumor purity, mutation burden, RNA library size, plate.

## Modality-defining verdict (Step 5)
- CNV defines the split (AUC=1.00). RNA invisible (AUC=0.57, n.s.) — the split is
  expression-invisible, hence invisible to PAM50. Methylation faint but significant (AUC=0.60).
- Model-native gates: RNA suppressed in both clusters; CNV<->methylation routing shift localized to driver
  amplicons (CCND1, FGF3/19, MDM2, AURKA, PPM1D). Directional mechanism = hypothesis pending rho_pred.

## Ablation verdict (Step 6) — CEILING RESULT
- Every representation that can access burden (U2/U1/U0b/RAW/MOFA+-raw) recovers the SAME partition
  (ARI 0.92-1.00) at the SAME reproducibility (PAC ~0.012). U0 (RNA-only) splits elsewhere (ARI ~0).
- The graph neither reorganizes nor stabilizes the partition. Its value is interpretability, not discovery.
- The dominant axis of LumA heterogeneity is aneuploidy burden: low-rank and linearly accessible.

## Biology (Step 7, exploratory, outcome-blind)
- CNV genome-wide gained in B (all 433 genes FDR<0.05). RNA/methylation weak (<0.3 sigma).
- See step7 for known-axis (proliferation / PI3K-TP53 / luminal) shifts.

## What Phase 4 does NOT claim
- Not a superior subgroup discovery (a linear burden score recovers it). Not yet a validated mechanism
  (rho_pred owed). Not a prognostic claim (Phase 7 confirmatory PFI Cox, with age/stage adjustment).
