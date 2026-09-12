# Phase 5 — Figure captions
 
## Step 1
 
**Figure A — `step1_km_pfi_by_cluster.png`.**
Kaplan–Meier estimates of progression-free interval (PFI) for the outcome-blind K=2
Luminal-A burden split (Cluster A = low CNV/aneuploidy burden, n=218, 22 events;
Cluster B = high burden, n=116, 13 events; one patient with missing PFI time
excluded, n=334). Shaded bands are 95% confidence intervals; vertical ticks mark
censoring. The unadjusted hazard ratio (B vs A) is 1.28 (95% CI 0.64–2.55;
log-rank p=0.49); median follow-up 32 months
(reverse Kaplan–Meier). Numbers at risk are tabulated below the axis. The high-burden arm
trends toward earlier progression, but the interval includes 1: at 35 events the
honest deliverable is the effect size and its interval, not statistical significance.
Fixed colors A=#0072B2, B=#D55E00. 300 dpi.
 
**Figure B — `step1_adjustment_forest.png`.**
Forest plot of the burden-split hazard ratio (Cluster B vs A) for PFI across the
pre-registered adjustment ladder: unadjusted (total effect), +age (confounder; age was
null in Phase 4), and +age+stage run in parallel as ordinal (I–IV) and binary (early/late)
codings. The two stage-adjusted rungs (shaded band) estimate the direct, non-stage-mediated
effect and are not "the" hazard ratio; stage is treated as a mediator, so their attenuation
toward the null is expected over-adjustment, shown rather than hidden. An unadjusted
core-only sensitivity (dropping the single non-core patient TCGA-PE-A5DE) is shown below
the divider and is stable. Markers are HRs, whiskers 95% CIs, dashed line the null. 300 dpi.

## Step 2

**Figure — `step2_endpoint_forest.png`.**
Forest of the burden-split hazard ratio (Cluster B vs A) across three endpoints — progression-free interval (PFI), overall survival (OS), and disease-specific survival (DSS) — each shown unadjusted (filled) and age-adjusted (open). All point estimates exceed 1 (PFI 1.28, OS 1.58, DSS 1.33 unadjusted), so the direction is concordant across endpoints. OS carries the strongest signal; DSS is widest (18 events) as expected. Because DSS counts only cancer deaths, its agreement with PFI cross-checks the non-cancer-death censoring in the PFI endpoint. Endpoint colors are fixed (Okabe–Ito, distinct from the A/B cluster palette); markers are HRs, whiskers 95% CIs, dashed line the null. 300 dpi.

## Step 3

**Figure A — `step3_armB_km.png`.** Kaplan–Meier PFI for the Arm-B out-of-fold risk split (high vs low, median-of-OOF cut). The survival-supervised score separates progression (OOF Harrell C = 0.45, 95% CI 0.31–0.58); it was trained to, so the decisive question is which patients it selects (Step 4). Encoder frozen, head cross-validated, risk strictly out-of-fold. 300 dpi.

**Figure B — `step3_risk_vs_burden.png`.** Arm-B OOF risk against CNV/aneuploidy burden (PC1), points colored by the outcome-blind Arm-A cluster. Pearson r = 0.37 (p<0.001). If risk tracks burden and the clusters separate along the same axis, the survival signal is the aneuploidy axis Arm A found blind — the mechanistic crux of convergence. 300 dpi.

## Step 4

**Figure A — `step4_agreement_matrix.png`.** 2x2 agreement between the Arm-A outcome-blind cluster (A/B) and the Arm-B survival-trained OOF median-risk split (low/high). Agreement is weak (ARI = 0.10, Cohen's κ = 0.32); concordant cells (A–low, B–high) are outlined. High-risk lands on high-burden cluster B. 300 dpi.

**Figure B — `step4_km_overlay.png`.** Kaplan–Meier overlay of both splits on one axis (solid = Arm A clusters, dashed = Arm B risk groups). The curves are not superimposable, to be read together with the Arm-B OOF C-index from Step 3: convergence here means both methods gravitate to the CNV-burden axis, not that either alone proves prognosis. 300 dpi.
