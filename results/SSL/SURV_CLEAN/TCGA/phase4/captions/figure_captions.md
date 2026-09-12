## Step 1
**step1_stability_curves** — PAC, Monti Δ-area, and consensus silhouette over K∈[2, 3, 4, 5] on the
dimension-standardized 128-d U2 embedding (n=335, B=1000 subsamples at 80%). Primary
clusterer: kmeans. Stability-optimal K=2 (min PAC); K=2 pre-registered, marked distinctly.
Clustering performed in 128-d; no survival used.
**step1_consensus_cdf** — empirical CDF of off-diagonal consensus values per K; cleaner separations
hug 0 and 1 with little mid-range mass (basis of PAC).

## Step 1
**step1_stability_curves** — PAC, Monti Δ-area, and consensus silhouette over K∈[2, 3, 4, 5] on the
dimension-standardized 128-d U2 embedding (n=335, B=1000 subsamples at 80%). Primary
clusterer: kmeans. Stability-optimal K=2 (min PAC); K=2 pre-registered, marked distinctly.
Clustering performed in 128-d; no survival used.
**step1_consensus_cdf** — empirical CDF of off-diagonal consensus values per K; cleaner separations
hug 0 and 1 with little mid-range mass (basis of PAC).

## Step 2
**step2_consensus_heatmap_K2** — K=2 consensus matrix (mako), patients ordered by the average-linkage
tree, cluster color strip A=#0072B2/B=#D55E00. Two near-1 diagonal blocks with near-0 off-diagonal =
a clean, reproducible split (PAC=0.012, n=335).
**step2_embedding_projection** — PCA and UMAP of the 128-d embedding, colored by cluster with
2σ ellipses. Clustering was performed in 128-d; projections are display-only.
**step2_baseclusterer_agreement** — k-means vs GMM confusion at K=2 (ARI=1.000); the two
clusterers recover the same partition. Cluster sizes A=219, B=116.
**step2_standardization_robustness** — z-scored vs raw-embedding partitions (ARI=0.988),
showing the dimension-standardization choice does not change which patients group.
**step2_confidence_landscape** — per-patient consensus index over PCA (ambiguous patients ringed);
PC1 KDE by cluster shows the split is near-1-D along PC1, the candidate CNV-burden axis (to be
confirmed against a CNV score, not asserted here).

## Step 2
**step2_consensus_heatmap_K2** — K=2 consensus matrix (mako), patients ordered by the average-linkage
tree, cluster color strip A=#0072B2/B=#D55E00. Two near-1 diagonal blocks with near-0 off-diagonal =
a clean, reproducible split (PAC=0.012, n=335).
**step2_embedding_projection** — PCA and UMAP of the 128-d embedding, colored by cluster with
2σ ellipses. Clustering was performed in 128-d; projections are display-only.
**step2_baseclusterer_agreement** — k-means vs GMM confusion at K=2 (ARI=1.000); the two
clusterers recover the same partition. Cluster sizes A=219, B=116.
**step2_standardization_robustness** — z-scored vs raw-embedding partitions (ARI=0.988),
showing the dimension-standardization choice does not change which patients group.
**step2_confidence_landscape** — per-patient consensus index over PCA (ambiguous patients ringed);
PC1 KDE by cluster shows the split is near-1-D along PC1, the candidate CNV-burden axis (to be
confirmed against a CNV score, not asserted here).

## Step 3
**step3_silhouette_by_cluster** — per-patient embedding-space silhouette sorted within cluster (left)
and the per-patient consensus-index distribution (right; core >= 0.8). 334/335
patients are core; the confirmatory test rests on a near-complete core cohort.
**step3_core_composition** — core vs ambiguous counts per cluster; the split has almost no fragile margin.
**step3_cnv_pc1_confirmation** — PC1 confirmed as a per-patient CNV-burden axis, restricted to the
433 gene-universe genes present in the CNV matrix (the burden the model actually ingested).
(top-left) PC1 vs universe CNV load: Pearson r=0.08, Spearman rho=0.10; (top-right) load by
cluster (Mann-Whitney p=4.6e-01, AUC=0.51, higher-burden cluster=A); (bottom-left) proxy
fidelity, universe vs genome-wide load (Spearman rho=0.58) — licenses the genome-wide reading only
if high; (bottom-right) PCA landscape recolored by universe load. Encoding auto-detected: continuous log2 (|x|>0.3).

## Step 3
**step3_silhouette_by_cluster** — per-patient embedding-space silhouette sorted within cluster (left)
and the per-patient consensus-index distribution (right; core >= 0.8). 334/335
patients are core; the confirmatory test rests on a near-complete core cohort.
**step3_core_composition** — core vs ambiguous counts per cluster; the split has almost no fragile margin.
**step3_cnv_pc1_confirmation** — PC1 confirmed as a per-patient CNV-burden axis, restricted to the
433 gene-universe genes present in the CNV matrix (the burden the model actually ingested).
(top-left) PC1 vs universe CNV load: Pearson r=0.93, Spearman rho=0.82; (top-right) load by
cluster (Mann-Whitney p=3.2e-51, AUC=1.00, higher-burden cluster=B); (bottom-left) proxy
fidelity, universe vs genome-wide load (Spearman rho=0.97) — licenses the genome-wide reading only
if high; (bottom-right) PCA landscape recolored by universe load. Encoding auto-detected: absolute copy number (altered = CN<=1 or CN>=3, diploid=2).

## Step 4
**step4_confound_barplot** — association of each covariate with the K=2 split: Cramér's V (categorical:
stage/N/M/TSS) and |AUC−0.5| (continuous: CNV burden, age), permutation p (B=5000), Holm-corrected;
* = Holm-significant. Bars colored by role — biology (CNV burden, green), clinical (blue), technical
(TSS, red). Methylation platform asserted constant (HM450-only) and recorded as a passed guard.
Verdict: PROCEED — clean molecular split; dominant driver CNV_burden (biology); no confound dominates.
Documented limitation: grade, ER/PR/HER2, tumor purity, total mutation burden, RNA library size, and
sequencing plate were unavailable and thus untested; CNV burden (which defines the split) partially
proxies purity, so purity acting alone cannot explain the partition. Stage/N are correlated with CNV
burden as downstream markers of aneuploidy-driven aggressiveness (mediators), not independent
technical confounds.

## Step 4
**step4_confound_barplot** — association of each covariate with the K=2 split: Cramér's V (categorical:
stage/N/M/TSS) and |AUC−0.5| (continuous: CNV burden, age), permutation p (B=5000), Holm-corrected;
* = Holm-significant. Bars colored by role — biology (CNV burden, green), clinical (blue), technical
(TSS, red). Methylation platform asserted constant (HM450-only) and recorded as a passed guard.
Verdict: PROCEED — clean molecular split; dominant driver CNV_burden (biology); no confound dominates.
Documented limitation: grade, ER/PR/HER2, tumor purity, total mutation burden, RNA library size, and
sequencing plate were unavailable and thus untested; CNV burden (which defines the split) partially
proxies purity, so purity acting alone cannot explain the partition. Stage/N are correlated with CNV
burden as downstream markers of aneuploidy-driven aggressiveness (mediators), not independent
technical confounds.

## Step 4
**step4_confound_barplot** — association of each covariate with the K=2 split: Cramér's V (categorical:
stage/N/M/TSS) and |AUC−0.5| (continuous: CNV burden, age), permutation p (B=5000), Holm-corrected;
* = Holm-significant. Bars colored by role — biology (CNV burden, green), clinical (blue), technical
(TSS, red). Methylation platform asserted constant (HM450-only) and recorded as a passed guard.
Verdict: PROCEED — clean molecular split; dominant driver CNV_burden (biology); no confound dominates.
Documented limitation: grade, ER/PR/HER2, tumor purity, total mutation burden, RNA library size, and
sequencing plate were unavailable and thus untested; CNV burden (which defines the split) partially
proxies purity, so purity acting alone cannot explain the partition. Stage/N are correlated with CNV
burden as downstream markers of aneuploidy-driven aggressiveness (mediators), not independent
technical confounds.

## Step 4
**step4_confound_barplot** — association of each covariate with the K=2 split: Cramér's V (categorical:
stage/N/M/TSS) and |AUC−0.5| (continuous: CNV burden, age), permutation p (B=5000), Holm-corrected;
* = Holm-significant. Bars colored by role — biology (CNV burden, green), clinical (blue), technical
(TSS, red). Methylation platform asserted constant (HM450-only) and recorded as a passed guard.
Verdict: PROCEED — clean molecular split; dominant driver CNV_burden (biology); no confound dominates.
Documented limitation: grade, ER/PR/HER2, tumor purity, total mutation burden, RNA library size, and
sequencing plate were unavailable and thus untested; CNV burden (which defines the split) partially
proxies purity, so purity acting alone cannot explain the partition. Stage/N are correlated with CNV
burden as downstream markers of aneuploidy-driven aggressiveness (mediators), not independent
technical confounds.

## Step 4
**step4_confound_barplot** — association of each covariate with the K=2 split: Cramér's V (categorical:
stage/N/M/TSS) and |AUC−0.5| (continuous: CNV burden, age), permutation p (B=5000), Holm-corrected;
* = Holm-significant. Bars colored by role — biology (CNV burden, green), clinical (blue), technical
(TSS, red). Methylation platform asserted constant (HM450-only) and recorded as a passed guard.
Verdict: PROCEED — clean molecular split; dominant driver CNV_burden (biology); no confound dominates.
Documented limitation: grade, ER/PR/HER2, tumor purity, total mutation burden, RNA library size, and
sequencing plate were unavailable and thus untested; CNV burden (which defines the split) partially
proxies purity, so purity acting alone cannot explain the partition. Stage/N are correlated with CNV
burden as downstream markers of aneuploidy-driven aggressiveness (mediators), not independent
technical confounds.

## Step 5a (classifier half; gate readout deferred to 5b)
**step5_modality_attribution_classifiers** — cross-validated (5-fold, per-gene z fit on train only)
logistic AUC predicting Cluster A/B from each modality alone and combined, over 335 LumA
patients. Winner: CNV. Grey band = permutation-null ceiling; * = perm p<0.05. CNV dominance is
expected by construction (the split is a CNV-burden axis, Step 3); the informative reads are whether
methylation and RNA independently exceed chance. Cross-reference Step 4: methylation platform is
constant (HM450-only) and TSS is null, so any methylation signal here is biology, not the platform
artefact that dominated a prior split.

## Step 5b (model-native gates)
**step5_modality_attribution** — (left) model-agnostic CV-AUC per modality; (right) per-cluster mean
GatedFusion gate weights (softmax over rna/cnv/methy, order fixed by the frozen encoder). Gate profile
A vs B: rna 0.15/0.17, cnv 0.54/0.42,
methy 0.31/0.41. Inputs bit-exact (dumped).
**step5_gene_gate_differential** — top 30 genes by per-cluster gate shift (A−B); which genes' modality
routing differs between clusters.

## Step 6 (ablation ladder)
**step6_pac_by_variant** — PAC at K=2 (identical consensus protocol, B=1000, 80%) for each
representation on 335 common LumA patients; lower = more reproducible. Rungs: U0 (RNA-only),
U0b (trimodal graph-off), U1 (graph-on GRN-off), U2 (full), MOFA+ (linear), RAW (concat features).
U2 PAC=0.012 (reproduces Step 1). Whether the graph/multimodality lowers PAC below the floor is the
downstream counterpart of the Phase-3 reconstruction ladder.
**step6_variant_partition_agreement** — pairwise partition ARI. High ARI across rungs = the same patients
group regardless of representation (the burden axis is architecture-independent); the graph then earns
its complexity on interpretability (Step 5), not on reorganizing the partition. Low ARI to U0b would mean
the graph changes *which* patients group.

## Step 6 (ablation ladder)
**step6_pac_by_variant** — PAC at K=2 (identical consensus protocol, B=1000, 80%) for each
representation on 335 common LumA patients; lower = more reproducible. Rungs: U0 (RNA-only),
U0b (trimodal graph-off), U1 (graph-on GRN-off), U2 (full), MOFA+ (linear), RAW (concat features).
U2 PAC=0.012 (reproduces Step 1). Whether the graph/multimodality lowers PAC below the floor is the
downstream counterpart of the Phase-3 reconstruction ladder.
**step6_variant_partition_agreement** — pairwise partition ARI. High ARI across rungs = the same patients
group regardless of representation (the burden axis is architecture-independent); the graph then earns
its complexity on interpretability (Step 5), not on reorganizing the partition. Low ARI to U0b would mean
the graph changes *which* patients group.

## Step 6 (ablation ladder)
**step6_pac_by_variant** — PAC at K=2 (identical consensus protocol, B=1000, 80%) for each
representation on 335 common LumA patients; lower = more reproducible. Rungs: U0 (RNA-only),
U0b (trimodal graph-off), U1 (graph-on GRN-off), U2 (full), MOFA+ (linear), RAW (concat features).
U2 PAC=0.012 (reproduces Step 1). Whether the graph/multimodality lowers PAC below the floor is the
downstream counterpart of the Phase-3 reconstruction ladder.
**step6_variant_partition_agreement** — pairwise partition ARI. High ARI across rungs = the same patients
group regardless of representation (the burden axis is architecture-independent); the graph then earns
its complexity on interpretability (Step 5), not on reorganizing the partition. Low ARI to U0b would mean
the graph changes *which* patients group.

## Step 6 (ablation ladder)
**step6_pac_by_variant** — PAC at K=2 (identical consensus protocol, B=1000, 80%) for each
representation on 335 common LumA patients; lower = more reproducible. Rungs: U0 (RNA-only),
U0b (trimodal graph-off), U1 (graph-on GRN-off), U2 (full), MOFA+ (linear), RAW (concat features).
U2 PAC=0.012 (reproduces Step 1). Whether the graph/multimodality lowers PAC below the floor is the
downstream counterpart of the Phase-3 reconstruction ladder.
**step6_variant_partition_agreement** — pairwise partition ARI. High ARI across rungs = the same patients
group regardless of representation (the burden axis is architecture-independent); the graph then earns
its complexity on interpretability (Step 5), not on reorganizing the partition. Low ARI to U0b would mean
the graph changes *which* patients group.

## Step 7 (exploratory molecular characterization; outcome-blind)
**step7_volcano_by_modality** — standardized mean difference (B−A) vs −log10 p per modality, BH-FDR.
CNV dominates by construction (the split is burden); RNA/methylation leaders show which expression and
epigenetic programs co-vary with the burden axis without defining it.
**step7_top_features_heatmap** — top 25 features (patients grouped A|B); row labels colored by
modality. Qualitative evidence the clusters differ on interpretable biology, to be quantified
prognostically in Phase 7 and mechanistically in Phase 8. No survival used.

## Step 8 (freeze & report)
**step8_phase4_summary** — composite: consensus heatmap (stable), confound barplot (clean), PAC-by-variant
(ceiling). Confirmatory partition frozen to step8_confirmatory_partition.parquet (A=219, B=116,
core=334/335) as the locked input to Phase 7.

