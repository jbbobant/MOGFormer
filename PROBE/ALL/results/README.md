# rho_pred probe — provenance
sign convention : injected methylation UP  ->  predicted RNA DOWN  (rho_pred < 0 = silencing)
head            : Head L (local, primary); Head G saved as robustness (response_summary.mean_headG)
grid            : [np.float64(-2.0), np.float64(-1.5), np.float64(-1.0), np.float64(-0.5), np.float64(0.0), np.float64(0.5), np.float64(1.0), np.float64(1.5), np.float64(2.0)]  (z-units of the per-gene-standardized methylation channel)
cohort          : all  ; genes run : 433 (genome-wide)
mask            : deterministic (rna, i) only  (NOT the stochastic training masker)
seed            : 20240712 ; bootstrap : 1000
sign gate (§4)  : PASS  median -0.254
direction (§3.1): median rho_pred -0.206  Wilcoxon p 1.11e-72
specificity(§3.2): rho_pred MW p 0.493 ; Delta MW p 0.455
nulls (§3.4)    : ['permute_readout_gene', 'permute_graph', 'shuffle_methy_patients'] (screen set)
