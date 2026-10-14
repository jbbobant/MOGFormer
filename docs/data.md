# Data

The cohorts are not distributed with the repository. This page lists every file
the pipeline reads, where it comes from and where it is expected. All paths are
relative to the repository root; `data/` is not tracked by git.

The pipeline reads **cleaned matrices**, aligns them into **cohort bundles**
(`scripts/build_cohort_bundles.py`), and from then on reads only the bundles.
Everything fitted to patients — gene selection, imputation, scaling, batch
correction, gene programs — happens later, inside each run, on its fitting
patients only.

## Cleaned inputs

### TCGA-BRCA

Source: TCGA-BRCA via cBioPortal (expression, copy number, methylation,
clinical) and the GDC portal (receptor status follow-up).

| Expected path | Contents |
| --- | --- |
| `data/clean/data_rna_common.csv` | RNA-seq (RSEM), genes × patients, on the common gene and patient axis |
| `data/clean/data_cnv_common.csv` | Gene-level absolute copy number, same axis |
| `data/clean/gene_promoter_methylation_common_meta.csv` | Gene-promoter methylation (beta values), same axis |
| `data/clean/data_clinical.csv` | Clinical table: PAM50 subtype, stage, age, follow-up |
| `data/clean/survival_features_common.csv` | Survival endpoints of the supervised cohort |
| `data/download/clinical.project-tcga-brca.2026-10-03/follow_up.tsv` | GDC clinical follow-up: ER, PR and HER2 tests |
| `data/download/data_clinical_sample.txt` | cBioPortal sample table, fills receptor gaps |
| `data/raw/data_rna_seq_v2_rsem.csv`, `data/raw/data_cnv.csv`, `data/raw/data_methylation450.csv`, `data/raw/data_methylation_M.csv` | Raw matrices of the supervised cohort |

### METABRIC

Source: METABRIC via cBioPortal.

| Expected path | Contents |
| --- | --- |
| `data/METABRIC/clean/data_rna_common.csv` | Microarray expression (per-gene z-scores) |
| `data/METABRIC/clean/data_cnv_common.csv` | Ordinal copy-number calls (0–4) |
| `data/METABRIC/clean/gene_promoter_methylation_common_beta.csv`, `..._M.csv`, `..._meta.csv` | Gene-promoter methylation |
| `data/METABRIC/clean/data_clinical.csv` | Clinical table: PAM50 (Claudin), integrative cluster, ER, HER2, NPI, therapies, processing batch (`COHORT`), relapse-free and overall survival |

### Networks and reference lists

| Expected path | Source |
| --- | --- |
| `data/clean/9606.protein.links.v12.0.txt`, `data/clean/9606.protein.aliases.v12.0.txt` | STRING v12.0, human (string-db.org) |
| `data/clean/9606.protein.links.detailed.v12.5.txt` | STRING v12.5 detailed links (experimental and database channels) |
| `data/clean/collectri_signed_edges.tsv` | CollecTRI signed TF → target edges, from OmniPath via decoupleR (`decoupler.op.collectri(organism="human")`), written as a TSV with columns `source`, `target`, `sign` (±1), `n_sources` |
| `data/raw/PAM50.txt` | The 50 PAM50 genes |
| `data/raw/CGenes.txt` | Curated breast-cancer genes, force-included in every gene axis |
| `data/raw/FROGS-ARCHS4/FROGS-ARCHS4_emb.csv`, `..._symbollist.txt` | Pretrained gene vectors (FRoGS, ARCHS4), used by one supervised variant |
| `data/clean/folds.json` | The shared 5 × 5 cross-validation partition of the supervised cohort (`uv run mogformer folds --config configs/tcga.yaml`) |

## Bundles

`uv run python -m scripts.build_cohort_bundles` writes:

| Bundle | Used by |
| --- | --- |
| `data/bundle/tcga/` | The supervised experiments; with `cohort_ssl.npz`, `ploidy.csv` and the evidence-channel STRING edges, the TCGA self-supervised cohort (748 patients) |
| `data/bundle/metabric/<target>/` | The METABRIC supervised targets, one partition per target |
| `data/bundle/metabric/ssl/` | The METABRIC self-supervised cohort (1,416 patients) |

## Known gap

The step from the cBioPortal and GDC downloads to the cleaned matrices above
(gene alignment, promoter aggregation of methylation probes, the common
patient axis) was done with notebooks that are not part of this repository.
It is not yet part of the package, so the cleaned matrices cannot be
regenerated from the downloads by a single command. Everything downstream of them can.
