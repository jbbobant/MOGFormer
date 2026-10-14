# Transformer variant configurations

Three transformer variants, re-run at the full 25-estimate protocol on the
shared partition. Each has a five-class and a four-class
twin; the four-class one differs only by `exclude_classes`, and reuses the
five-class partition rather than drawing its own, so the two stay paired.

## Why these three

From `results/ablation_index.md`. Only four archived runs ever reached 25
estimates; the top three of those are re-run here. The two highest *means* in
the whole archive — `E300p40_inside_lgemb` at 0.878 and
`E300p100_inside_ptemb32_meth_M` at 0.874 — rest on ten and five estimates
respectively, which is one partition rather than a distribution over
partitions. They are not selected on that basis, and `pretrained_embedding`
is the same architecture as the second of them, taken the whole way.

| config | archived run | archived mean (n=25) | distinguishing setting |
| --- | --- | --- | --- |
| `pretrained_embedding` | `E300p100_inside_ptemb32_gamma2` | 0.8347 | frozen ARCHS4 gene vectors, adapter rank 32 |
| `learned_embedding_mask` | `E300p40_inside_meth_M_lgemb_mask` | 0.8283 | learned gene identity, mask-token dropout fill, M-values |
| `gated_fusion` | `E300p40_inside_met_M_gated` | 0.8236 | gated intra-gene fusion instead of attention, M-values |

Two further configurations are ablations of these, not re-runs:

| config | differs from | setting changed | question |
| --- | --- | --- | --- |
| `gated_fusion` | `learned_embedding_mask` | `fusion_type: gated` | does a competing modality gate match attention? |
| `attention_no_gene_identity` | `learned_embedding_mask` | no gene-identity embedding, zero dropout fill | can the graph alone tell genes apart? |
| `pretrained_embedding_rna_only` | `pretrained_embedding` | expression only | what do copy number and methylation add? |

`gated_fusion` cannot reproduce its archived run exactly: the archive records
no gene-identity setting for it. It is defined as a single-variable change from
`learned_embedding_mask` instead, so its comparison is clean.

Without a gene-identity embedding every gene is tokenised by the same function
of its values, and genes differ only in their graph positional encoding and
shortest-path profile. On fold 0, 50 of 467 selected genes are isolated in the
STRING subgraph with identical encodings, and 320 have another gene within 5%
of the encoding's spread. The `attention_no_gene_identity` ablation measures
what that costs.

## These numbers are not comparable to the archived ones

Three properties of the pipeline make a re-run mandatory rather than optional.

**Every model setting must reach the model.** A harness that builds the
classifier from a hand-written argument list can drop a setting silently; one
run configured as `fusion_type: gated` trained attention fusion that way. A
test now requires every field of the model configuration to arrive as an
estimator parameter with its configured value.

**The summary token was in the wrong bias bucket.** The self-supervised branch
padded the shortest-path matrix's summary row and column with
`max_distance + 2`, giving the token its own bucket in a table sized
`max_distance + 3`. The supervised branch padded with `0`, which is the bucket
every gene's own diagonal occupies — so the summary token was biased as though
it were each gene attending to itself. It never raised an error, so it affected
every supervised run silently. The unified layer takes the self-supervised
treatment.

**The cohort composition is now stated, not defaulted.** The baseline harness
hardcoded a `BRCA_Normal` filter inside its loader, so baselines were scored on
914 patients and the transformer on 949. Exclusion is now a configuration field
and the partition is projected rather than redrawn.

## Running them

```bash
uv run python -m mogformer.cli.experiments transformer_variants --device cuda \
  --variants configs/transformer_variants/pretrained_embedding_5class.yaml \
             configs/transformer_variants/learned_embedding_mask_5class.yaml \
             configs/transformer_variants/gated_fusion_5class.yaml
```

Swap `_5class` for `_4class` for the four-class twins. Results land under
`results/experiments/transformer_variants/`, beside the baselines scored on the
same partition.

The driver picks `cohort_m.npz` or `cohort_beta.npz` from each config's
`methy_file`, so the M-value variants read the M-value cohort without anything
being passed by hand.
