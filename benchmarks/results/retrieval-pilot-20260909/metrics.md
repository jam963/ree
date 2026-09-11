# Complete pilot metric tables

Generated from the completed `summary.json`; raw scores are scaled by 100 below.
Query provider / index provider labels are literal. Intervals are paired 95%
percentile bootstrap intervals, 10,000 replicates per corpus, unadjusted for
multiple comparisons. **Exploratory two-corpus results, not a shipping winner.**

Summary SHA-256: `bc0d407d81768fca8ba6b7618a739c2c4fdfc0269288688fe5ef4530dfe4420c`

A **flag** means point loss >1.00 nDCG point against the better homogeneous
reference. `below cutoff` is not proof of equivalence; inspect the interval.
Negative signed losses are improvements. All point estimates/intervals are
rounded only for display; decisions use unrounded values in the JSON.

## nfcorpus — 3,633 documents, 100 queries

### All six quality rows per recipe

| Recipe | Query / index | nDCG@10 | 95% interval | Recall@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| arctic (768) | cpu/cpu | 35.76 | 29.15..42.77 | 17.24 | 53.50 |
| arctic (768) | cuda/cuda | 36.71 | 29.98..43.83 | 17.50 | 53.76 |
| arctic (768) | cpu/cuda | 36.61 | 29.92..43.68 | 17.49 | 53.90 |
| arctic (768) | cuda/cpu | 35.95 | 29.21..42.99 | 17.22 | 53.48 |
| arctic (768) | cpu/mixed | 36.23 | 29.60..43.29 | 17.41 | 54.00 |
| arctic (768) | cuda/mixed | 36.16 | 29.42..43.33 | 17.40 | 53.23 |
| granite (384) | cpu/cpu | 30.21 | 24.20..36.65 | 13.46 | 51.80 |
| granite (384) | cuda/cuda | 30.21 | 24.20..36.65 | 13.46 | 51.80 |
| granite (384) | cpu/cuda | 30.21 | 24.20..36.65 | 13.46 | 51.80 |
| granite (384) | cuda/cpu | 30.21 | 24.20..36.65 | 13.46 | 51.80 |
| granite (384) | cpu/mixed | 30.21 | 24.20..36.65 | 13.46 | 51.80 |
| granite (384) | cuda/mixed | 30.21 | 24.20..36.65 | 13.46 | 51.80 |
| e5 (384) | cpu/cpu | 31.98 | 25.55..38.89 | 15.54 | 49.99 |
| e5 (384) | cuda/cuda | 32.16 | 25.63..38.99 | 16.01 | 49.45 |
| e5 (384) | cpu/cuda | 31.71 | 25.29..38.58 | 15.83 | 49.11 |
| e5 (384) | cuda/cpu | 32.33 | 25.83..39.21 | 15.69 | 50.53 |
| e5 (384) | cpu/mixed | 31.45 | 25.01..38.35 | 15.31 | 50.79 |
| e5 (384) | cuda/mixed | 31.81 | 25.25..38.74 | 15.62 | 50.59 |
| jina-code (768) | cpu/cpu | 10.09 | 7.04..13.49 | 4.70 | 21.52 |
| jina-code (768) | cuda/cuda | 9.16 | 6.20..12.33 | 4.18 | 21.66 |
| jina-code (768) | cpu/cuda | 9.57 | 6.52..12.89 | 4.39 | 21.71 |
| jina-code (768) | cuda/cpu | 9.70 | 6.67..13.09 | 4.60 | 20.87 |
| jina-code (768) | cpu/mixed | 9.78 | 6.75..13.05 | 4.55 | 21.99 |
| jina-code (768) | cuda/mixed | 9.35 | 6.41..12.57 | 4.46 | 21.62 |
| bge-m3 (1024) | cpu/cpu | 33.83 | 27.30..40.65 | 16.12 | 52.75 |
| bge-m3 (1024) | cuda/cuda | 33.75 | 27.20..40.60 | 16.12 | 52.25 |
| bge-m3 (1024) | cpu/cuda | 33.83 | 27.30..40.65 | 16.12 | 52.75 |
| bge-m3 (1024) | cuda/cpu | 33.83 | 27.30..40.65 | 16.12 | 52.75 |
| bge-m3 (1024) | cpu/mixed | 33.75 | 27.21..40.61 | 16.12 | 52.25 |
| bge-m3 (1024) | cuda/mixed | 33.75 | 27.21..40.61 | 16.12 | 52.25 |
| arctic-256 (256) | cpu/cpu | 34.80 | 28.04..41.91 | 16.73 | 51.31 |
| arctic-256 (256) | cuda/cuda | 36.09 | 29.32..43.28 | 17.49 | 52.38 |
| arctic-256 (256) | cpu/cuda | 35.46 | 28.66..42.63 | 17.41 | 51.07 |
| arctic-256 (256) | cuda/cpu | 35.23 | 28.51..42.30 | 17.04 | 51.73 |
| arctic-256 (256) | cpu/mixed | 35.37 | 28.58..42.58 | 17.26 | 52.26 |
| arctic-256 (256) | cuda/mixed | 35.73 | 28.97..42.85 | 17.19 | 53.31 |

### Every cross-provider and mixed-index check

| Recipe | Query / index | nDCG loss | 95% loss interval | Point check |
| --- | --- | ---: | ---: | --- |
| arctic | cpu/cuda | 0.11 | -0.49..0.68 | below cutoff |
| arctic | cuda/cpu | 0.76 | -0.11..1.63 | below cutoff |
| arctic | cpu/mixed | 0.49 | -0.17..1.18 | below cutoff |
| arctic | cuda/mixed | 0.55 | -0.06..1.19 | below cutoff |
| granite | cpu/cuda | 0.00 | 0.00..0.00 | below cutoff |
| granite | cuda/cpu | 0.00 | 0.00..0.00 | below cutoff |
| granite | cpu/mixed | 0.00 | 0.00..0.00 | below cutoff |
| granite | cuda/mixed | 0.00 | 0.00..0.00 | below cutoff |
| e5 | cpu/cuda | 0.45 | -0.04..1.38 | below cutoff |
| e5 | cuda/cpu | -0.16 | -0.75..0.92 | below cutoff |
| e5 | cpu/mixed | 0.72 | 0.15..1.66 | below cutoff |
| e5 | cuda/mixed | 0.36 | -0.26..1.32 | below cutoff |
| jina-code | cpu/cuda | 0.53 | -0.22..1.33 | below cutoff |
| jina-code | cuda/cpu | 0.40 | -0.28..1.08 | below cutoff |
| jina-code | cpu/mixed | 0.32 | -0.35..1.06 | below cutoff |
| jina-code | cuda/mixed | 0.74 | 0.03..1.56 | below cutoff |
| bge-m3 | cpu/cuda | 0.00 | 0.00..0.00 | below cutoff |
| bge-m3 | cuda/cpu | 0.00 | 0.00..0.00 | below cutoff |
| bge-m3 | cpu/mixed | 0.08 | -0.01..0.24 | below cutoff |
| bge-m3 | cuda/mixed | 0.08 | -0.01..0.24 | below cutoff |
| arctic-256 | cpu/cuda | 0.63 | 0.06..1.24 | below cutoff |
| arctic-256 | cuda/cpu | 0.86 | -0.26..2.04 | below cutoff |
| arctic-256 | cpu/mixed | 0.72 | -0.16..1.63 | below cutoff |
| arctic-256 | cuda/mixed | 0.36 | -0.42..1.17 | below cutoff |

### Paired candidate-minus-Arctic-768 nDCG deltas

Same query/index combination on both sides; negative means below Arctic 768.
The Arctic-256 rows are also the paired dimension-reduction comparisons.

| Recipe | Query / index | nDCG delta | 95% delta interval |
| --- | --- | ---: | ---: |
| arctic | cpu/cpu | 0.00 | 0.00..0.00 |
| arctic | cuda/cuda | 0.00 | 0.00..0.00 |
| arctic | cpu/cuda | 0.00 | 0.00..0.00 |
| arctic | cuda/cpu | 0.00 | 0.00..0.00 |
| arctic | cpu/mixed | 0.00 | 0.00..0.00 |
| arctic | cuda/mixed | 0.00 | 0.00..0.00 |
| granite | cpu/cpu | -5.56 | -8.84..-2.61 |
| granite | cuda/cuda | -6.51 | -9.83..-3.48 |
| granite | cpu/cuda | -6.40 | -9.80..-3.40 |
| granite | cuda/cpu | -5.75 | -9.07..-2.71 |
| granite | cpu/mixed | -6.02 | -9.34..-3.07 |
| granite | cuda/mixed | -5.96 | -9.27..-2.95 |
| e5 | cpu/cpu | -3.78 | -6.30..-1.45 |
| e5 | cuda/cuda | -4.55 | -7.02..-2.26 |
| e5 | cpu/cuda | -4.89 | -7.42..-2.63 |
| e5 | cuda/cpu | -3.63 | -6.20..-1.21 |
| e5 | cpu/mixed | -4.78 | -7.30..-2.46 |
| e5 | cuda/mixed | -4.36 | -6.96..-1.98 |
| jina-code | cpu/cpu | -25.67 | -31.60..-20.06 |
| jina-code | cuda/cuda | -27.56 | -33.61..-21.92 |
| jina-code | cpu/cuda | -27.04 | -33.05..-21.47 |
| jina-code | cuda/cpu | -26.26 | -32.40..-20.47 |
| jina-code | cpu/mixed | -26.45 | -32.37..-20.94 |
| jina-code | cuda/mixed | -26.81 | -32.85..-21.15 |
| bge-m3 | cpu/cpu | -1.93 | -3.94..-0.01 |
| bge-m3 | cuda/cuda | -2.96 | -5.05..-0.93 |
| bge-m3 | cpu/cuda | -2.78 | -4.89..-0.74 |
| bge-m3 | cuda/cpu | -2.13 | -4.23..-0.12 |
| bge-m3 | cpu/mixed | -2.48 | -4.44..-0.59 |
| bge-m3 | cuda/mixed | -2.41 | -4.40..-0.49 |
| arctic-256 | cpu/cpu | -0.96 | -1.87..-0.07 |
| arctic-256 | cuda/cuda | -0.62 | -1.57..0.28 |
| arctic-256 | cpu/cuda | -1.15 | -2.04..-0.33 |
| arctic-256 | cuda/cpu | -0.73 | -1.55..0.07 |
| arctic-256 | cpu/mixed | -0.86 | -1.52..-0.20 |
| arctic-256 | cuda/mixed | -0.43 | -1.09..0.23 |

### Homogeneous provider deltas and truncation

Delta is CUDA/CUDA minus CPU/CPU. No selected queries were truncated.

| Recipe | nDCG delta | 95% delta interval | Truncated documents | Document fraction |
| --- | ---: | ---: | ---: | ---: |
| arctic | 0.95 | 0.08..1.83 | 515/3633 | 14.18% |
| granite | 0.00 | 0.00..0.00 | 515/3633 | 14.18% |
| e5 | 0.19 | -0.97..1.35 | 524/3633 | 14.42% |
| jina-code | -0.94 | -1.96..-0.01 | 762/3633 | 20.97% |
| bge-m3 | -0.08 | -0.24..0.00 | 515/3633 | 14.18% |
| arctic-256 | 1.29 | 0.16..2.43 | 515/3633 | 14.18% |

## scifact — 5,183 documents, 100 queries

### All six quality rows per recipe

| Recipe | Query / index | nDCG@10 | 95% interval | Recall@10 | MRR@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| arctic (768) | cpu/cpu | 72.00 | 64.41..79.25 | 82.10 | 69.74 |
| arctic (768) | cuda/cuda | 72.86 | 65.30..80.01 | 82.60 | 70.64 |
| arctic (768) | cpu/cuda | 72.43 | 64.89..79.67 | 82.10 | 70.68 |
| arctic (768) | cuda/cpu | 71.68 | 64.23..78.83 | 83.10 | 69.20 |
| arctic (768) | cpu/mixed | 71.88 | 64.27..79.21 | 82.10 | 69.46 |
| arctic (768) | cuda/mixed | 72.89 | 65.37..80.13 | 82.10 | 71.12 |
| granite (384) | cpu/cpu | 64.54 | 56.65..72.16 | 78.40 | 61.70 |
| granite (384) | cuda/cuda | 64.54 | 56.65..72.16 | 78.40 | 61.70 |
| granite (384) | cpu/cuda | 64.54 | 56.65..72.16 | 78.40 | 61.70 |
| granite (384) | cuda/cpu | 64.54 | 56.65..72.16 | 78.40 | 61.70 |
| granite (384) | cpu/mixed | 64.54 | 56.65..72.16 | 78.40 | 61.70 |
| granite (384) | cuda/mixed | 64.54 | 56.65..72.16 | 78.40 | 61.70 |
| e5 (384) | cpu/cpu | 64.97 | 56.61..72.94 | 75.70 | 62.89 |
| e5 (384) | cuda/cuda | 66.81 | 58.93..74.55 | 78.90 | 64.04 |
| e5 (384) | cpu/cuda | 65.27 | 57.17..73.28 | 76.90 | 62.75 |
| e5 (384) | cuda/cpu | 65.05 | 56.98..72.94 | 76.70 | 62.53 |
| e5 (384) | cpu/mixed | 62.11 | 53.71..70.32 | 72.70 | 59.74 |
| e5 (384) | cuda/mixed | 64.43 | 56.13..72.58 | 74.90 | 62.07 |
| jina-code (768) | cpu/cpu | 19.88 | 12.99..27.17 | 24.92 | 19.43 |
| jina-code (768) | cuda/cuda | 20.13 | 13.32..27.30 | 27.17 | 18.65 |
| jina-code (768) | cpu/cuda | 19.82 | 12.94..27.04 | 24.83 | 19.05 |
| jina-code (768) | cuda/cpu | 20.82 | 14.09..27.95 | 28.92 | 19.43 |
| jina-code (768) | cpu/mixed | 19.63 | 12.78..26.81 | 24.58 | 18.92 |
| jina-code (768) | cuda/mixed | 20.13 | 13.33..27.29 | 26.92 | 18.74 |
| bge-m3 (1024) | cpu/cpu | 65.58 | 57.43..73.35 | 76.27 | 63.03 |
| bge-m3 (1024) | cuda/cuda | 65.58 | 57.43..73.35 | 76.27 | 63.03 |
| bge-m3 (1024) | cpu/cuda | 65.58 | 57.43..73.35 | 76.27 | 63.03 |
| bge-m3 (1024) | cuda/cpu | 65.58 | 57.43..73.35 | 76.27 | 63.03 |
| bge-m3 (1024) | cpu/mixed | 65.58 | 57.43..73.35 | 76.27 | 63.03 |
| bge-m3 (1024) | cuda/mixed | 65.58 | 57.43..73.35 | 76.27 | 63.03 |
| arctic-256 (256) | cpu/cpu | 69.96 | 62.42..77.28 | 81.60 | 67.51 |
| arctic-256 (256) | cuda/cuda | 71.48 | 63.86..78.73 | 82.60 | 69.04 |
| arctic-256 (256) | cpu/cuda | 70.09 | 62.43..77.47 | 81.10 | 67.75 |
| arctic-256 (256) | cuda/cpu | 70.06 | 62.52..77.42 | 81.60 | 67.60 |
| arctic-256 (256) | cpu/mixed | 70.29 | 62.73..77.68 | 81.10 | 67.85 |
| arctic-256 (256) | cuda/mixed | 71.29 | 63.71..78.59 | 82.10 | 69.00 |

### Every cross-provider and mixed-index check

| Recipe | Query / index | nDCG loss | 95% loss interval | Point check |
| --- | --- | ---: | ---: | --- |
| arctic | cpu/cuda | 0.43 | -0.53..1.76 | below cutoff |
| arctic | cuda/cpu | 1.19 | -0.46..3.35 | **flag** |
| arctic | cpu/mixed | 0.98 | -0.71..3.05 | below cutoff |
| arctic | cuda/mixed | -0.03 | -1.46..1.70 | below cutoff |
| granite | cpu/cuda | 0.00 | 0.00..0.00 | below cutoff |
| granite | cuda/cpu | 0.00 | 0.00..0.00 | below cutoff |
| granite | cpu/mixed | 0.00 | 0.00..0.00 | below cutoff |
| granite | cuda/mixed | 0.00 | 0.00..0.00 | below cutoff |
| e5 | cpu/cuda | 1.54 | 0.17..3.42 | **flag** |
| e5 | cuda/cpu | 1.76 | -0.37..4.35 | **flag** |
| e5 | cpu/mixed | 4.70 | 2.55..7.36 | **flag** |
| e5 | cuda/mixed | 2.38 | 0.88..4.39 | **flag** |
| jina-code | cpu/cuda | 0.31 | -0.53..2.27 | below cutoff |
| jina-code | cuda/cpu | -0.69 | -1.97..1.52 | below cutoff |
| jina-code | cpu/mixed | 0.50 | -0.35..2.45 | below cutoff |
| jina-code | cuda/mixed | 0.00 | -0.89..2.42 | below cutoff |
| bge-m3 | cpu/cuda | 0.00 | 0.00..0.00 | below cutoff |
| bge-m3 | cuda/cpu | 0.00 | 0.00..0.00 | below cutoff |
| bge-m3 | cpu/mixed | 0.00 | 0.00..0.00 | below cutoff |
| bge-m3 | cuda/mixed | 0.00 | 0.00..0.00 | below cutoff |
| arctic-256 | cpu/cuda | 1.39 | 0.17..3.08 | **flag** |
| arctic-256 | cuda/cpu | 1.43 | 0.08..3.34 | **flag** |
| arctic-256 | cpu/mixed | 1.19 | -0.69..3.47 | **flag** |
| arctic-256 | cuda/mixed | 0.19 | -1.53..2.22 | below cutoff |

### Paired candidate-minus-Arctic-768 nDCG deltas

Same query/index combination on both sides; negative means below Arctic 768.
The Arctic-256 rows are also the paired dimension-reduction comparisons.

| Recipe | Query / index | nDCG delta | 95% delta interval |
| --- | --- | ---: | ---: |
| arctic | cpu/cpu | 0.00 | 0.00..0.00 |
| arctic | cuda/cuda | 0.00 | 0.00..0.00 |
| arctic | cpu/cuda | 0.00 | 0.00..0.00 |
| arctic | cuda/cpu | 0.00 | 0.00..0.00 |
| arctic | cpu/mixed | 0.00 | 0.00..0.00 |
| arctic | cuda/mixed | 0.00 | 0.00..0.00 |
| granite | cpu/cpu | -7.47 | -12.42..-2.80 |
| granite | cuda/cuda | -8.33 | -12.79..-4.07 |
| granite | cpu/cuda | -7.89 | -12.27..-3.65 |
| granite | cuda/cpu | -7.14 | -11.94..-2.57 |
| granite | cpu/mixed | -7.34 | -11.97..-2.86 |
| granite | cuda/mixed | -8.36 | -12.95..-3.87 |
| e5 | cpu/cpu | -7.04 | -12.22..-2.28 |
| e5 | cuda/cuda | -6.05 | -10.94..-1.53 |
| e5 | cpu/cuda | -7.16 | -12.09..-2.52 |
| e5 | cuda/cpu | -6.62 | -11.65..-2.12 |
| e5 | cpu/mixed | -9.77 | -15.29..-4.79 |
| e5 | cuda/mixed | -8.46 | -13.51..-3.83 |
| jina-code | cpu/cpu | -52.13 | -60.77..-43.10 |
| jina-code | cuda/cuda | -52.74 | -61.36..-43.91 |
| jina-code | cpu/cuda | -52.61 | -61.48..-43.60 |
| jina-code | cuda/cpu | -50.85 | -59.18..-42.33 |
| jina-code | cpu/mixed | -52.25 | -61.28..-43.19 |
| jina-code | cuda/mixed | -52.77 | -61.55..-43.80 |
| bge-m3 | cpu/cpu | -6.43 | -11.18..-1.76 |
| bge-m3 | cuda/cuda | -7.28 | -11.71..-2.83 |
| bge-m3 | cpu/cuda | -6.85 | -11.43..-2.15 |
| bge-m3 | cuda/cpu | -6.10 | -10.73..-1.49 |
| bge-m3 | cpu/mixed | -6.30 | -10.82..-1.75 |
| bge-m3 | cuda/mixed | -7.31 | -11.98..-2.54 |
| arctic-256 | cpu/cpu | -2.04 | -4.07..-0.12 |
| arctic-256 | cuda/cuda | -1.38 | -3.31..0.45 |
| arctic-256 | cpu/cuda | -2.34 | -4.27..-0.76 |
| arctic-256 | cuda/cpu | -1.62 | -3.24..-0.15 |
| arctic-256 | cpu/mixed | -1.59 | -3.72..0.37 |
| arctic-256 | cuda/mixed | -1.60 | -3.48..0.14 |

### Homogeneous provider deltas and truncation

Delta is CUDA/CUDA minus CPU/CPU. No selected queries were truncated.

| Recipe | nDCG delta | 95% delta interval | Truncated documents | Document fraction |
| --- | ---: | ---: | ---: | ---: |
| arctic | 0.86 | -1.18..2.97 | 712/5183 | 13.74% |
| granite | 0.00 | 0.00..0.00 | 712/5183 | 13.74% |
| e5 | 1.84 | -1.16..4.92 | 731/5183 | 14.10% |
| jina-code | 0.25 | -2.17..2.73 | 892/5183 | 17.21% |
| bge-m3 | 0.00 | 0.00..0.00 | 712/5183 | 13.74% |
| arctic-256 | 1.52 | -0.94..4.09 | 712/5183 | 13.74% |

Raw per-query rankings, both-reference top-1/top-10 agreement, cosine/batch
diagnostics and exact token totals remain in the JSON/JSONL evidence.
See [assessment and limitations](README.md).
