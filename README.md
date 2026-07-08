# Document Fingerprints

This repository contains the code for **H3D**, an unsupervised benchmark for
document fingerprinting, similarity search, and text deduplication. It evaluates
classic locality-sensitive or lexical fingerprinting methods together with
embedding-based hashing methods on scientific-document similarity datasets.

Supported fingerprint methods include:

- `SimHash`
- `MinHash`
- `Winnowing`
- `FuzzyHash`
- `FlyHash`
- `BIGAE`
- `EmbedBinHash`
- `BGEHash`
- `LSH-BGE`
- `fc-BGE`

Supported similarity functions include:

- `hamming`
- `jaccard`
- `multiset_jaccard`
- `levenshtein`
- `wmd`
- `cosine`
- `manhattan`
- `mahalanobis`

## Repository Layout

```text
.
|-- main.py                     # Main CLI for scoring and evaluation
|-- hashes.py                   # Fingerprint generation methods
|-- similarities.py             # Similarity and distance functions
|-- embmodels.py                # Embedding-related helpers
|-- fingerprint.py              # Standalone fingerprint utilities
|-- requirements.txt            # Python dependencies
|-- run.sh                      # Example shell command
|-- evaluation/
|   |-- dataset.py              # Dataset loader
|   |-- evaluate.py             # Scoring and metric computation
|   |-- metrics.py              # MAP, NDCG, precision/recall metrics
|   |-- model.py                # SimilarityModel wrapper
|   `-- utils.py                # Result path helpers
|-- fp_generate_methods/        # Individual fingerprint implementations
|-- datasets/
|   |-- CSFCube/
|   |-- RELISH/
|   `-- DBpedia/
`-- results/                    # Generated scores and evaluation files
```

## Installation

Create a Python environment and install the dependencies:

```bash
conda create -n h3d python=3.10
conda activate h3d
pip install -r requirements.txt
```

Some dependencies, especially `pyssdeep` / `ssdeep`, can be platform-specific.
If installation fails on Windows, use the Windows package already listed in
`requirements.txt` or install the system `ssdeep` library before reinstalling
the Python package.

The embedding-based methods (`BGEHash`, `EmbedBinHash`, `LSH-BGE`, `fc-BGE`)
expect local BGE model folders under:

```text
FlagEmbedding/bge-base-en-v1.5
FlagEmbedding/bge-large-en-v1.5
FlagEmbedding/bge-small-en-v1.5
```

If these folders are not available, start with non-embedding methods such as
`SimHash`, `MinHash`, `Winnowing`, `FuzzyHash`, or `FlyHash`.

## Data

The evaluation code expects each dataset directory to contain files in the same
format as the included RELISH and CSFCube folders:

```text
datasets/RELISH/
|-- abstracts-relish.jsonl
|-- test-pid2anns-relish.json
|-- relish-queries-release.csv
`-- relish-evaluation_splits.json

datasets/CSFCube/
|-- abstracts-csfcube.jsonl
|-- test-pid2anns-csfcube-background.json
|-- test-pid2anns-csfcube-method.json
|-- test-pid2anns-csfcube-result.json
`-- csfcube-queries-release.csv
```

Each row in `abstracts-<dataset>.jsonl` should contain at least:

```json
{
  "paper_id": "paper id",
  "title": "paper title",
  "abstract": ["sentence 1", "sentence 2"]
}
```

For CSFCube-style faceted evaluation, each record should also include
`pred_labels`.

## Quick Start

Run a full RELISH scoring and evaluation job with MinHash:

```bash
python main.py \
  --dataset_name relish \
  --dataset_dir datasets/RELISH \
  --hash_name MinHash \
  --similarity_name jaccard \
  --hash_dim 128 \
  --ngram 3 \
  --run_name minhash_jaccard
```

Run SimHash with Hamming similarity:

```bash
python main.py \
  --dataset_name relish \
  --dataset_dir datasets/RELISH \
  --hash_name SimHash \
  --similarity_name hamming \
  --hash_dim 128 \
  --run_name simhash_hamming
```

Run Winnowing on CSFCube for all facets:

```bash
python main.py \
  --dataset_name csfcube \
  --dataset_dir datasets/CSFCube \
  --facet all \
  --hash_name Winnowing \
  --similarity_name multiset_jaccard \
  --ngram 5 \
  --winnowing_window 5 \
  --run_name winnowing_ngram5_window5
```

Run LSH-BGE with cosine similarity:

```bash
python main.py \
  --dataset_name relish \
  --dataset_dir datasets/RELISH \
  --hash_name LSH-BGE \
  --similarity_name cosine \
  --dense_embedding_model bge-base-en-v1.5 \
  --lsh_num_bucket 8 \
  --lsh_hash_size 128 \
  --run_name lsh_bge_8_128
```

## Common Command-Line Arguments

| Argument | Description |
| --- | --- |
| `--dataset_name` | Dataset name. Common values: `relish`, `csfcube`. |
| `--dataset_dir` | Directory containing the dataset files. |
| `--results_dir` | Base directory for generated outputs. Default: `results`. |
| `--facet` | CSFCube facet: `background`, `method`, `result`, or `all`. |
| `--actions` | Run `score`, `evaluate`, or both. Default: both. |
| `--hash_name` | Fingerprint method. |
| `--similarity_name` | Similarity or distance function. |
| `--hash_dim` | Fingerprint size for methods such as SimHash and MinHash. |
| `--ngram` | N-gram size used by MinHash and Winnowing. |
| `--winnowing_window` | Window size used by Winnowing. |
| `--run_name` | Output subdirectory name for this run. |
| `--dense_embedding_model` | Local BGE model folder name. |
| `--lsh_num_bucket` | Number of LSH buckets for `LSH-BGE`. |
| `--lsh_hash_size` | Total hash size for `LSH-BGE`. |

## Outputs

Results are written under:

```text
results/<dataset_name>/<hash_name+similarity_name>/<run_name>/
```

For `LSH-BGE`, the output folder includes the LSH parameters:

```text
results/<dataset_name>/LSH-BGE-<similarity>-<lsh_num_bucket>-<lsh_hash_size>/<run_name>/
```

Typical output files are:

- `scores.json` or `scores-<facet>.json`: ranked candidate papers for each query.
- `query-evaluations.csv` or `query-evaluations-<facet>.csv`: per-query metrics.
- `aggregated-evaluations.csv` or `aggregated-evaluations-<facet>.csv`: averaged metrics.
- `run.log`: runtime and evaluation logs.

The main reported metrics include average precision and `NDCG@20`.

## Reproducing Multiple Runs

You can use `run.sh` as a template:

```bash
bash run.sh
```

Or run several method and similarity combinations manually:

```bash
python main.py --dataset_name relish --dataset_dir datasets/RELISH --hash_name MinHash --similarity_name jaccard --run_name minhash_jaccard
python main.py --dataset_name relish --dataset_dir datasets/RELISH --hash_name SimHash --similarity_name hamming --run_name simhash_hamming
python main.py --dataset_name csfcube --dataset_dir datasets/CSFCube --facet all --hash_name Winnowing --similarity_name multiset_jaccard --run_name winnowing_all_facets
```

To recompute metrics from an existing `scores.json` without rescoring:

```bash
python main.py \
  --dataset_name relish \
  --dataset_dir datasets/RELISH \
  --hash_name MinHash \
  --similarity_name jaccard \
  --run_name minhash_jaccard \
  --actions evaluate
```

## Notes

- `FuzzyHash` uses ssdeep-style fuzzy hashing and may require longer text to
  produce useful fingerprints.
- `BIGAE` expects precomputed hash files passed through `--hash_path`.
- `wmd` may require extra model downloads and can be slow on large datasets.
- Large datasets and embedding-based methods can take substantial time; start
  with a small run or a classic hashing method when checking the environment.

## Citation

If you use this repository, please cite the associated H3D paper/preprint when
it becomes available.
