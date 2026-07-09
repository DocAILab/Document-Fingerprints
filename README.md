# Document Fingerprints

This repository contains the code for **H3D**, a benchmark for unsupervised document fingerprinting, similarity search, and fine-grained document deduplication. It compares classic lexical or locality-sensitive fingerprints with embedding-based hashing methods under a shared ranking-evaluation protocol.

## Methods

Implemented fingerprint methods:

- `SimHash`
- `MinHash`
- `Winnowing`
- `FuzzyHash`
- `FlyHash`
- `BIGAE` for precomputed external hash codes
- `EmbedBinHash`
- `BGEHash`
- `LSH-BGE`
- `fc-BGE`

Implemented similarity functions:

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
|-- main.py                      # Main CLI for scoring and evaluation
|-- hashes.py                    # Fingerprint generation methods
|-- similarities.py              # Similarity and distance functions
|-- embmodels.py                 # Optional embedding helper functions
|-- fingerprint.py               # Standalone fingerprint utilities
|-- requirements.txt             # Runtime dependencies
|-- run.sh                       # Example command template
|-- evaluation/
|   |-- dataset.py               # Dataset loader
|   |-- evaluate.py              # Scoring and metric computation
|   |-- metrics.py               # MAP, NDCG, precision/recall metrics
|   |-- model.py                 # SimilarityModel wrapper
|   `-- utils.py                 # Result path helpers
|-- fp_generate_methods/         # Individual fingerprint implementations
`-- datasets/
    |-- CSFCube-H3D/             # Processed CSFCube files and dataset README
    |-- RELISH-H3D/              # RELISH metadata/annotations and dataset README
    `-- DBpedia-H3D/             # DBpedia split files and dataset README
```

Generated evaluation outputs are written to `results/`, which is intentionally ignored by git.

## Installation

Create a Python environment and install the base dependencies:

```bash
conda create -n h3d python=3.10
conda activate h3d
pip install -r requirements.txt
```

`FuzzyHash` uses ssdeep-style fuzzy hashing. On some systems this may require installing the system `ssdeep` library before installing the Python package.

BGE-based methods require optional embedding dependencies:

```bash
pip install FlagEmbedding sentence-transformers
```

The default BGE model argument is `BAAI/bge-base-en-v1.5`. You can also pass a local model directory through `--dense_embedding_model`.

## Data

This repository keeps lightweight metadata, annotations, conversion scripts, and dataset documentation. `CSFCube-H3D` includes the processed files needed by the quick-start commands below. Some full text or generated files for the larger datasets are too large for a normal GitHub code repository and are ignored by git. Recreate those files from the original sources with the conversion scripts in each dataset folder.

Each dataset folder has its own `README.md` with source, license, citation, processing, and file-format notes.

Expected scientific-document JSONL format:

```json
{
  "paper_id": "paper id",
  "title": "paper title",
  "abstract": ["sentence 1", "sentence 2"]
}
```

CSFCube additionally uses `pred_labels` for background, method, and result facet processing.

The H3D/BIGAE-style generated files use:

```json
{
  "src": ["text segment", "..."],
  "tgt": ["target text or none"],
  "labels": ["paper_id_or_class_label"]
}
```

## Quick Start

Run a small CSFCube-H3D smoke test with MinHash:

```bash
python main.py \
  --dataset_name csfcube \
  --dataset_dir datasets/CSFCube-H3D \
  --facet all \
  --hash_name MinHash \
  --similarity_name jaccard \
  --hash_dim 16 \
  --ngram 3 \
  --run_name smoke_minhash_jaccard
```

Run SimHash with Hamming similarity on CSFCube-H3D:

```bash
python main.py \
  --dataset_name csfcube \
  --dataset_dir datasets/CSFCube-H3D \
  --facet all \
  --hash_name SimHash \
  --similarity_name hamming \
  --hash_dim 128 \
  --run_name simhash_hamming
```

Run Winnowing on all CSFCube facets:

```bash
python main.py \
  --dataset_name csfcube \
  --dataset_dir datasets/CSFCube-H3D \
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
  --dataset_dir datasets/RELISH-H3D \
  --hash_name LSH-BGE \
  --similarity_name cosine \
  --dense_embedding_model BAAI/bge-base-en-v1.5 \
  --lsh_num_bucket 8 \
  --lsh_hash_size 128 \
  --run_name lsh_bge_8_128
```

To recompute metrics from existing scores without rescoring:

```bash
python main.py \
  --dataset_name csfcube \
  --dataset_dir datasets/CSFCube-H3D \
  --facet all \
  --hash_name MinHash \
  --similarity_name jaccard \
  --run_name smoke_minhash_jaccard \
  --actions evaluate
```

RELISH-H3D and DBpedia-H3D can be evaluated with the same CLI after their generated H3D JSON files have been recreated from the original data sources. See each dataset `README.md` for the source, license, citation, and conversion notes.

## CLI Arguments

| Argument | Description |
| --- | --- |
| `--dataset_name` | Dataset key used for file names, usually `csfcube`, `relish`, or `dbpedia`. |
| `--dataset_dir` | Dataset directory, such as `datasets/CSFCube-H3D`. |
| `--results_dir` | Output base directory. Default: `results`. |
| `--facet` | CSFCube facet: `background`, `method`, `result`, or `all`. |
| `--actions` | Run `score`, `evaluate`, or both. |
| `--hash_name` | Fingerprint method. |
| `--similarity_name` | Similarity or distance function. |
| `--hash_dim` | Fingerprint size for methods such as SimHash and MinHash. |
| `--ngram` | N-gram size used by MinHash and Winnowing. |
| `--winnowing_window` | Window size used by Winnowing. |
| `--run_name` | Output subdirectory name for this run. |
| `--dense_embedding_model` | Hugging Face model name or local path for BGE-based methods. |
| `--lsh_num_bucket` | Number of LSH buckets for `LSH-BGE`. |
| `--lsh_hash_size` | Total hash size for `LSH-BGE`. |
| `--hash_path` | Path to precomputed hash codes for `BIGAE`. |

## Outputs

Results are written under:

```text
results/<dataset_name>/<hash_name+similarity_name>/<run_name>/
```

Typical output files:

- `scores.json` or `scores-<facet>.json`
- `query-evaluations.csv` or `query-evaluations-<facet>.csv`
- `aggregated-evaluations.csv` or `aggregated-evaluations-<facet>.csv`
- `run.log`

The main reported metrics include average precision and `NDCG@20`.

## Dataset Citations

Please cite the original datasets when using their processed H3D versions. The dataset-specific README files include BibTeX entries and license notes for CSFCube, RELISH-Aspire, and DBpedia-derived data.

## Citation

If you use this repository, please cite the associated H3D paper/preprint when it becomes available.
