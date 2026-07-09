#!/bin/bash

dataset_name="csfcube"
dataset_dir="./datasets/CSFCube-H3D"
results_dir="./results"
run_name="smoke_minhash_jaccard"

python main.py \
  --dataset_name "$dataset_name" \
  --dataset_dir "$dataset_dir" \
  --results_dir "$results_dir" \
  --facet all \
  --hash_name MinHash \
  --similarity_name jaccard \
  --hash_dim 16 \
  --ngram 3 \
  --run_name "$run_name"
