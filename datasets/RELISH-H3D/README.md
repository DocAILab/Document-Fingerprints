RELISH-H3D dataset description
==============================

Dataset
-------

This directory contains RELISH-Aspire data processed for the H3D benchmark. RELISH-H3D is used as an overall scientific document similarity ranking benchmark. The processed release preserves the paper texts, query papers, candidate pools, relevance annotations, and dev/test split identifiers used by the RELISH-Aspire release. The changes are limited to file naming and H3D/BIGAE-style input files used by the H3D evaluation pipeline.

Source
------

Original dataset: RELISH-Aspire.

Reference paper: Large expert-curated database for benchmarking document similarity detection in biomedical literature search.

Original authors: Peter Brown and Yaoqi Zhou.

Original release: https://figshare.com/articles/dataset/RELISH-Aspire/19425506

Files in this release
---------------------

abstracts-relish.jsonl

JSONL file with 163,170 paper records. Each line contains title, abstract, and paper_id. The abstract field is a list of sentences.

relish-queries-release.csv

CSV file with 3,274 query-paper metadata rows. The columns are paper_id and title.

test-pid2anns-relish.json

Query-candidate annotation file with 3,190 annotated queries. Each top-level key is a query paper_id. Each value contains cands and relevance_adju. The candidate-pool sizes in this processed file are 53, 54, 56, 57, 58, 59, or 60. The adjudicated relevance scores range from 0 to 2.

relish-evaluation_splits.json

Evaluation split file with 1,637 dev query IDs and 1,637 test query IDs.

relish_train.0.json

H3D/BIGAE-style JSON file generated from abstracts-relish.jsonl. It contains 163,170 records with the following schema:

{
  "src": ["paper title", "abstract sentence", "..."],
  "tgt": ["paper title"],
  "labels": ["paper_id"]
}

The src field starts with the title and then appends the abstract sentences.

hash_results.json
hash_result_all.json

Auxiliary precomputed hash outputs retained with this processed release. These files are not required to understand the source dataset format.

change_style.py

Utility script used to convert abstracts-relish.jsonl into relish_train.0.json.

statistic.py

Utility script for checking file formats and basic statistics.

Processing summary
------------------

1. Kept the RELISH-Aspire paper records, query metadata, annotations, and split identifiers.
2. Converted abstracts-relish.jsonl into the H3D/BIGAE-style src, tgt, labels schema.
3. Built src from the paper title followed by abstract sentences.
4. Stored paper_id as a one-element labels list to match the processed H3D schema used by CSFCube-H3D and DBpedia-H3D.
5. Retained the original overall relevance annotations without adding facet-level labels.

Evaluation use
--------------

RELISH-H3D evaluates overall scientific document similarity ranking. H3D uses the query-candidate annotations and the dev/test split file to report retrieval metrics such as MAP and NDCG@20. Unlike CSFCube-H3D, RELISH-H3D does not provide background, method, or result facet labels.

License and citation
--------------------

This processed release derives from RELISH-Aspire. Please follow the license terms of the original RELISH-Aspire release and cite the original dataset paper when using this data:

@article{RELISH,
  title={Large expert-curated database for benchmarking document similarity detection in biomedical literature search},
  author={Brown, Peter and Zhou, Yaoqi},
  journal={Database},
  volume={2019},
  pages={baz085},
  year={2019},
  publisher={Oxford University Press}
}

Acknowledgments
---------------

H3D acknowledges the RELISH-Aspire authors and the biomedical literature resources used to build the original document similarity benchmark.
