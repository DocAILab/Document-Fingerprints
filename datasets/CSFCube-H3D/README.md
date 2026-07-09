CSFCube-H3D dataset description
================================

Dataset
-------

This directory contains the CSFCube data processed for the H3D benchmark. The processed release preserves the original paper texts, query papers, candidate pools, and facet relevance annotations. The changes are limited to file naming, identifier normalization, and H3D/BIGAE-style input files used by the H3D evaluation pipeline.

Source
------

Original dataset: CSFCube - A Test Collection of Computer Science Research Articles for Faceted Query by Example.

Original authors: Sheshera Mysore, Tim O'Gorman, Andrew McCallum, and Hamed Zamani.

Original release: https://github.com/iesl/CSFCube

Original paper: https://openreview.net/forum?id=8Y50dBbmGU

The unprocessed source files are kept in ../CSFCube-origin.

Files in this release
---------------------

abstracts-csfcube.jsonl

JSONL file with 4,207 paper records. Each line contains paper_id, metadata, title, abstract, pred_labels_truncated, and pred_labels. The abstract field is a list of sentences. The pred_labels values are background_label, objective_label, method_label, result_label, and other_label.

csfcube-queries-release.csv

CSV file with 50 query papers and metadata. The original query identifier column is normalized from pid to paper_id.

test-pid2anns-csfcube-background.json
test-pid2anns-csfcube-method.json
test-pid2anns-csfcube-result.json

Facet-level query-candidate annotation files. Each top-level key is a query paper_id. Each value contains cands, relevance_max, relevance_min, relevance_ann1, relevance_ann2, and relevance_adju. The adjudicated relevance scores range from 0 to 3. The files contain 16 background queries, 17 method queries, and 17 result queries.

csfcube_train_background.0.json
csfcube_train_method.0.json
csfcube_train_result.0.json

H3D/BIGAE-style JSON files generated from abstracts-csfcube.jsonl. Each file contains 4,207 records with the same schema:

{
  "src": ["text segment", "...", "paper title"],
  "tgt": ["paper title"],
  "labels": ["paper_id"]
}

For the background file, src uses sentences labeled background_label or objective_label. For the method and result files, src uses sentences labeled method_label and result_label, respectively. The title is appended to src in all three files.

change_style.py

Utility script used to generate the three facet-specific H3D/BIGAE-style JSON files.

statistic.py

Utility script for checking file formats and basic statistics.

Processing summary
------------------

1. Renamed abstracts-csfcube-preds.jsonl to abstracts-csfcube.jsonl for the H3D data-loader naming convention.
2. Renamed the query metadata identifier column from pid to paper_id.
3. Kept the original background, method, and result query-candidate annotation structures unchanged.
4. Generated three facet-specific H3D/BIGAE-style JSON files.
5. Merged objective_label into the background facet for H3D facet processing.
6. Left original auxiliary files, including guidelines, datasheet, readable annotations, license, and original evaluation scripts, in ../CSFCube-origin.

Evaluation use
--------------

CSFCube-H3D is used for facet-level scientific document similarity ranking in H3D. The ranking tasks use background, method, and result annotations and report metrics such as MAP and NDCG@20.

License and citation
--------------------

This processed release derives from CSFCube. Please follow the license terms of the original CSFCube release and cite the original dataset paper when using this data:

@inproceedings{mysore2021csfcube,
  title={{CSFC}ube - A Test Collection of Computer Science Research Articles for Faceted Query by Example},
  author={Sheshera Mysore and Tim O'Gorman and Andrew McCallum and Hamed Zamani},
  booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2)},
  year={2021},
  url={https://openreview.net/forum?id=8Y50dBbmGU}
}

Acknowledgments
---------------

The original CSFCube dataset was built using the Semantic Scholar Open Research Corpus and uses pretrained models from sequential_sentence_classification and SPECTER. H3D acknowledges the original dataset creators and associated source projects.
