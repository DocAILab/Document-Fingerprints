DBpedia-H3D dataset description
===============================

Dataset
-------

This directory contains DBpedia text classification data processed for the H3D benchmark. In H3D, DBpedia-H3D is used as a label-based retrieval benchmark: each test document is treated as a query, documents from the training split are ranked by a hashing or embedding method, and retrieved documents are considered relevant when they share the same class label as the query.

Source
------

Original knowledge base: DBpedia.

Reference paper: DBpedia: A multilingual cross-domain knowledge base.

Original authors: Pablo N. Mendes, Max Jakob, and Christian Bizer.

Reference URL: https://madoc.bib.uni-mannheim.de/59280/1/570_Paper.pdf

The CSV files in this directory are the DBpedia text classification splits used by this repository. They contain text records paired with DBpedia class labels.

Files in this release
---------------------

train.csv
test.csv

CSV files with one example per line. Each line follows this structure:

label,text

The label is a DBpedia class identifier represented as a string. The text field contains the document content. The files contain 50,000 training rows and 5,000 test rows with 14 labels. The H3D conversion script filters out rows shorter than 20 characters before writing the JSON files.

dbpedia_train.0.json
dbpedia_test.0.json

H3D/BIGAE-style JSON files generated from train.csv and test.csv. The processed files contain 49,994 training records and 5,000 test records. Each record has the same schema:

{
  "src": ["document text"],
  "tgt": ["none"],
  "labels": ["class_label"]
}

The labels field is a one-element list so that DBpedia-H3D follows the same H3D/BIGAE-style schema as the other processed datasets.

change_style.py

Utility script used to convert train.csv and test.csv into dbpedia_train.0.json and dbpedia_test.0.json.

evaluation.py

Utility script for the label-based retrieval protocol. For each test query, it retrieves the top K training documents and computes precision from the fraction of retrieved documents whose class label matches the query label.

Processing summary
------------------

1. Kept the original train.csv and test.csv text-classification files.
2. Parsed each CSV row into a class label and document text.
3. Filtered out rows shorter than 20 characters during JSON conversion.
4. Generated H3D/BIGAE-style JSON files with src, tgt, and labels fields.
5. Used ["none"] as tgt because DBpedia-H3D is a class-label retrieval dataset, not a title-to-document scientific-paper dataset.
6. Stored class labels as one-element lists to match the processed H3D schema used by CSFCube-H3D and RELISH-H3D.

Evaluation use
--------------

DBpedia-H3D evaluates retrieval over class labels. For every query document in the test split, the evaluation searches the training split, selects the top 100 nearest documents, and computes precision as:

precision@100 = number of retrieved documents with the same label as the query / 100

The final score is the average precision@100 over all test queries.

License and citation
--------------------

This processed release derives from DBpedia text classification data. Please follow the license terms of the original DBpedia data source and cite the DBpedia reference when using this data:

@book{mendes2012dbpedia,
  title={DBpedia: A multilingual cross-domain knowledge base},
  author={Mendes, Pablo N and Jakob, Max and Bizer, Christian},
  year={2012},
  booktitle={Proceedings of the Eighth International Conference on Language Resources and Evaluation, LREC},
  publisher={European Language Resources Association (ELRA)},
  url={https://madoc.bib.uni-mannheim.de/59280/1/570_Paper.pdf}
}

Acknowledgments
---------------

H3D acknowledges the DBpedia project and the maintainers of the DBpedia data resources from which this text classification benchmark is derived.
