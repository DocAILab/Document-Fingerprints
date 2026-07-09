import argparse
import logging
import os
import sys
import time

from evaluation.dataset import EvalDataset
from evaluation.evaluate import evaluate, score
from evaluation.model import SimilarityModel
from evaluation.utils import FACETS, get_results_dir, get_scores_filename


def main(args):
    if "LSH" in args.hash_name:
        model_dir_name = (
            f"{args.hash_name}-{args.similarity_name}-"
            f"{args.lsh_num_bucket}-{args.lsh_hash_size}"
        )
    else:
        model_dir_name = args.hash_name + "+" + args.similarity_name

    results_dir = get_results_dir(
        results_basedir=args.results_dir,
        dataset_name=args.dataset_name,
        model_name=model_dir_name,
        run_name=args.run_name,
    )
    os.makedirs(results_dir, exist_ok=True)

    if args.log_fname is not None:
        logging.basicConfig(
            level="INFO",
            format="%(message)s",
            filename=os.path.join(results_dir, args.log_fname),
        )
    else:
        logging.basicConfig(level="INFO", format="%(message)s", stream=sys.stdout)

    dataset = EvalDataset(name=args.dataset_name, root_path=args.dataset_dir)
    model = SimilarityModel(
        hash_name=args.hash_name,
        similarity_name=args.similarity_name,
        args=args,
    )

    if "score" in args.actions:
        start_time = time.time()
        if args.facet == "all":
            for facet in FACETS:
                score(
                    model,
                    dataset,
                    facet=facet,
                    scores_filename=get_scores_filename(results_dir, facet=facet),
                )
        else:
            score(
                model,
                dataset,
                facet=args.facet,
                scores_filename=get_scores_filename(results_dir, facet=args.facet),
            )
        logging.info(f"Finished all scoring in {time.time() - start_time} seconds")

    if "evaluate" in args.actions:
        evaluate(results_dir, facet=args.facet, dataset=dataset)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run H3D document fingerprint scoring and evaluation."
    )
    parser.add_argument(
        "--dataset_name",
        help="Dataset key used in dataset file names.",
        choices=[
            "gorcmatscicit",
            "csfcube",
            "relish",
            "treccovid",
            "scidcite",
            "scidcocite",
            "scidcoread",
            "scidcoview",
        ],
        default="relish",
    )
    parser.add_argument(
        "--dataset_dir",
        help="Directory containing dataset files.",
        default="datasets/RELISH-H3D",
    )
    parser.add_argument(
        "--results_dir",
        help="Base directory for scores and metrics.",
        default="results",
    )
    parser.add_argument(
        "--facet",
        choices=["result", "method", "background", "all", None],
        default=None,
        help="CSFCube facet to evaluate. Use 'all' to evaluate all facets.",
    )
    parser.add_argument("--log_fname", help="Log filename.", default="run.log")
    parser.add_argument(
        "--actions",
        choices=["score", "evaluate"],
        nargs="+",
        default=["score", "evaluate"],
        help="Run scoring, evaluation, or both.",
    )
    parser.add_argument(
        "--hash_path",
        default="",
        help="Path to precomputed hash results for BIGAE.",
    )
    parser.add_argument(
        "--hash_name",
        default="MinHash",
        choices=[
            "SimHash",
            "MinHash",
            "Winnowing",
            "FuzzyHash",
            "FlyHash",
            "BIGAE",
            "EmbedBinHash",
            "BGEHash",
            "LSH-BGE",
            "fc-BGE",
        ],
        help="Fingerprint method.",
    )
    parser.add_argument(
        "--similarity_name",
        default="jaccard",
        choices=[
            "hamming",
            "jaccard",
            "multiset_jaccard",
            "levenshtein",
            "wmd",
            "cosine",
            "manhattan",
            "mahalanobis",
        ],
        help="Similarity or distance function.",
    )
    parser.add_argument(
        "--hash_dim",
        type=int,
        default=128,
        help="Hash size for methods such as SimHash and MinHash.",
    )
    parser.add_argument(
        "--hash_func",
        default=None,
        choices=["md5", "sha1", "sha256", "sha512"],
        help="Optional hash function.",
    )
    parser.add_argument("--ngram", type=int, default=3, help="N-gram size.")
    parser.add_argument(
        "--winnowing_window",
        type=int,
        default=5,
        help="Winnowing window size.",
    )
    parser.add_argument(
        "--run_name",
        help="Name for this evaluation run.",
        default="run1",
    )
    parser.add_argument(
        "--text_to_vector_method",
        default="None",
        choices=["tfidf", "word2vec", "onehot", "pad", "truncate", "ascii_sum"],
        help="Text-to-vector method used by selected similarity functions.",
    )
    parser.add_argument("--lsh_num_bucket", type=int, default=2)
    parser.add_argument("--lsh_hash_size", type=int, default=128)
    parser.add_argument(
        "--dense_embedding_model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Hugging Face model name or local path for BGE-based methods.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
