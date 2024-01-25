import argparse
import logging
import sys
from evaluation.utils import *
from evaluation.model import SimilarityModel
from evaluation.dataset import EvalDataset
from evaluation.evaluate import score, evaluate


def main(args):
    # check validity of command-line arguments
    # check_args(args)

    # init results dir
    results_dir = get_results_dir(args.results_dir, args.dataset_name, args.hash_name + '+' + args.similarity_name,
                                  args.run_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # init log
    if args.log_fname is not None:
        log_dir = results_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(level='INFO', format='%(message)s', filename=os.path.join(results_dir, args.log_fname))
    else:
        logging.basicConfig(level='INFO', format='%(message)s', stream=sys.stdout)

    # init model and dataset
    dataset = EvalDataset(name=args.dataset_name, root_path=args.dataset_dir)
    model = SimilarityModel(hash_name=args.hash_name, similarity_name=args.similarity_name, args=args)
    # if 'encode' in args.actions or 'score' in args.actions:
    #     logging.info(f'Loading model: {args.model_name}')
    #     model = get_model(model_name=args.model_name)
    #     logging.info(f'Loading dataset: {args.dataset_name}')
    #     if args.cache:
    #         # init cache
    #         encodings_filename = get_encodings_filename(results_dir)
    #         logging.info(f'Setting model cache at: {encodings_filename}')
    #         model.set_encodings_cache(encodings_filename)
    #
    # if 'encode' in args.actions:
    #     # cache model's encodings of entire dataset
    #     encode(model, dataset)

    if 'score' in args.actions:
        # score model on dataset's test pool
        if args.facet == 'all':
            for facet in FACETS:
                score(model, dataset, facet=facet, scores_filename=get_scores_filename(results_dir, facet=facet))
        else:
            score(model, dataset, facet=args.facet, scores_filename=get_scores_filename(results_dir, facet=args.facet))

    if 'evaluate' in args.actions:
        # evaluate metrics for model scores
        evaluate(results_dir,
                 facet=args.facet,
                 dataset=dataset)


def parse_args():
    parser = argparse.ArgumentParser()
    # 基本参数
    # parser.add_argument('--model_name', required=True, help='The name of the model to run. Choose from a model_name '
    #                                                         'with an implementation in evaluation_models.get_model')
    parser.add_argument('--dataset_name', help='Dataset to evaluate similarities on',
                        choices=['gorcmatscicit', 'csfcube', 'relish', 'treccovid',
                                 'scidcite', 'scidcocite', 'scidcoread', 'scidcoview'], default='relish')
    parser.add_argument('--dataset_dir',
                        help="Dir to dataset files (e.g. abstracts-{dataset}.jsonl)", default='datasets/RELISH')
    parser.add_argument('--results_dir',
                        help="Results base dir to store encodings cache, scores and metrics", default='results')
    parser.add_argument('--facet', choices=['result', 'method', 'background', 'all'], default=None,
                        help='Relevant only to csfcube dataset. Select a facet to use for the task'
                             ' of faceted similarity search. If "all", tests all facets one at a time.')
    # parser.add_argument('--cache', action='store_true',
    #                     help='Use if we would like to cache the encodings of papers.\n'
    #                     'If action "encode" is selected, this is set automatically to true.')
    parser.add_argument('--run_name', help='Name of this evaluation run.\n'
                                           'Saves results under results_dir/hash_name+similarity_name/run_name/\n'
                                           'to allow different results to same model_name',
                        default='run1-classic_params')
    # parser.add_argument('--trained_model_path', help='Basename for a trained model we would like to evaluate on.')
    parser.add_argument('--log_fname', help='Filename of log file', default="run1" + '.log')
    parser.add_argument('--actions', choices=['score', 'evaluate'], nargs="+", default=['score', 'evaluate'],
                        help="""'Encode' creates vector representations for the entire dataset.
                                'Score' calculates similarity scores on the dataset's test pool.
                                'Evaluate' calculates metrics based on the similarity scores predicted.
                                By default does all three.""")

    # 各种哈希和相似度的选择及相关参数
    parser.add_argument('--hash_name', default='FuzzyHash',
                        choices=['SimHash', 'MinHash', 'Winnowing', 'FuzzyHash', 'FlyHash'],
                        help='Hashing method')
    parser.add_argument('--similarity_name', default='levenshtein',
                        choices=['hamming', 'jaccard', 'multiset_jaccard', 'levenshtein', 'wmd', 'cosine', 'manhattan',
                                 'mahalanobis'],
                        help='Similarity method')
    parser.add_argument('--hash_dim', type=int, default=128, help='Hash size for SimHash and MinHash')
    parser.add_argument('--hash_func', default=None, choices=['md5', 'sha1', 'sha256', 'sha512'],
                        help='Hash function or None for defaultHash')
    parser.add_argument('--ngram', type=int, default=5, help='Ngram分词大小')
    parser.add_argument('--winnowing_window', type=int, default=5, help='Winnowing窗口大小')
    parser.add_argument('--text_to_vector_method', default=None,
                        choices=['tfidf', 'word2vec', 'onehot', 'pad', 'truncate'],
                        help='Text to vector method')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
