import logging

import pandas as pd

from evaluation.metrics import compute_metrics
from evaluation.utils import *
import collections
from tqdm import tqdm
from typing import Union
from evaluation.model import SimilarityModel


def score(model: SimilarityModel,
          dataset: EvalDataset,
          facet: Union[str, None],
          scores_filename: str):
    """
    Calculate similarity scores between queries and their candidates in a test pool
    :param model: Model to test
    :param dataset: Dataset to take test pool from
    :param facet: Facet of query to use. Relevant only to CSFcube dataset.
    :param scores_filename: Saves results here
    :return:
    """

    # load test pool
    test_pool = dataset.get_test_pool(facet=facet)

    log_msg = f"Scoring {len(test_pool)} queries in {dataset.name}"
    if facet is not None:
        log_msg += f', facet: {facet}'
    logging.info(log_msg)

    # Get model similarity scores between each query and its candidates
    results = collections.defaultdict(list)
    for query_pid, query_pool in tqdm(list(test_pool.items())):

        # get query encoding
        # if faceted, also filter the encoding by facet
        # query_encoding = model.get_encoding(pids=[query_pid], dataset=dataset)[query_pid]
        # if facet is not None:
        #     query_encoding = model.get_faceted_encoding(query_encoding, facet, dataset.get(query_pid))
        query_content = dataset.get(query_pid)
        query_encoding = query_content['TITLE'] + "".join(query_content['ABSTRACT'])

        # get candidates encoding
        candidate_pids = query_pool['cands']
        # candidate_encodings = model.get_encoding(pids=candidate_pids, dataset=dataset)

        # For calculate similarities of each candidate to query encoding
        candidate_similarities = dict()
        for candidate_pid in candidate_pids:
            candidate_content = dataset.get(candidate_pid)
            similarity = model.get_similarity(query_encoding, candidate_content['TITLE'] + "".join(candidate_content['ABSTRACT']))
            candidate_similarities[candidate_pid] = similarity
        # sort candidates by similarity, ascending (lower score == closer encodings)
        # sorted_candidates = sorted(candidate_similarities.items(), key=lambda i: i[1], reverse=True)
        # results[query_pid] = [(cpid, -1*sim) for cpid, sim in sorted_candidates]
        # sort candidates by similarity, descending (higher score == closer encodings)
        sorted_candidates = sorted(candidate_similarities.items(), key=lambda i: i[1], reverse=True)
        results[query_pid] = [(cpid, sim) for cpid, sim in sorted_candidates]

    # write scores
    with codecs.open(scores_filename, 'w', 'utf-8') as fp:
        json.dump(results, fp)
        logging.info(f'Wrote: {scores_filename}')


def evaluate(results_dir: str,
             facet: Union[str, None],
             dataset: EvalDataset,
             comet_exp_key=None):
    """
    Compute metrics based on a model's similarity scores on a dataset's test pool
    Assumes score() has already been called with relevant model_name, dataset and facet
    :param results_dir: Directory where scores are saved
    :param facet: Facet of query to use. Relevant only to CSFcube dataset.
    :param dataset: Dataset to take test pool from
    :param comet_exp_key: Optional comet experiment key to log metrics to
    :return:
    """
    logging.info('Computing metrics')

    # load score results
    results = dict()
    if facet == 'all':
        for facet_i in FACETS:
            results[facet_i] = load_score_results(results_dir, dataset, facet_i)
    else:
        facet_key = 'unfaceted' if facet is None else facet
        results[facet_key] = load_score_results(results_dir, dataset, facet)

    # get queries metadata
    query_metadata = dataset.get_query_metadata()
    query_test_dev_split = dataset.get_test_dev_split()
    threshold_grade = dataset.get_threshold_grade()

    # compute metrics per query
    metrics = []
    metric_columns = None
    for facet_i, facet_results in results.items():
        for query_id, sorted_relevancies in facet_results.items():
            query_metrics = compute_metrics(sorted_relevancies,
                                            pr_atks=[5, 10, 20],
                                            threshold_grade=threshold_grade)
            if metric_columns is None:
                metric_columns = list(query_metrics.keys())
            query_metrics['facet'] = facet_i
            query_metrics['split'] = 'test' if query_test_dev_split is None else query_test_dev_split[query_id]
            query_metrics['paper_id'] = query_id
            query_metrics['title'] = query_metadata.loc[query_id]['title']
            metrics.append(query_metrics)
    metrics = pd.DataFrame(metrics)

    # write evaluations file per query
    query_metrics_filename = get_evaluations_filename(results_dir, facet, aggregated=False)
    metrics.to_csv(query_metrics_filename, index=False)
    logging.info(f'Wrote: {query_metrics_filename}')

    # aggergate metrics per (facet, dev/test_split)
    aggregated_metrics = []
    for facet_i in metrics.facet.unique():
        for split in metrics.split.unique():
            agg_results = metrics[(metrics.facet == facet_i) & (metrics.split == split)][metric_columns].mean().round(
                4).to_dict()
            logging.info(f'----------Results for {split}/{facet_i}----------')
            logging.info('\n'.join([f'{k}\t{agg_results[k]}' for k in ('av_precision', 'ndcg%20')]))
            agg_results['facet'] = facet_i
            agg_results['split'] = split
            aggregated_metrics.append(agg_results)
    if facet == 'all':
        for split in metrics.split.unique():
            agg_results = metrics[metrics.split == split][metric_columns].mean().round(4).to_dict()
            logging.info(f'----------Results for {split}/{facet}----------')
            logging.info('\n'.join([f'{k}\t{agg_results[k]}' for k in ('av_precision', 'ndcg%20')]))
            agg_results['facet'] = facet
            agg_results['split'] = split
            aggregated_metrics.append(agg_results)
    aggregated_metrics = pd.DataFrame(aggregated_metrics)

    # Write evaluation file aggregated per (facet, dev/test_split)
    aggregated_metrics_filename = get_evaluations_filename(results_dir, facet, aggregated=True)
    aggregated_metrics.to_csv(aggregated_metrics_filename, index=False)
    logging.info(f'Wrote: {aggregated_metrics_filename}')
