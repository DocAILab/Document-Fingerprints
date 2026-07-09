import pandas as pd
import hashes
from similarities import calculate_similarity
from tqdm import tqdm
import heapq
import time

loginfo = []


def load_data(train_path, test_path):
    # 自定义函数，用于过滤掉不符合条件的行
    def filter_rows(filename):
        filtered_data = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                if len(line.strip()) >= 20:
                    filtered_data.append(line.strip().split(',', 1))
        return filtered_data

    train = filter_rows(train_path)
    test = filter_rows(test_path)
    loginfo.append(f"len(train_fps) = {len(train)}, len(test_fps) = {len(test)}")
    return train, test


def get_fingerprints(train_data, test_data, fp_method='Winnowing'):
    if fp_method == 'BVGAE':
        return [sub[-1] for sub in train_data], [sub[-1] for sub in test_data]
    train_fps = []
    test_fps = []
    if fp_method == 'Winnowing':
        fp_func = hashes.fp_with_winnowing
        loginfo.append(f"fp_func: {fp_method}; parameters: n=5, w=5, hash_func=None")
    else:
        raise ValueError("Invalid fp method")
    start_time = time.time()
    for i in tqdm(range(len(train_data)), desc='Calculating training fingerprints'):
        train_fps.append(fp_func(train_data[i][1], n=5, w=5, hash_func=None))
    for i in tqdm(range(len(test_data)), desc='Calculating testing fingerprints'):
        test_fps.append(fp_func(test_data[i][1], n=5, w=5, hash_func=None))
    end_time = time.time() - start_time
    loginfo.append(f'Time taken to calculate fingerprints: {end_time}s')
    return train_fps, test_fps


def get_topK_nearest(train_fps, test_fps, similarity_method='multiset_jaccard', K=100):
    '''
    对每一个test内容计算和所有train内容的相似度，并选出最相似的K歌
    :param train_fps:
    :param test_fps:
    :param similarity_method:
    :param K:
    :return: topK_nearest, 二维列表，第一维是test数据集下标，第二维是选出的train中最相关K个文档的下标
    '''
    topK_nearest = []
    similarities = []
    loginfo.append(f"similarity_func: {similarity_method}; parameters: text_to_vector_method='None'")
    start_time = time.time()
    for i in tqdm(range(len(test_fps)), desc='Calculating top K nearest'):
        similarity_row = []
        for j in range(len(train_fps)):
            similarity = calculate_similarity(test_fps[i], train_fps[j], algorithm_type=similarity_method
                                              , text_to_vector_method='None')
            similarity_row.append(similarity)
        similarities.append(similarity_row)
        topK_nearest.append(heapq.nlargest(K, range(len(train_fps)), key=lambda x: similarity_row[x]))

    end_time = time.time() - start_time
    loginfo.append(f'Time taken to calculate top K nearest: {end_time}')

    # 保存相似度信息到CSV文件
    similarities_df = pd.DataFrame(similarities)
    similarities_df.to_csv('similarities.csv', index=False, header=False)

    return topK_nearest


def get_av_precision(train_data, test_data, topK_nearest, K=100):
    av_precision = 0.
    for i in range(len(topK_nearest)):
        precision = 0
        for j in topK_nearest[i]:
            precision += 1 if test_data[i][0] == train_data[j][0] else 0
        precision = precision / K * 100
        av_precision += precision / len(topK_nearest)
    return av_precision


if __name__ == '__main__':
    # train_path = './train.csv'
    # test_path = './test.csv'
    # fp_method = 'Winnowing'
    # similarity_method = 'multiset_jaccard'
    train_path = "./hashresult/hash_result_train_vgae_mb_DBpedia-mse+klloss0.01-l5.5-b8-e1-h10.csv"
    test_path = "./hashresult/hash_result_test_vgae_mb_DBpedia-mse+klloss0.01-l5.5-b8-e1-h10.csv"
    fp_method = 'BVGAE'
    similarity_method = 'levenshtein'

    K = 100
    train_data, test_data = load_data(train_path, test_path)
    train_fps, test_fps = get_fingerprints(train_data, test_data,
                                           fp_method=fp_method)
    topK_nearest = get_topK_nearest(train_fps, test_fps, similarity_method=similarity_method, K=K)
    av_precision = get_av_precision(train_data, test_data, topK_nearest, K=K)
    print(av_precision)
    loginfo.append("av_precision: " + str(av_precision))
    with open(f'./log-{fp_method}-{similarity_method}-K{K}.txt', 'w', encoding='utf-8') as f:
        for l in loginfo:
            f.write(l + "\n")
