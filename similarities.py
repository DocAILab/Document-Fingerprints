"""
文档指纹的相似度计算算法集合
"""
from keras.preprocessing.text import Tokenizer

import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from simhash import Simhash
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash
from utils import remove_stopwords
import gensim.downloader as api
from scipy.spatial import distance
from collections import Counter
from gensim.similarities import WmdSimilarity

# 汉明距
# jaccard
# WMD
# levenshtein
# tfidf-based cosine

# 字符串距离计算
"""
最长公共子序列-解决文档长短差异问题；
"""


def hamming_distance(fingerprint1, fingerprint2, cal_simi=True):
    """
    计算两个指纹的汉明距离。
    :param fingerprint1: 第一个指纹, 可以是Simhash, int或str
    :param fingerprint2: 第二个指纹, 可以是Simhash, int或str（需保证两个指纹等长）
    :param cal_simi: 是否计算相似度
    :return: 汉明距离, [相似度]
    """
    if isinstance(fingerprint1, Simhash):
        hamming_dist = fingerprint1.distance(fingerprint2)
        if cal_simi:
            if fingerprint1.f != fingerprint2.f:
                raise ValueError("fingerprint1 and fingerprint2 in type of Simhash must have same dimension!")
            simi = 1 - (hamming_dist / fingerprint1.f)  # fingerprint1.f为simhash指纹维度
            return hamming_dist, simi
        return hamming_dist
    elif isinstance(fingerprint1, int):
        bin1 = bin(fingerprint1)[2:]
        bin2 = bin(fingerprint2)[2:]
        if len(bin1) != len(bin2):
            raise ValueError("input fingerprints in type of int must have same dimension!")
        xor_result = fingerprint1 ^ fingerprint2
        hamming_dist = bin(xor_result).count('1')
        if cal_simi:
            simi = 1 - (hamming_dist / len(bin1))
            return hamming_dist, simi
        return hamming_dist
    elif isinstance(fingerprint1, str):
        if len(fingerprint1) != len(fingerprint2):
            raise ValueError("input fingerprints in type of str must have same dimension!")
        hamming_dist = sum(bit1 != bit2 for bit1, bit2 in zip(list(fingerprint1), list(fingerprint2)))
        if cal_simi:
            dim = len(fingerprint1)
            simi = 1 - (hamming_dist / dim)
            return hamming_dist, simi
        return hamming_dist
    else:
        raise TypeError("input fingerprints must be in type of Simhash, int or str")


def levenshtein_distance(text1, text2, cal_simi=True):
    """
    计算两个文本之间的编辑距离和相似度。
    :param text1: 第一个文本
    :param text2: 第二个文本
    :param cal_simi: 是否计算相似度
    :return: 编辑距离, [相似度]
    """
    dist = Levenshtein.distance(text1, text2)
    if cal_simi:
        max_length = max(len(text1), len(text2))
        simi = 1 - (dist / max_length)
        return dist, simi
    return dist


def jaccard_similarity(data1, data2):
    """
    计算集合的jaccard相似度。
    :param data1: 第一个数据，set/list/MinHash
    :param data2: 第二个数据，set/list/MinHash
    :return: jaccard值，float
    """
    if isinstance(data1, set):
        intersection = len(data1.intersection(data2))
        union = len(data1.union(data2))
        return intersection / union
    elif isinstance(data1, list):
        data1 = set(data1)
        data2 = set(data2)
        intersection = len(data1.intersection(data2))
        union = len(data1.union(data2))
        return intersection / union
    elif isinstance(data1, MinHash):
        return data1.jaccard(data2)  # 使用minhash估计的jaccard值
    else:
        raise TypeError("input fingerprints must be in type of Simhash, set or list")


def multiset_jaccard_similarity(data1, data2):
    """
    计算两个多重集合的jaccard相似度。
    :param data1: 第一个数据，list
    :param data2: 第二个数据，list
    :return:
    """
    if isinstance(data1, list):
        counter1 = Counter(data1)
        counter2 = Counter(data2)
        intersection = sum((counter1 & counter2).values())
        union = sum((counter1 | counter2).values())
        similarity = intersection / union
        return similarity
    else:
        raise TypeError("input data must be in type of list")


def wmd_distance(text1, text2):
    """
    计算两个文本之间的wmd距离。
    :param text1: 第一个文本，str
    :param text2: 第二个文本，str
    :return: wmd距离，float
    """
    text1 = remove_stopwords(text1)
    text2 = remove_stopwords(text2)
    model = api.load('word2vec-google-news-300')
    dist = model.wmdistance(text1, text2)
    return dist


def wmd_similarity(text1, text2):
    """
        计算两个文本之间的wmd距离。
        :param text1: 第一个文本，str
        :param text2: 第二个文本，str
        :return: wmd相似度，float
    """
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    model = Word2Vec([tokens1, tokens2], min_count=1)
    similarity_index = WmdSimilarity([tokens1], model, num_best=1)
    similarity = similarity_index[tokens1][0][1]
    return similarity


def tfidf_based_cosine_similarity(vector1, vector2):
    """
    Parameters:
    - vector1: 第一个文本
    - vector2: 第二个文本

    Returns:
    - 曼哈顿相似度
    """
    cosine_sim = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
    return cosine_sim


def manhattan_similarity(vector1, vector2):
    """
    Parameters:
    - vector1: 第一个文本
    - vector2: 第二个文本

    Returns:
    - 曼哈顿相似度
    """
    manhattan_dist = distance.cityblock(vector1, vector2)
    similarity = 1 / (1 + manhattan_dist)  # 将距离转化为相似度
    return similarity


def mahalanobis_distance(vector1, vector2, cov_estimator=None):
    """
    Parameters:
    - vector1: 第一个文本
    - vector2: 第二个文本

    Returns:
    - 马氏距离
    """
    # 如果未提供协方差矩阵估计器，则使用单位矩阵
    if cov_estimator is None:
        cov_matrix = np.identity(len(vector1))
    else:
        cov_matrix = cov_estimator.fit([vector1, vector2]).covariance_
    # 计算马氏距离
    mahalanobis_dist = distance.mahalanobis(vector1, vector2, np.linalg.inv(cov_matrix))
    # 将马氏距离映射为相似度
    similarity = np.exp(-0.5 * mahalanobis_dist ** 2)
    return similarity


def calculate_similarity(fingerprint1, fingerprint2, algorithm_type='cosine', text_to_vector_method='tf222df'):
    vector1 = fingerprint1
    vector2 = fingerprint2

    if isinstance(fingerprint1, str) and isinstance(fingerprint2, str):
        if text_to_vector_method == 'tfidf':
            vectorized = TfidfVectorizer(use_idf=True, norm='l2')
            # todo: 优化模型的训练
            tfidf_matrix = vectorized.fit_transform([fingerprint1, fingerprint2])
            vector1 = tfidf_matrix[0].toarray().flatten()
            vector2 = tfidf_matrix[1].toarray().flatten()
        elif text_to_vector_method == 'word2vec':
            tokens1 = word_tokenize(fingerprint1.lower())
            tokens2 = word_tokenize(fingerprint2.lower())
            # todo: 优化模型的训练
            model = Word2Vec([tokens1, tokens2], vector_size=100, window=5, min_count=1, workers=4)
            vector1 = sum(model.wv[word] for word in tokens1) / len(tokens1)
            vector2 = sum(model.wv[word] for word in tokens2) / len(tokens2)
        elif text_to_vector_method == 'onehot':
            tokenizer = Tokenizer()
            # todo: 优化模型的训练
            tokenizer.fit_on_texts([fingerprint1, fingerprint2])
            seq1 = tokenizer.texts_to_sequences([fingerprint1])[0]
            seq2 = tokenizer.texts_to_sequences([fingerprint2])[0]
            one_hot_encoding_seq1 = np.zeros((len(tokenizer.word_index) + 1,))
            one_hot_encoding_seq2 = np.zeros((len(tokenizer.word_index) + 1,))
            for index in seq1:
                one_hot_encoding_seq1[index] = 1
            for index in seq2:
                one_hot_encoding_seq2[index] = 1
            vector1 = one_hot_encoding_seq1
            vector2 = one_hot_encoding_seq2
        elif text_to_vector_method == 'pad':
            target_length = max(len(fingerprint1), len(fingerprint2))
            fingerprint1 = fingerprint1.ljust(target_length, '\0')
            fingerprint2 = fingerprint2.ljust(target_length, '\0')
            vector1 = [ord(char) for char in fingerprint1]
            vector1 = np.array(vector1)
            vector2 = [ord(char) for char in fingerprint2]
            vector2 = np.array(vector2)
        elif text_to_vector_method == 'truncate':
            target_length = min(len(fingerprint1), len(fingerprint2))
            fingerprint1 = fingerprint1[:target_length]
            fingerprint2 = fingerprint2[:target_length]
            vector1 = [ord(char) for char in fingerprint1]
            vector1 = np.array(vector1)
            vector2 = [ord(char) for char in fingerprint2]
            vector2 = np.array(vector2)

    if algorithm_type == 'hamming':
        return hamming_distance(fingerprint1, fingerprint2)
    elif algorithm_type == 'levenshtein':
        return levenshtein_distance(fingerprint1, fingerprint2)
    elif algorithm_type == 'wmd':
        return wmd_similarity(fingerprint1, fingerprint2)
    elif algorithm_type == 'cosine':
        return tfidf_based_cosine_similarity(vector1, vector2)
    elif algorithm_type == 'manhattan':
        return manhattan_similarity(vector1, vector2)
    elif algorithm_type == 'mahalanobis':
        return mahalanobis_distance(vector1, vector2)
    elif algorithm_type == 'jaccard_similarity':
        return jaccard_similarity(vector1, vector2)
    elif algorithm_type == 'multiset_jaccard_similarity':
        return multiset_jaccard_similarity(vector1, vector2)
    else:
        raise ValueError("Unsupported algorithm_type.")


def evaluate_similarity_functions(inputs1, inputs2):
    algorithm_types = ['hamming', 'levenshtein', 'cosine', 'manhattan', 'mahalanobis']
    text_to_vector_methods = ['tfidf', 'word2vec', 'onehot', 'pad', 'truncate']

    for algorithm_type in algorithm_types:
        for text_to_vector_method in text_to_vector_methods:
            try:
                print(f"Algorithm: {algorithm_type}, Text-to-Vector Method: {text_to_vector_method}:loading.........")
                similarity = calculate_similarity(inputs1, inputs2, algorithm_type, text_to_vector_method)
                print(
                    f"Algorithm: {algorithm_type}, Text-to-Vector Method: {text_to_vector_method}, Similarity: {similarity}")
            except ValueError as e:
                print(f"Algorithm: {algorithm_type}, Text-to-Vector Method: {text_to_vector_method}, Error: {e}")


if __name__ == "__main__":
    print("This is a test for calculating similarities.")
    str1 = '北京增值税电子普通发票.pdf'
    str2 = '福建增值税电子普通发票.pdf'
    str3 = '福建工程学院计算机学院培养方案.pdf'

    print(evaluate_similarity_functions(str1, str2))
