"""
文档指纹的相似度计算算法集合
"""
import numpy as np
from simhash import Simhash
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash
from utils import remove_stopwords
import gensim.downloader as api
from scipy.spatial import distance


# 汉明距
# jaccard
# WMD
# levenshtein
# tfidf-based cosine

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


def tfidf_based_cosine_similarity(text1, text2):
    """
    基于TF-IDF计算两个文本之间的余弦相似度。
    :param text1: 第一个文本
    :param text2: 第二个文本
    :return: 余弦相似度
    """
    vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return cosine_sim


def manhattan_similarity(text1, text2):
    """
    计算两个文本之间的曼哈顿相似度

    Parameters:
    - text1: 第一个文本
    - text2: 第二个文本
    - vectorizer: 词袋模型向量化器

    Returns:
    - 曼哈顿相似度
    """
    vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    vector1 = tfidf_matrix[0].toarray().flatten()
    vector2 = tfidf_matrix[1].toarray().flatten()

    manhattan_dist = distance.cityblock(vector1, vector2)
    similarity = 1 / (1 + manhattan_dist)  # 将距离转化为相似度
    return similarity


def mahalanobis_distance(text1, text2, cov_estimator=None):
    """
    计算两个文档指纹之间的马氏距离

    Parameters:
    - fingerprint1: 第一个文档指纹
    - fingerprint2: 第二个文档指纹
    - cov_estimator: 协方差矩阵估计器，默认为None，表示使用单位矩阵

    Returns:
    - 马氏距离
    """
    vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    vector1 = tfidf_matrix[0].toarray().flatten()
    vector2 = tfidf_matrix[1].toarray().flatten()

    # 如果未提供协方差矩阵估计器，则使用单位矩阵
    if cov_estimator is None:
        cov_matrix = np.identity(len(vector1))
    else:
        cov_matrix = cov_estimator.fit([vector1, vector2]).covariance_

    # 计算马氏距离
    mahalanobis_dist = distance.mahalanobis(vector1, vector2, np.linalg.inv(cov_matrix))
    
    # 将马氏距离映射为相似度
    similarity = np.exp(-0.5 * mahalanobis_dist ** 2)

    return mahalanobis_dist


if __name__ == "__main__":
    print("This is a test for calculating similarities.")
    str1 = '北京增值税电子普通发票.pdf'
    str2 = '福建增值税电子普通发票.pdf'
    str3 = '福建工程学院计算机学院培养方案.pdf'

    print(tfidf_based_cosine_similarity(str1, str2))
