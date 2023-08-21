"""
文档指纹的相似度计算算法集合
"""
from simhash import Simhash
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash
from utils import remove_stopwords
import gensim.downloader as api

# 汉明距
# jaccard
# WMD

def hamming_distance(fingerprint1, fingerprint2, cal_simi=True, dim=None):
    """
    计算两个指纹的汉明距离。
    :param fingerprint1: 第一个指纹
    :param fingerprint2: 第二个指纹
    :param cal_simi: 是否计算相似度
    :param dim: 指纹维度，当fingerprint为int，且需计算相似度时必须输入
    :return: 汉明距离, [相似度]
    """
    if isinstance(fingerprint1, Simhash):
        # xor_result = fingerprint1.value ^ fingerprint2.value
        # hamming_dist = bin(xor_result).count('1')
        hamming_dist = fingerprint1.distance(fingerprint2)
        if cal_simi:
            simi = 1 - (hamming_dist / fingerprint1.f)  # fingerprint1.f为simhash指纹维度
            return hamming_dist, simi
        return hamming_dist
    elif isinstance(fingerprint1, int):
        xor_result = fingerprint1 ^ fingerprint2
        hamming_dist = bin(xor_result).count('1')
        if cal_simi:
            if dim is None:
                raise ValueError("parameter 'dim' must be int!")
            simi = 1 - (hamming_dist / dim)
            return hamming_dist, simi
        return hamming_dist
    elif isinstance(fingerprint1, str):
        hamming_dist = sum(bit1 != bit2 for bit1, bit2 in zip(fingerprint1, fingerprint2))
        if cal_simi:
            dim = len(fingerprint1)
            simi = 1 - (hamming_dist / dim)
            return hamming_dist, simi
        return hamming_dist


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


