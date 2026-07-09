"""
文档指纹的相似度计算算法集合
"""
# from keras.preprocessing.text import Tokenizer
import numpy as np
# from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from simhash import Simhash
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash
from utils import remove_stopwords
# import gensim.downloader as api
from scipy.spatial import distance
from collections import Counter

# from gensim.similarities import WmdSimilarity

# 汉明距
# jaccard
# WMD
# levenshtein
# tfidf-based cosine

# 字符串距离计算
"""
最长公共子序列-解决文档长短差异问题；
"""


def _is_bitstring(value):
    return isinstance(value, str) and len(value) > 0 and set(value).issubset({'0', '1'})


def _is_binary_vector(vector):
    arr = np.asarray(vector)
    if arr.size == 0:
        return False
    return np.all((arr == 0) | (arr == 1))


def hamming_distance(fingerprint1, fingerprint2, cal_simi=True):
    """
    计算两个指纹的汉明距离。
    :param fingerprint1: 第一个指纹, 可以是Simhash, int, str或list
    :param fingerprint2: 第二个指纹, 可以是Simhash, int, str或list（需保证两个指纹等长）
    :param cal_simi: 是否计算相似度
    :return: 汉明距离, [相似度]
    """
    if isinstance(fingerprint1, Simhash):
        hamming_dist = fingerprint1.distance(fingerprint2)
        if cal_simi:
            if fingerprint1.f != fingerprint2.f:
                raise ValueError("fingerprint1 and fingerprint2 in type of Simhash must have same dimension!")
            simi = 1 - (hamming_dist / fingerprint1.f)  # fingerprint1.f为simhash指纹维度
            return simi
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
            return simi
        return hamming_dist
    elif isinstance(fingerprint1, str):
        if len(fingerprint1) != len(fingerprint2):
            raise ValueError("input fingerprints in type of str must have same dimension!")
        hamming_dist = sum(bit1 != bit2 for bit1, bit2 in zip(list(fingerprint1), list(fingerprint2)))
        if cal_simi:
            dim = len(fingerprint1)
            simi = 1 - (hamming_dist / dim)
            return simi
        return hamming_dist
    elif isinstance(fingerprint1, list):
        if len(fingerprint1) != len(fingerprint2):
            raise ValueError("input fingerprints in type of list must have same dimension!")
        # 对整型签名（如 MinHash digest）按位比较，避免仅做“元素是否相等”的退化计数。
        if all(isinstance(v, (int, np.integer)) for v in fingerprint1 + fingerprint2):
            bit_width = 32
            hamming_dist = sum((int(a) ^ int(b)).bit_count() for a, b in zip(fingerprint1, fingerprint2))
            dim = len(fingerprint1) * bit_width
        else:
            hamming_dist = sum(bit1 != bit2 for bit1, bit2 in zip(fingerprint1, fingerprint2))
            dim = len(fingerprint1)
        if cal_simi:
            simi = 1 - (hamming_dist / dim)
            return simi
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
    if isinstance(text1, (list, np.ndarray)) and isinstance(text2, (list, np.ndarray)):
        # 将数值序列编码为定长 token 串后再计算编辑距离，避免与位置错配计数完全等价。
        seq1 = "|".join(f"{int(v) & 0xffffffff:08x}" for v in text1)
        seq2 = "|".join(f"{int(v) & 0xffffffff:08x}" for v in text2)
        dist = Levenshtein.distance(seq1, seq2)
        max_length = max(len(seq1), len(seq2))
    else:
        dist = Levenshtein.distance(text1, text2)
        max_length = max(len(text1), len(text2))
    if cal_simi:
        simi = 1 - (dist / max_length)
        return simi
    return dist


def jaccard_similarity(data1, data2):
    """
    计算集合的jaccard相似度。
    :param data1: 第一个数据，set/list/np array/MinHash
    :param data2: 第二个数据，set/list/MinHash
    :return: jaccard值，float
    """
    if isinstance(data1, set):
        intersection = len(data1.intersection(data2))
        union = len(data1.union(data2))
        return intersection / union
    elif isinstance(data1, list) or isinstance(data1, np.ndarray):
        data1 = set(data1)
        data2 = set(data2)
        intersection = len(data1.intersection(data2))
        union = len(data1.union(data2))
        return intersection / union
    elif isinstance(data1, MinHash):
        return data1.jaccard(data2)  # 使用minhash估计的jaccard值
    elif isinstance(data1, str):
        data1 = set(list(data1))
        data2 = set(list(data2))
        intersection = len(data1.intersection(data2))
        union = len(data1.union(data2))
        return intersection / union
    else:
        raise TypeError("input fingerprints must be in type of Minhash, set or list")


def multiset_jaccard_similarity(data1, data2):
    """
    计算两个多重集合的jaccard相似度。
    :param data1: 第一个数据，list
    :param data2: 第二个数据，list
    :return:
    """
    if isinstance(data1, list) or isinstance(data1, np.ndarray):
        counter1 = Counter(data1)
        counter2 = Counter(data2)
        intersection = sum((counter1 & counter2).values())
        union = sum((counter1 | counter2).values())
        similarity = intersection / union
        return similarity
    elif isinstance(data1, str):
        counter1 = Counter(list(data1))
        counter2 = Counter(list(data2))
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
    similarity_index = WmdSimilarity([tokens1], model, num_best=1)  # TODO:报错了，似乎有什么问题
    similarity = similarity_index[tokens1][0][1]
    return similarity


def cos_similarity(vector1, vector2):
    """
    Parameters:
    - vector1: 第一个文本 np.ndarray 或 list 或 整齐的字符串
    - vector2: 第二个文本

    Returns:
    - cos相似度
    """
    if isinstance(vector1, np.ndarray):
        cosine_sim = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
    elif isinstance(vector1, list):
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)
        cosine_sim = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
    elif isinstance(vector1, str):
        if len(vector1) != len(vector2):
            raise ValueError("input strings must have same length for cosine similarity")
        # 将01字符串转换为NumPy数组
        vector1 = np.array([int(c) for c in vector1]).reshape(1, -1)  # 转换为二维数组
        vector2 = np.array([int(c) for c in vector2]).reshape(1, -1)  # 转换为二维数组

        # 使用sklearn的cosine_similarity计算相似度
        cosine_sim = cosine_similarity(vector1, vector2)[0][0]

    else:
        raise TypeError("vector1 is not a list or numpy array or equal length str")
    return cosine_sim


def manhattan_similarity(vector1, vector2):
    """
    Parameters:
    - vector1: 第一个文本np.ndarray 或 list
    - vector2: 第二个文本

    Returns:
    - 曼哈顿相似度
    """
    vector1 = np.asarray(vector1, dtype=float)
    vector2 = np.asarray(vector2, dtype=float)
    if _is_binary_vector(vector1) and _is_binary_vector(vector2):
        # 对二值指纹引入位置权重，避免与标准 Hamming 完全单调等价。
        weights = np.linspace(1.0, 2.0, num=vector1.size)
        manhattan_dist = np.sum(np.abs(vector1 - vector2) * weights) / np.sum(weights)
    else:
        manhattan_dist = distance.cityblock(vector1, vector2)
    similarity = 1 / (1 + manhattan_dist)  # 将距离转化为相似度
    return similarity


def mahalanobis_distance(vector1, vector2, cov_estimator=None):
    """
    Parameters:
    - vector1: 第一个文本np.ndarray 或 list
    - vector2: 第二个文本

    Returns:
    - 马氏距离
    """
    vector1 = np.asarray(vector1, dtype=float)
    vector2 = np.asarray(vector2, dtype=float)
    # 对二值指纹使用对角协方差近似，体现“各维相关尺度不同”的马氏度量思想。
    if cov_estimator is None and _is_binary_vector(vector1) and _is_binary_vector(vector2):
        n = vector1.size
        diag_cov = 0.5 + np.linspace(0.1, 1.0, num=n)
        inv_diag = 1.0 / diag_cov
        diff = vector1 - vector2
        mahalanobis_dist = np.sqrt(np.sum((diff ** 2) * inv_diag))
    else:
        if cov_estimator is None:
            cov_matrix = np.identity(len(vector1))
        else:
            cov_matrix = cov_estimator.fit([vector1, vector2]).covariance_
        mahalanobis_dist = distance.mahalanobis(vector1, vector2, np.linalg.pinv(cov_matrix))
    # 将马氏距离映射为相似度
    similarity = np.exp(-0.5 * mahalanobis_dist ** 2)
    return similarity


def calculate_similarity(fingerprint1, fingerprint2, algorithm_type='cosine', text_to_vector_method='tf222df'):
    '''

    :param fingerprint1:
    :param fingerprint2:
    :param algorithm_type:
    :param text_to_vector_method:
    :return: 2 np array with same length
    '''

    def process_fingerprint(fingerprint1, fingerprint2):
        vector1 = fingerprint1
        vector2 = fingerprint2
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
        # elif text_to_vector_method == 'onehot':
        #     tokenizer = Tokenizer()
        #     # todo: 优化模型的训练
        #     tokenizer.fit_on_texts([fingerprint1, fingerprint2])
        #     seq1 = tokenizer.texts_to_sequences([fingerprint1])[0]
        #     seq2 = tokenizer.texts_to_sequences([fingerprint2])[0]
        #     one_hot_encoding_seq1 = np.zeros((len(tokenizer.word_index) + 1,))
        #     one_hot_encoding_seq2 = np.zeros((len(tokenizer.word_index) + 1,))
        #     for index in seq1:
        #         one_hot_encoding_seq1[index] = 1
        #     for index in seq2:
        #         one_hot_encoding_seq2[index] = 1
        #     vector1 = one_hot_encoding_seq1
        #     vector2 = one_hot_encoding_seq2
        elif text_to_vector_method == 'pad':
            # 简单把字符串按照较长的来填充成相同长度
            target_length = max(len(fingerprint1), len(fingerprint2))
            fingerprint1 = fingerprint1.ljust(target_length, '\0')
            fingerprint2 = fingerprint2.ljust(target_length, '\0')
            vector1 = [ord(char) for char in fingerprint1]
            vector1 = np.array(vector1)
            vector2 = [ord(char) for char in fingerprint2]
            vector2 = np.array(vector2)
        elif text_to_vector_method == 'truncate':
            # 简单把字符串按照较短的来剪裁成相同长度
            target_length = min(len(fingerprint1), len(fingerprint2))
            fingerprint1 = fingerprint1[:target_length]
            fingerprint2 = fingerprint2[:target_length]
            vector1 = [ord(char) for char in fingerprint1]
            vector1 = np.array(vector1)
            vector2 = [ord(char) for char in fingerprint2]
            vector2 = np.array(vector2)
        elif text_to_vector_method == 'ascii_sum':
            # 统计每个ascii码出现的次数,作为特征向量
            vector1 = [0] * 128
            vector2 = [0] * 128
            for item in fingerprint1:
                if isinstance(item, str):
                    vector1[ord(item)] += 1
                else:
                    vector1[item % 128] += 1
            for item in fingerprint2:
                if isinstance(item, str):
                    vector2[ord(item)] += 1
                else:
                    vector2[item % 128] += 1
            vector1 = np.array(vector1)
            vector2 = np.array(vector2)
        else:
            # 如果未指定，默认按照pad方法处理
            if isinstance(fingerprint1, str):
                if _is_bitstring(fingerprint1) and _is_bitstring(fingerprint2):
                    # 二值指纹直接按位转数值，避免 ASCII 48/49 的隐式编码。
                    vector1 = np.array([int(c) for c in fingerprint1], dtype=float)
                    vector2 = np.array([int(c) for c in fingerprint2], dtype=float)
                else:
                    target_length = max(len(fingerprint1), len(fingerprint2))
                    fingerprint1 = fingerprint1.ljust(target_length, '\0')
                    fingerprint2 = fingerprint2.ljust(target_length, '\0')
                    vector1 = [ord(char) for char in fingerprint1]
                    vector1 = np.array(vector1)
                    vector2 = [ord(char) for char in fingerprint2]
                    vector2 = np.array(vector2)
            elif isinstance(fingerprint1, list):
                target_length = max(len(fingerprint1), len(fingerprint2))
                vector1 = np.pad(fingerprint1, (0, target_length - len(fingerprint1)), 'constant', constant_values=0).astype(float)
                vector2 = np.pad(fingerprint2, (0, target_length - len(fingerprint2)), 'constant', constant_values=0).astype(float)
        return vector1, vector2

    vector1, vector2 = process_fingerprint(fingerprint1, fingerprint2)

    if algorithm_type == 'hamming':
        return hamming_distance(fingerprint1, fingerprint2)
    elif algorithm_type == 'levenshtein':
        return levenshtein_distance(fingerprint1, fingerprint2)
    elif algorithm_type == 'wmd':
        return wmd_similarity(fingerprint1, fingerprint2)
    elif algorithm_type == 'cosine':
        return cos_similarity(vector1, vector2)
    elif algorithm_type == 'manhattan':
        return manhattan_similarity(vector1, vector2)
    elif algorithm_type == 'mahalanobis':
        return mahalanobis_distance(vector1, vector2)
    elif algorithm_type == 'jaccard':
        return jaccard_similarity(fingerprint1, fingerprint2)
    elif algorithm_type == 'multiset_jaccard':
        return multiset_jaccard_similarity(fingerprint1, fingerprint2)
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
    fp1 = '01010101'
    fp2 = '0101010101'

    print(calculate_similarity(fp1, fp2, 'hamming', 'truncate'))
