import hashlib
from collections import defaultdict
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein


def generate_simhash_lsh_fingerprint(texts, k=3, num_perm=64):
    """
    使用Simhash + LSH算法生成文本的指纹（哈希值）。
    :param texts: 要生成指纹的文本列表。
    :param k: Simhash的k-gram参数。
    :param num_perm: Simhash的排列数。
    :return: 文本的指纹（哈希值）。
    """
    # 创建LSH索引
    lsh_index = defaultdict(list)

    # 生成Simhash值并添加到LSH索引中
    for text in texts:
        simhash = generate_simhash(text, k=k, num_perm=num_perm)
        lsh_index[simhash].append(text)

    return lsh_index


def generate_simhash(text, k=3, num_perm=64):
    """
    生成文本的Simhash值。
    :param text: 要生成Simhash的文本。
    :param k: Simhash的k-gram参数。
    :param num_perm: Simhash的排列数。
    :return: 文本的Simhash值。
    """
    # 初始化特征向量
    feature_vector = [0] * num_perm

    # 生成N-gram序列
    ngram_sequence = ngrams(text, k)

    # 计算特征向量
    for ngram in ngram_sequence:
        ngram_hash = hash(' '.join(ngram))
        for i in range(num_perm):
            if ngram_hash & (1 << i):
                feature_vector[i] += 1
            else:
                feature_vector[i] -= 1

    # 构建Simhash值
    simhash = 0
    for i in range(num_perm):
        if feature_vector[i] >= 0:
            simhash |= (1 << i)

    return simhash


def calculate_similarity(texts1, texts2, threshold=0.5):
    """
    计算两个文本列表的相似度。
    :param texts1: 第一个文本列表。
    :param texts2: 第二个文本列表。
    :param threshold: 相似度阈值。
    :return: 两个文本列表的相似度。
    """
    # 生成指纹
    fingerprint1 = generate_simhash_lsh_fingerprint(texts1)
    fingerprint2 = generate_simhash_lsh_fingerprint(texts2)

    # 计算相似度
    similarity = 0.0

    for simhash1 in fingerprint1:
        similar_texts = find_similar_texts(fingerprint2, simhash1, threshold)
        for text1 in fingerprint1[simhash1]:
            for text2 in similar_texts:
                similarity += calculate_cosine_similarity_text(text1, text2)

    similarity /= len(texts1) * len(texts2)

    return similarity


def find_similar_texts(lsh_index, simhash, threshold):
    """
    根据相似度阈值在LSH索引中查询相似的文本。
    :param lsh_index: LSH索引。
    :param simhash: 目标Simhash值。
    :param threshold: 相似度阈值。
    :return: 相似的文本列表。
    """
    similar_texts = []
    for stored_simhash, texts in lsh_index.items():
        if hamming_distance(simhash, stored_simhash) <= threshold:
            similar_texts.extend(texts)
    return similar_texts


def hamming_distance(simhash1, simhash2):
    """
    计算两个Simhash值的汉明距离。
    :param simhash1: 第一个Simhash值。
    :param simhash2: 第二个Simhash值。
    :return: 汉明距离。
    """
    xor_result = simhash1 ^ simhash2
    hamming_dist = bin(xor_result).count('1')
    return hamming_dist


def calculate_cosine_similarity_text(text1, text2):
    """
    计算两个文本之间的余弦相似度。
    :param text1: 第一个文本。
    :param text2: 第二个文本。
    :return: 余弦相似度。
    """
    # 使用正确的TF-IDF计算方法
    vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return cosine_sim


def calculate_similarity_str(str1, str2):
    """
    计算两个字符串的相似度。
    :param str1: 第一个字符串。
    :param str2: 第二个字符串。
    :return: 相似度。
    """
    distance = Levenshtein.distance(str1, str2)
    max_length = max(len(str1), len(str2))
    similarity = 1 - (distance / max_length)
    return similarity


if __name__ == '__main__':
    # 示例文本列表
    str1 = '北京增值税电子普通发票.pdf'
    str2 = '福建增值税电子普通发票.pdf'
    str3 = '福建工程学院计算机学院培养方案.pdf'

    # 生成文件指纹
    texts = [str1, str2, str3]
    fingerprints = generate_simhash_lsh_fingerprint(texts)
    print("文件指纹：", fingerprints)
    # 计算相似度
    similarity1 = calculate_similarity_str(str1, str2)
    similarity2 = calculate_similarity_str(str1, str3)
    similarity3 = calculate_similarity_str(str2, str3)

    print("相似度1和2:", similarity1)
    print("相似度1和3:", similarity2)
    print("相似度2和3:", similarity3)
