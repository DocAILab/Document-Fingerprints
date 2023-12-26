import hashlib
from nltk import ngrams
from datasketch import MinHash, MinHashLSH


def generate_ngram_minhash_fingerprint(text, n=3, num_perm=128, threshold=0.5):
    """
    使用N-gram + LSH算法生成文本的指纹（哈希值）。
    :param text: 要生成指纹的文本。
    :param n: N-gram的大小。
    :param num_perm: MinHash的排列数。
    :param threshold: LSH的阈值。
    :return: 文本的指纹（哈希值）。
    """
    # 创建MinHash对象
    minhash = MinHash(num_perm=num_perm)

    # 创建哈希函数（可以选择MD5、SHA-1等哈希算法）
    hash_function = hashlib.md5

    # 生成N-gram序列
    ngram_sequence = ngrams(text, n)

    # 添加N-gram序列到MinHash中
    for ngram in ngram_sequence:
        for item in ngram:
            minhash.update(hash_function(item.encode()).digest())

    # 返回指纹的哈希摘要
    fingerprint_digest = minhash.digest()

    return fingerprint_digest


def hash_fingerprint(fingerprint):
    """
    使用哈希函数对指纹进行进一步的处理。
    :param fingerprint: 要处理的指纹。
    :return: 处理后的指纹（哈希值）。
    """
    # 创建哈希函数（可以选择MD5、SHA-1、SHA-256等哈希算法）
    hash_function = hashlib.md5

    # 计算指纹的哈希值
    hashed_fingerprint = hash_function(fingerprint).hexdigest()

    return hashed_fingerprint


def hamming_distance(fingerprint1, fingerprint2):
    """
    计算两个指纹之间的汉明距离。
    :param fingerprint1: 第一个指纹。
    :param fingerprint2: 第二个指纹。
    :return: 汉明距离。
    """
    assert len(fingerprint1) == len(fingerprint2), "指纹长度不一致"

    distance = sum(c1 != c2 for c1, c2 in zip(fingerprint1, fingerprint2))
    return distance


def hamming_similarity(fingerprint1, fingerprint2):
    """
    计算两个指纹之间的相似度。
    :param fingerprint1: 第一个指纹。
    :param fingerprint2: 第二个指纹。
    :return: 相似度。
    """
    hamming_dist = hamming_distance(fingerprint1, fingerprint2)
    similarity = 1 - (hamming_dist / len(fingerprint1))
    return similarity

# 定义一个函数，用于计算两个文档指纹的Jaccard相似度
def jaccard_similarity(minhash1, minhash2):
    return minhash1.jaccard(minhash2)

if __name__ == '__main__':
    str1 = '北京增值税电子普通发票.pdf'
    str2 = '福建增值税电子普通发票.pdf'
    str3 = '福建工程学院计算机学院培养方案.pdf'
    # 生成示例文本的N-gram + minhash文件指纹
    fingerprint1 = generate_ngram_minhash_fingerprint(str1)
    fingerprint2 = generate_ngram_minhash_fingerprint(str2)
    fingerprint3 = generate_ngram_minhash_fingerprint(str3)

    print("N-gram + minhash文件指纹示例:")
    print("str1:", fingerprint1)
    print("str2:", fingerprint2)
    print("str3:", fingerprint3)

    # 使用MD5哈希函数对指纹进行进一步处理
    hashed_fingerprint1 = hash_fingerprint(fingerprint1)
    hashed_fingerprint2 = hash_fingerprint(fingerprint2)
    hashed_fingerprint3 = hash_fingerprint(fingerprint3)

    print("处理后的指纹（哈希值）:")
    print("str1:", hashed_fingerprint1)
    print("str2:", hashed_fingerprint2)
    print("str3:", hashed_fingerprint3)

    # 计算相似度
    similarity1 = hamming_similarity(fingerprint1, fingerprint2)
    similarity2 = hamming_similarity(fingerprint1, fingerprint3)
    similarity3 = hamming_similarity(fingerprint2, fingerprint3)
    print("1和2两个文本的hamming相似度为:", similarity1)
    print("1和3两个文本的hamming相似度为:", similarity2)
    print("2和3两个文本的hamming相似度为:", similarity3)

    similarity1 = jaccard_similarity(fingerprint1, fingerprint2)
    similarity2 = jaccard_similarity(fingerprint1, fingerprint3)
    similarity3 = jaccard_similarity(fingerprint2, fingerprint3)
    print("1和2两个文本的hamming相似度为:", similarity1)
    print("1和3两个文本的hamming相似度为:", similarity2)
    print("2和3两个文本的hamming相似度为:", similarity3)