from collections import Counter

from simhash import Simhash
import hashlib


def generate_kgrams(text, k):
    """
    生成k-grams列表。
    :param text: 要生成k-grams的文本。
    :param k: k-grams的长度。
    :return: k-grams列表。
    """
    kgrams = []
    for i in range(len(text) - k + 1):
        kgram = text[i:i + k]
        kgrams.append(kgram)
    return kgrams


def calculate_hashes(kgrams):
    """
    计算k-grams的哈希值列表。
    :param kgrams: k-grams列表。
    :return: 哈希值列表(10进制)。
    """
    hashes = []
    for kgram in kgrams:
        hash_value = hashlib.sha1(kgram.encode('utf-8'))
        hash_value = hash_value.hexdigest()[-4:]
        hash_value = int(hash_value, 16)  # using last 16 bits of sha-1 digest
        # 也可以换其他hash
        hashes.append(hash_value)
    return hashes


def winnowing(hashes, window_size):
    """
    使用Winnowing算法从哈希列表中生成文本的指纹。
    :param hashes: 哈希值列表。
    :param window_size: 滑动窗口大小。
    :return: 文本的指纹集合(10进制)。
    """
    min_hashes = []
    # 找到初始窗口中的最小哈希值(若有重复选最右边)
    min_pos, min_hash = 0, hashes[0]
    for i, x in enumerate(hashes[0:window_size]):
        if x <= min_hash:
            min_pos, min_hash = i, x
    min_hashes.append(min_hash)

    # 滑动窗口，选择局部最小哈希值
    for i in range(1, len(hashes) - window_size + 1):
        if min_pos < i:
            min_pos, min_hash = i, hashes[i]
            for pos, x in enumerate(hashes[i:window_size + i]):
                if x <= min_hash:
                    min_pos, min_hash = pos + i, x
            min_hashes.append(min_hash)
        elif hashes[i + window_size - 1] <= min_hash:
            min_pos = i + window_size - 1
            min_hash = hashes[i + window_size - 1]
            min_hashes.append(min_hash)

    return min_hashes


def kgrams_winnowing(text, k, window_size):
    """
    使用k-grams+Winnowing算法生成文本的指纹。
    :param text: 要生成指纹的文本。
    :param k: k-grams的长度。
    :param window_size: 滑动窗口大小。
    :return: 文本的指纹(一个list结构，包含选出的多个hash值，10进制int表示)。
    """
    kgrams = generate_kgrams(text, k)
    hashes = calculate_hashes(kgrams)
    fingerprints = winnowing(hashes, window_size)
    return fingerprints


# def calculate_hamming_similarity(fingerprints1, fingerprints2):
#     """
#     计算两个文本指纹之间的汉明距离相似度（基于思想略做修改——原始汉明距离是算01串的差别，但是winnowing算法得到的值其实每一个小部分的十进制哈希和
#     其它算法得到的一个位的01值维度差不多，所以直接计算每个十进制相同与否就好）。
#     （其实感觉winnowing得到的hash好像不太适合用汉明距离算）
#     :param fingerprints1: 第一个文本的指纹（16进制）。
#     :param fingerprints2: 第二个文本的指纹（16进制）。
#     :return: 相似度。
#     """
#     # 填充两个指纹为一样长
#     if len(fingerprints1) > len(fingerprints2):
#         fingerprints2.extend([0] * (len(fingerprints1) - len(fingerprints2)))
#     elif len(fingerprints1) < len(fingerprints2):
#         fingerprints1.extend([0] * (len(fingerprints2) - len(fingerprints1)))
#     # 计算汉明距离
#     distance = sum(bit1 != bit2 for bit1, bit2 in zip(fingerprints1, fingerprints2))
#     similarity = 1 - (distance / len(fingerprints2))
#     return similarity


def calculate_jaccard_similarity(set1, set2):
    # 计算Jaccard相似度
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity


def multiset_jaccard_similarity(list1, list2):
    counter1 = Counter(list1)
    counter2 = Counter(list2)

    intersection = sum((counter1 & counter2).values())
    union = sum((counter1 | counter2).values())

    similarity = intersection / union if union != 0 else 0.0
    return similarity


if __name__ == '__main__':
    str1 = 'This is a test text for similarity calculation using k-grams+Winnowing algorithm'
    str2 = 'This is a  calculation using k-grams+Winnowing algorithm and a little long'
    # str1 = '北京增值税电子普通发票.pdf'
    # str2 = '福建增值税电子普通发票.pdf'
    str3 = '福建工程学院计算机学院培养方案.pdf'

    k = 2  # k值，根据实际情况设定
    window_size = 4  # 滑动窗口大小，根据实际情况设定

    fingerprint1 = kgrams_winnowing(str1, k, window_size)
    fingerprint2 = kgrams_winnowing(str2, k, window_size)
    fingerprint3 = kgrams_winnowing(str3, k, window_size)

    print("Winnowing文件指纹示例:")
    print("str1:", fingerprint1)
    print("str2:", fingerprint2)
    print("str3:", fingerprint3)

    # 计算相似度
    # similarity_12 = calculate_hamming_similarity(fingerprint1, fingerprint2)
    # similarity_13 = calculate_hamming_similarity(fingerprint1, fingerprint3)
    # similarity_23 = calculate_hamming_similarity(fingerprint2, fingerprint3)
    #
    # print(f"汉明相似度1和2: {similarity_12}")
    # print(f"汉明相似度1和3: {similarity_13}")
    # print(f"汉明相似度2和3: {similarity_23}")

    similarity2_12 = calculate_jaccard_similarity(set(fingerprint1), fingerprint2)
    similarity2_13 = calculate_jaccard_similarity(set(fingerprint1), fingerprint3)
    similarity2_23 = calculate_jaccard_similarity(set(fingerprint2), fingerprint3)

    print(f"jaccard相似度1和2: {similarity2_12}")
    print(f"jaccard相似度1和3: {similarity2_13}")
    print(f"jaccard相似度2和3: {similarity2_23}")

    similarity3_12 = multiset_jaccard_similarity(set(fingerprint1), fingerprint2)
    similarity3_13 = multiset_jaccard_similarity(set(fingerprint1), fingerprint3)
    similarity3_23 = multiset_jaccard_similarity(set(fingerprint2), fingerprint3)

    print(f"multiset jaccard相似度1和2: {similarity3_12}")
    print(f"multiset jaccard相似度1和3: {similarity3_13}")
    print(f"multiset jaccard相似度2和3: {similarity3_23}")
