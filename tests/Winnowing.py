import hashlib

from Hash_algorithm.example import str1, str2, str3


def generate_winnowing_fingerprint(text, k=5, w=10):
    """
    使用Winnowing算法生成文本的指纹（哈希值）。
    :param text: 要生成指纹的文本。
    :param k: 移动窗口的大小。
    :param w: 选取指纹的窗口大小。
    :return: 文本的指纹（哈希值）。
    """
    # 初始化指纹列表
    fingerprints = []

    # 创建哈希函数（可以选择MD5、SHA-1等哈希算法）
    hash_function = hashlib.md5

    # 获取文本的哈希值列表
    hash_values = [hash_function(text[i:i + k].encode()).hexdigest() for i in range(len(text) - k + 1)]

    # 选择指纹的窗口并生成指纹
    for i in range(len(hash_values) - w + 1):
        window = hash_values[i:i + w]
        fingerprint = min(window)
        fingerprints.append(fingerprint)

    # 返回指纹列表的哈希摘要
    fingerprint_digest = hash_function(''.join(fingerprints).encode()).hexdigest()

    return fingerprint_digest


def calculate_jaccard_similarity(set1, set2):
    # 计算Jaccard相似度
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity


if __name__ == '__main__':
    str1 = '北京增值税电子普通发票.pdf'
    str2 = '福建增值税电子普通发票.pdf'
    str3 = '福建工程学院计算机学院培养方案.pdf'
    # 生成示例文本的Winnowing文件指纹
    fingerprint1 = generate_winnowing_fingerprint(str1)
    fingerprint2 = generate_winnowing_fingerprint(str2)
    fingerprint3 = generate_winnowing_fingerprint(str3)

    print("Winnowing文件指纹示例:")
    print("str1:", fingerprint1)
    print("str2:", fingerprint2)
    print("str3:", fingerprint3)

    # 将指纹转换为集合
    set1 = set(fingerprint1)
    set2 = set(fingerprint2)
    set3 = set(fingerprint3)

    # 计算相似度
    similarity_12 = calculate_jaccard_similarity(set1, set2)
    similarity_13 = calculate_jaccard_similarity(set1, set3)
    similarity_23 = calculate_jaccard_similarity(set2, set3)

    print(f"相似度1和2: {similarity_12}")
    print(f"相似度1和3: {similarity_13}")
    print(f"相似度2和3: {similarity_23}")