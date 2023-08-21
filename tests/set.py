import hashlib

def generate_set_fingerprint(text, k=5, w=10):
    """
    使用Winnowing算法生成set类型文本的指纹（哈希值）。
    :param text: 要生成指纹的set文本。
    :param k: 移动窗口的大小。
    :param w: 选取指纹的窗口大小。
    :return: set文本的指纹（哈希值）。
    """
    # 将set转换为字符串
    text_str = ''.join(text)

    # 初始化指纹列表
    fingerprints = []

    # 创建哈希函数（可以选择MD5、SHA-1等哈希算法）
    hash_function = hashlib.md5

    # 获取文本的哈希值列表
    hash_values = [hash_function(text_str[i:i + k].encode()).hexdigest() for i in range(len(text_str) - k + 1)]

    # 选择指纹的窗口并生成指纹
    for i in range(len(hash_values) - w + 1):
        window = hash_values[i:i + w]
        fingerprint = min(window)
        fingerprints.append(fingerprint)

    # 返回指纹列表的哈希摘要
    fingerprint_digest = hash_function(''.join(fingerprints).encode()).hexdigest()

    return fingerprint_digest


def calculate_jaccard_similarity(set1, set2):
    """
    计算两个set的Jaccard相似度。
    :param set1: 第一个set文本。
    :param set2: 第二个set文本。
    :return: Jaccard相似度。
    """
    # 计算Jaccard相似度
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity


if __name__ == '__main__':
    # 示例set类型文本
    set1 = {'北京', '增值税', '电子', '普通发票', 'pdf'}
    set2 = {'福建', '增值税', '电子', '普通发票', 'pdf'}
    set3 = {'福建', '工程学院', '计算机学院', '培养方案', 'pdf'}
    # set3 = {'密码学', '期末', '课程论文', 'pdf'}

    # 生成文件指纹
    fingerprint1 = generate_set_fingerprint(set1)
    fingerprint2 = generate_set_fingerprint(set2)
    fingerprint3 = generate_set_fingerprint(set3)

    print("set1的文件指纹:", fingerprint1)
    print("set2的文件指纹:", fingerprint2)
    print("set3的文件指纹:", fingerprint3)

    # 计算相似度
    similarity1_2 = calculate_jaccard_similarity(set1, set2)
    similarity1_3 = calculate_jaccard_similarity(set1, set3)
    similarity2_3 = calculate_jaccard_similarity(set2, set3)

    print("set1和set2的Jaccard相似度:", similarity1_2)
    print("set1和set3的Jaccard相似度:", similarity1_3)
    print("set2和set3的Jaccard相似度:", similarity2_3)

