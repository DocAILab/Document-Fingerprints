from simhash import Simhash


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
    :return: 哈希值列表。
    """
    hashes = []
    for kgram in kgrams:
        # 使用Simhash计算k-gram的哈希值(可以替换为其他哈希函数）
        simhash = Simhash(kgram)
        # 将Simhash值转换为十六进制数，并添加到哈希列表中
        hash_value = hex(simhash.value)
        hashes.append(hash_value)
    return hashes


def winnowing(hashes, window_size):
    """
    使用Winnowing算法从哈希列表中生成文本的指纹。
    :param hashes: 哈希值列表。
    :param window_size: 滑动窗口大小。
    :return: 文本的指纹。
    """
    min_hashes = []
    # 找到初始窗口中的最小哈希值
    min_hash = min(hashes[:window_size])
    min_hashes.append(min_hash)

    # 滑动窗口，选择局部最小哈希值
    for i in range(1, len(hashes) - window_size + 1):
        if hashes[i + window_size - 1] < min_hash:
            min_hash = hashes[i + window_size - 1]
            min_hashes.append(min_hash)

    return min_hashes


def hamming_distance(hash1, hash2):
    """
    计算两个字符串的汉明距离。
    :param hash1: 第一个字符串的哈希值。
    :param hash2: 第二个字符串的哈希值。
    :return: 汉明距离。
    """
    distance = sum(bit1 != bit2 for bit1, bit2 in zip(hash1, hash2))
    return distance


def calculate_similarity(fingerprints1, fingerprints2):
    """
    计算两个文本指纹之间的相似度。
    :param fingerprints1: 第一个文本的指纹。
    :param fingerprints2: 第二个文本的指纹。
    :return: 相似度。
    """
    num_bits = len(fingerprints1[0]) * 4  # 由于Simhash是64位的，每个十六进制数对应4位，所以要乘以4
    distance = hamming_distance(fingerprints1, fingerprints2)
    similarity = 1 - (distance / num_bits)
    return similarity


def kgrams_winnowing(text, k, window_size):
    """
    使用k-grams+Winnowing算法生成文本的指纹。
    :param text: 要生成指纹的文本。
    :param k: k-grams的长度。
    :param window_size: 滑动窗口大小。
    :return: 文本的指纹。
    """
    kgrams = generate_kgrams(text, k)
    hashes = calculate_hashes(kgrams)
    fingerprints = winnowing(hashes, window_size)
    return fingerprints


if __name__ == '__main__':
    str1 = 'This is a test text for similarity calculation using k-grams+Winnowing algorithm'
    str2 = 'This is a  calculation using k-grams+Winnowing algorithm and a little long'

    k = 5  # k值，根据实际情况设定
    window_size = 5  # 滑动窗口大小，根据实际情况设定

    fingerprints1 = kgrams_winnowing(str1, k, window_size)
    fingerprints2 = kgrams_winnowing(str2, k, window_size)

    print("Winnowing文件指纹示例:")
    print("str1:", fingerprints1)
    print("str2:", fingerprints2)

    # 计算相似度
    similarity = calculate_similarity(fingerprints1, fingerprints2)
    print(f"相似度: {similarity}")
