import hashlib


def generate_karprabin_hash(text, window_size=5, base=256, prime=101):
    """
    使用Karp-Rabin算法生成文本的指纹（哈希值）。
    :param text: 要生成指纹的文本。
    :param window_size: 滑动窗口的大小。
    :param base: 基数，用于计算哈希值。
    :param prime: 大素数，用于计算哈希值。
    :return: 文本的指纹（哈希值）。
    """
    # 初始化哈希值
    hash_value = 0
    # 计算窗口内的初始哈希值
    for i in range(window_size):
        hash_value = (hash_value * base + ord(text[i])) % prime

    # 存储哈希值的摘要（可以选择MD5、SHA-1等哈希算法）
    hash_digest = hashlib.md5(str(hash_value).encode()).hexdigest()

    # 滑动窗口生成哈希值
    for i in range(1, len(text) - window_size + 1):
        # 移除滑动窗口前一个字符的贡献
        hash_value = (hash_value - ord(text[i - 1]) * pow(base, window_size - 1, prime)) % prime
        # 添加滑动窗口后一个字符的贡献
        hash_value = (hash_value * base + ord(text[i + window_size - 1])) % prime

        # 更新摘要
        hash_digest = hashlib.md5(str(hash_value).encode()).hexdigest()

    return hash_digest


def hamming_distance(hash1, hash2):
    """
    计算两个二进制哈希值之间的汉明距离。
    :param hash1: 第一个二进制哈希值。
    :param hash2: 第二个二进制哈希值。
    :return: 汉明距离。
    """
    distance = sum(bit1 != bit2 for bit1, bit2 in zip(hash1, hash2))
    return distance


def calculate_similarity(hash1, hash2):
    """
    计算两个文本指纹之间的相似度。
    :param hash1: 第一个文本指纹的二进制哈希值。
    :param hash2: 第二个文本指纹的二进制哈希值。
    :return: 相似度。
    """
    num_bits = len(hash1)
    distance = hamming_distance(hash1, hash2)
    similarity = 1 - (distance / num_bits)
    return similarity


if __name__ == '__main__':
    str1 = '北京增值税电子普通发票.pdf'
    str2 = '福建增值税电子普通发票.pdf'
    str3 = '福建工程学院计算机学院培养方案.pdf'
    # 生成示例文本的Karp-Rabin文件指纹
    fingerprint1 = generate_karprabin_hash(str1)
    fingerprint2 = generate_karprabin_hash(str2)
    fingerprint3 = generate_karprabin_hash(str3)

    print("Karp-Rabin文件指纹示例:")
    print("str1:", fingerprint1)
    print("str2:", fingerprint2)
    print("str3:", fingerprint3)

    # 将Karp-Rabin文件指纹转换为二进制形式的哈希值
    binary_hash1 = bin(int(fingerprint1, 16))[2:]
    binary_hash2 = bin(int(fingerprint2, 16))[2:]
    binary_hash3 = bin(int(fingerprint3, 16))[2:]

    # 计算相似度
    similarity_12 = calculate_similarity(binary_hash1, binary_hash2)
    similarity_13 = calculate_similarity(binary_hash1, binary_hash3)
    similarity_23 = calculate_similarity(binary_hash2, binary_hash3)

    print(f"相似度1和2: {similarity_12}")
    print(f"相似度1和3: {similarity_13}")
    print(f"相似度2和3: {similarity_23}")
