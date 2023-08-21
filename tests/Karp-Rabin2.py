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
    # 存储所有窗口的哈希值
    window_hashes = []

    # 计算窗口内的初始哈希值
    for i in range(window_size):
        hash_value = (hash_value * base + ord(text[i])) % prime
    window_hashes.append(hash_value)

    # 滑动窗口生成哈希值
    for i in range(1, len(text) - window_size + 1):
        # 移除滑动窗口前一个字符的贡献
        hash_value = (hash_value - ord(text[i - 1]) * pow(base, window_size - 1, prime)) % prime
        # 添加滑动窗口后一个字符的贡献
        hash_value = (hash_value * base + ord(text[i + window_size - 1])) % prime
        # 存储当前窗口的哈希值
        window_hashes.append(hash_value)

    # 将所有窗口的哈希值连接起来，作为整个字符串的指纹
    fingerprint = ''.join(str(h) for h in window_hashes)

    return fingerprint


def hamming_distance(hash1, hash2):
    """
    计算两个字符串之间的汉明距离。
    :param hash1: 第一个字符串的哈希值。
    :param hash2: 第二个字符串的哈希值。
    :return: 汉明距离。
    """
    distance = sum(bit1 != bit2 for bit1, bit2 in zip(hash1, hash2))
    return distance


def calculate_similarity(hash1, hash2):
    """
    计算两个字符串之间的相似度。
    :param hash1: 第一个字符串的哈希值。
    :param hash2: 第二个字符串的哈希值。
    :return: 相似度。
    """
    num_bits = len(hash1)
    distance = hamming_distance(hash1, hash2)
    similarity = 1 - (distance / num_bits)
    return similarity


if __name__ == '__main__':
    str1 = 'This is a test text for similarity calculation using Karp-Rabin algorithm'
    str2 = 'This is a test text for similarity calculation using Karp-Rabin algorithm and a little long'

    # 生成示例文本的Karp-Rabin文件指纹
    fingerprint1 = generate_karprabin_hash(str1)
    fingerprint2 = generate_karprabin_hash(str2)

    print("Karp-Rabin文件指纹示例:")
    print("str1:", fingerprint1)
    print("str2:", fingerprint2)

    # 计算相似度
    similarity = calculate_similarity(fingerprint1, fingerprint2)
    print(f"相似度: {similarity}")
