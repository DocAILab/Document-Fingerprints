from simhash import Simhash, SimhashIndex


def generate_simhash_fingerprint(text):
    """
    使用Simhash算法生成文本的指纹。
    :param text: 要生成指纹的文本。
    :return: 文本的指纹（Simhash对象）。
    """
    simhash = Simhash(text)
    return simhash


def calculate_similarity(simhash1, simhash2):
    """
    计算两个Simhash指纹之间的相似度。
    :param simhash1: 第一个Simhash指纹。
    :param simhash2: 第二个Simhash指纹。
    :return: 相似度。
    """
    similarity = simhash1.distance(simhash2)
    return 1 - (similarity / 64)  # 64为Simhash指纹的位数


if __name__ == '__main__':
    str1 = 'This is a test text for similarity calculation using Simhash and LSH algorithm'
    str2 = 'This is a test text for similarity calculation using Simhash and LSH algorithm and a little long'
    str3 = 'This is a completely different text'

    # 生成示例文本的Simhash指纹
    simhash1 = generate_simhash_fingerprint(str1)
    simhash2 = generate_simhash_fingerprint(str2)
    simhash3 = generate_simhash_fingerprint(str3)

    # 创建SimhashIndex对象，用于存储指纹和检索相似指纹
    index = SimhashIndex([(str(i), simhash) for i, simhash in enumerate([simhash1, simhash2])], k=3)

    # 查询与str3相似的指纹
    similar_simhashes = index.get_near_dups(simhash3)

    # 计算相似度
    similarity1 = calculate_similarity(simhash1, simhash2)
    similarity2 = calculate_similarity(simhash1, simhash3)
    similarity3 = calculate_similarity(simhash2, simhash3)

    print("相似度示例:")
    print("str1和str2的相似度:", similarity1)
    print("str1和str3的相似度:", similarity2)
    print("str2和str3的相似度:", similarity3)
    print("与str3相似的指纹:", similar_simhashes)
