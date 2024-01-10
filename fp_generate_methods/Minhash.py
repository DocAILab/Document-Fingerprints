from datasketch import MinHash, MinHashLSH


def ngrams(text, n=3):
    """
    生成N-gram序列。
    :param text: 要生成N-gram序列的文本。
    :param n: N-gram的大小。
    :return: N-gram序列。
    """
    ngrams = []
    for i in range(len(text) - n + 1):
        ngrams.append(text[i:i + n])
    return set(ngrams)


# 定义函数，用于生成MinHash
def generate_minhash(data, num_perm):
    minhash = MinHash(num_perm=num_perm)
    for item in data:
        # MinHash类的逻辑是提前确定好128个hash映射关系，每新增一个元素，重新算一遍该元素的128种映射结果，再和之前的128个最小值结果去比较，如果小于，则更新该位置的值，否则不更新
        minhash.update(item.encode('utf8'))
    return minhash


# 定义一个函数，用于计算两个文档指纹的Jaccard相似度
def jaccard_similarity(minhash1, minhash2):
    return minhash1.jaccard(minhash2)


# 直接对两个set类型数据计算Jaccard相似度
def jaccard_similarity2(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


# 一个用于在大规模数据中进行快速相似项查询的方法
def minhashLSH_search(minhash_sets, query_set, threshold=0.5, num_perm=128):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)  # threshhold是判定是否相似的jaccard阈值（初始化的 MinHash LSH 将通过最小化误报和漏报来针对阈值进行优化。）
    for name, set in minhash_sets.items():
        lsh.insert(name, set)
    return lsh.query(query_set)


if __name__ == '__main__':
    # 定义两个示例的set类型数据
    set1 = {'apple', 'banana', 'orange', 'grape'}
    set2 = {'apple', 'watermelon', 'banana', 'kiwi'}
    str3 = '北京增值税电子普通发票.pdf'
    str4 = '福建增值税电子普通发票.pdf'
    str5 = '福建工程学院计算机学院培养方案.pdf'

    # 使用MinHash生成文件指纹
    num_perm = 128  # MinHash的num_perm参数，影响MinHash的精度
    minhash1 = generate_minhash(set1, num_perm)
    minhash2 = generate_minhash(set2, num_perm)
    minhash3 = generate_minhash(ngrams(str3, 1), num_perm)
    minhash4 = generate_minhash(ngrams(str4, 1), num_perm)
    minhash5 = generate_minhash(ngrams(str5, 1), num_perm)

    # 打印哈希值
    print(f"文件指纹1: {minhash1.digest()}")
    print(f"文件指纹2: {minhash2.digest()}")
    print(f"文件指纹3: {minhash3.digest()}")
    print(f"文件指纹4: {minhash4.digest()}")
    print(f"文件指纹5: {minhash5.digest()}")

    # 使用Jaccard相似度计算相似度
    similarity = jaccard_similarity(minhash1, minhash2)
    print("文档指纹Jaccard相似度：", similarity)
    similarity2 = jaccard_similarity2(set1, set2)
    print("文档集合Jaccard相似度：", similarity2)
    similarity34 = jaccard_similarity(minhash3, minhash4)
    print("3和4两个文本的文档指纹Jaccard相似度：", similarity34)
    similarity35 = jaccard_similarity(minhash3, minhash5)
    print("3和5两个文本的文档指纹Jaccard相似度：", similarity35)
    similarity45 = jaccard_similarity(minhash4, minhash5)
    print("4和5两个文本的文档指纹Jaccard相似度：", similarity45)

    # 使用MinHashLSH查询相似项（这一步可以在大规模数据中进行快速相似项查询）
    sets = {"set1": minhash1, "set2": minhash2, "set3": minhash3, "set5": minhash5}
    result = minhashLSH_search(sets, minhash4)
    print("MinHashLSH查询相似项：", result)
