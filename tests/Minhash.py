from datasketch import MinHash, MinHashLSH


# 定义一个函数，用于计算Jaccard相似度
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


# 定义函数，用于生成MinHash
def generate_minhash(data, num_perm):
    minhash = MinHash(num_perm=num_perm)
    for item in data:
        minhash.update(item.encode('utf8'))
    return minhash


if __name__ == '__main__':
    # 定义两个示例的set类型数据
    set1 = {'apple', 'banana', 'orange', 'grape'}
    set2 = {'apple', 'watermelon', 'banana', 'kiwi'}

    # 使用MinHash生成文件指纹
    num_perm = 128  # MinHash的num_perm参数，影响MinHash的精度
    minhash1 = generate_minhash(set1, num_perm)
    minhash2 = generate_minhash(set2, num_perm)

    # 打印哈希值
    print(f"文件指纹1: {minhash1}")
    print(f"文件指纹2: {minhash2}")
    # 查看Minhash的值（128维向量）
    # print(minhash1.digest())
    # print(minhash2.digest())

    # 使用Jaccard相似度计算相似度
    similarity = jaccard_similarity(set1, set2)
    print("Jaccard相似度：", similarity)

    # 使用MinHashLSH查询相似项（这一步可以在大规模数据中进行快速相似项查询）
    lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)
    lsh.insert("set1", minhash1)
    lsh.insert("set2", minhash2)

    result = lsh.query(minhash1)
    print("MinHashLSH查询相似项：", result)
