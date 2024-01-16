from simhash import Simhash, SimhashIndex


def generate_simhash_fingerprint(text):
    simhash = Simhash(text)  # 默认指纹维度64，hash_func默认md5
    # 返回一个十六进制hash指纹
    # return hex(simhash.value)
    # 返回一个Simhash对象（便于之后使用内置函数求语义距离）
    return simhash


def calculate_similarity(simhash1, simhash2):
    distance = simhash1.distance(simhash2)
    similarity = 1 - (distance / 64)  # 假设Simhash的num_perm为64
    return similarity


def simhashLSH_search(objs, simhash, k=2):
    '''
    :param objs: (obj_name, simhash)的列表
    :param simhash: 待查询的simhash
    :param k: k为LSH算法里划分band的分片数，越大越精确，但是越慢；
    同时k也代表了对于相似度的容忍度，k越大，更不相似的文本更有可能放于同一个bucket中可以容忍更不相似的结果被查询出来
    :return:
    '''
    index = SimhashIndex(objs, k=k)
    print(index.bucket_size())
    return index.get_near_dups(simhash)


if __name__ == '__main__':
    str1 = '北京增值税电子普通发票.pdf'
    str2 = '福建增值税电子普通发票.pdf'
    str3 = '福建工程学院计算机学院培养方案.pdf'

    # 生成文件指纹并打印
    fingerprint1 = generate_simhash_fingerprint(str1)
    fingerprint2 = generate_simhash_fingerprint(str2)
    fingerprint3 = generate_simhash_fingerprint(str3)
    print(f"文件指纹1: {bin(fingerprint1.value)}")  # 不准确的二进制形式，可能会缺少前导0
    print(f"文件指纹2: {bin(fingerprint2.value)}")
    print(f"文件指纹3: {bin(fingerprint3.value)}")

    # 计算相似度
    similarity_12 = calculate_similarity(fingerprint1, fingerprint2)
    similarity_13 = calculate_similarity(fingerprint1, fingerprint3)
    similarity_23 = calculate_similarity(fingerprint2, fingerprint3)

    print(f"相似度1和2: {similarity_12}")
    print(f"相似度1和3: {similarity_13}")
    print(f"相似度2和3: {similarity_23}")

    # 使用SimhashIndex查询相似项（这一步可以在大规模数据中进行快速相似项查询）(实际上是一种LSH方法的实现)
    objs = [("str1", fingerprint1), ("str2", fingerprint2), ("str3", fingerprint3)]
    fingerprint4 = generate_simhash_fingerprint("北京电子普通发票.pdf")
    similar_doc = simhashLSH_search(objs, fingerprint4, k=16)
    print(similar_doc)
