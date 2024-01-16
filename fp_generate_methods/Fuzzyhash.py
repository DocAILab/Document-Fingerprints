import pyssdeep
import Levenshtein

def fp_with_fuzzyhash(data, is_path=False):
    """
    使用模糊哈希计算文本/文档指纹。
    ssdeep 是一个用来计算context triggered piecewise hashes(CTPH) 基于文本的分片哈希算法 ，同样也可以叫做模糊哈希 Fuzzy hashes。
    CTPH可以匹配同源文档（相似文档），这样的文档可能有一些顺序相同的字节，尽管这些字节可能在一个序列中长度和内容都不尽相同。
    SSDeep 对文本长度有要求，如果要生成有意义的结果，最好文本长度不小于 4096
    :param data: 文本或文件路径（pdf、txt），str; 或者具体的文本内容
    :param is_path: 是否是文件路径
    :return: 文档指纹，str(ascii码串的形式)
    """
    if is_path:
        return pyssdeep.get_hash_file(data)
    return pyssdeep.get_hash_buffer(data)

# 指纹相似度比较方法自实现
def levenshtein_distance(text1, text2, cal_simi=True):
    """
    计算两个文本之间的编辑距离和相似度。
    :param text1: 第一个文本
    :param text2: 第二个文本
    :param cal_simi: 是否计算相似度
    :return: 编辑距离, [相似度]
    """
    dist = Levenshtein.distance(text1, text2)
    if cal_simi:
        max_length = max(len(text1), len(text2))
        simi = 1 - (dist / max_length)
        return dist, simi
    return dist

def jaccard_similarity(data1, data2):
    """
    计算jaccard相似度。
    :param data1: 第一个数据，set/list/MinHash
    :param data2: 第二个数据，set/list/MinHash
    :return: jaccard值，float
    """
    data1 = set(list(data1))
    data2 = set(list(data2))
    intersection = len(data1.intersection(data2))
    union = len(data1.union(data2))
    return intersection / union


if __name__ == "__main__":
    str1 = '北京增值税电子普通发票.pdf'
    str2 = '福建增值税电子普通发票.pdf'
    str3 = '福建工程学院计算机学院培养方案.pdf'

    # 生成文件指纹并打印
    fingerprint1 = fp_with_fuzzyhash(str1)
    fingerprint2 = fp_with_fuzzyhash(str2)
    fingerprint3 = fp_with_fuzzyhash(str3)
    print(f"文件指纹1: {fingerprint1}")
    print(f"文件指纹2: {fingerprint2}")
    print(f"文件指纹3: {fingerprint3}")

    similarity_12 = pyssdeep.compare(fingerprint1, fingerprint2)
    similarity_13 = pyssdeep.compare(fingerprint1, fingerprint3)
    similarity_23 = pyssdeep.compare(fingerprint2, fingerprint3)

    print(f"相似度1和2: {similarity_12}")
    print(f"相似度1和3: {similarity_13}")
    print(f"相似度2和3: {similarity_23}")

    # 计算相似度2
    similarity2_12 = levenshtein_distance(fingerprint1, fingerprint2)
    similarity2_13 = levenshtein_distance(fingerprint1, fingerprint3)
    similarity2_23 = levenshtein_distance(fingerprint2, fingerprint3)

    print(f"自实现编辑距离相似度1和2: {similarity2_12}")
    print(f"自实现编辑距离相似度1和3: {similarity2_13}")
    print(f"自实现编辑距离相似度2和3: {similarity2_23}")

    # 计算相似度3
    similarity3_12 = jaccard_similarity(fingerprint1, fingerprint2)
    similarity3_13 = jaccard_similarity(fingerprint1, fingerprint3)
    similarity3_23 = jaccard_similarity(fingerprint2, fingerprint3)

    print(f"自实现jaccard相似度1和2: {similarity3_12}")
    print(f"自实现jaccard相似度1和3: {similarity3_13}")
    print(f"自实现jaccard相似度2和3: {similarity3_23}")