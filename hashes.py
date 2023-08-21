"""
生成文档指纹的哈希算法集合
"""
import hashlib
import ssdeep
import Levenshtein
from simhash import Simhash
import simhash
from nltk import ngrams
from datasketch import MinHash
import pyssdeep
import numpy as np
from flyhash import FlyHash

# Simhash
def fp_with_simhash(text, f=64, hash_func=simhash._hashfunc):
    """
    使用Simhash库生成文本指纹。
    :param text: 文本
    :param f: 指纹的二进制位数，必须为8的倍数
    :param hash_func: 哈希函数，默认为hashlib的md5
    :return: simhash指纹，可使用simhash.value查看
    """
    return Simhash(text, f, hash_func)


def fp_with_simhash2(text, k=3, num_perm=64):
    """
    使用自实现的simhash生成文本指纹。
    :param text: 要生成Simhash的文本。
    :param k: Simhash的k-gram参数。
    :param num_perm: Simhash的排列数，也即最后指纹的二进制维度
    :return: simhash指纹，可使用simhash.value查看
    """
    # 初始化特征向量
    feature_vector = [0] * num_perm

    # 生成N-gram序列
    ngram_sequence = ngrams(text, k)

    # 计算特征向量
    for ngram in ngram_sequence:
        ngram_hash = hash(' '.join(ngram))  # 使用了默认的哈希函数，可更换
        for i in range(num_perm):
            if ngram_hash & (1 << i):
                feature_vector[i] += 1
            else:
                feature_vector[i] -= 1

    # 构建Simhash值
    simhash = 0
    for i in range(num_perm):
        if feature_vector[i] >= 0:
            simhash |= (1 << i)

    return simhash


# MinHash
def fp_with_minhash(data, num_perm=128):
    """
    使用MinHash生成集合数据指纹，并可结合jaccard_similarity函数估计计算jaccard值。
    :param data: 集合数据，set
    :param num_perm: minhash维度，影响其精度，默认为128
    :return: minhash指纹，可使用minhash.digest()查看
    """
    minhash = MinHash(num_perm=num_perm)
    for item in data:
        minhash.update(str(item).encode('utf8'))
    return minhash


# Karp-Rabin
def fp_with_karprabin(text, window_size=5, base=256, prime=101):
    """
    使用Karp-Rabin算法生成文本的哈希指纹
    :param text: 要生成指纹的文本
    :param window_size: 滑动窗口的大小
    :param base: 基数，用于计算哈希值
    :param prime: 大素数，用于计算哈希值
    :return: 文本的指纹。str
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


# winnowing
def fp_with_winnowing(text, k=5, w=5):
    """
    使用Winnowing算法生成文本的哈希指纹。
    :param text: 要生成指纹的文本
    :param k: 移动窗口的大小
    :param w: 选取指纹的窗口大小
    :return: 文本的指纹（哈希值）, list
    """
    # 生成k-grams列表
    kgrams = []
    for i in range(len(text) - k + 1):
        kgrams.append(text[i:i + k])

    # 计算k-grams的哈希值列表
    hashes = []
    for kgram in kgrams:
        hashes.append(hashlib.md5(kgram.encode()).hexdigest())  # 可使用其他哈希函数

    # 使用Winnowing算法从哈希列表中生成文本的指纹
    min_hashes = []

    ## 找到初始窗口中的最小哈希值
    min_hash = min(hashes[:w])
    min_hashes.append(min_hash)

    ## 滑动窗口，选择局部最小哈希值
    for i in range(1, len(hashes) - w + 1):
        if hashes[i + w - 1] < min_hash:
            min_hash = hashes[i + w - 1]
            min_hashes.append(min_hash)

    return min_hashes


# Fuzzy hashes
def fp_with_fuzzyhash(data, is_path=False):
    """
    使用模糊哈希计算文本/文档指纹。
    ssdeep 是一个用来计算context triggered piecewise hashes(CTPH) 基于文本的分片哈希算法 ，同样也可以叫做模糊哈希 Fuzzy hashes。
    CTPH可以匹配同源文档（相似文档），这样的文档可能有一些顺序相同的字节，尽管这些字节可能在一个序列中长度和内容都不尽相同。
    SSDeep 对文本长度有要求，如果要生成有意义的结果，最好文本长度不小于 4096
    :param data: 文本或文件路径（pdf、txt），str
    :param is_path: 是否是文件路径
    :return: 文档指纹，str
    """
    if is_path:
        return pyssdeep.get_hash_file(data)
    return pyssdeep.get_hash_buffer(data)


# flyhash
def fp_with_flyhash(data, input_dim, hash_dim):
    """
    FlyHash 是一种 LSH 算法，它将输入数据映射到稀疏哈希嵌入， 其中哈希嵌入的维度远大于输入， 并将输入数据的位置保留在哈希嵌入中。
    FlyHash被设计为计算成本低廉，但内存效率低。它适用于散列中小型数据 （~10-1000）到大哈希嵌入（~100-10000）
    :param data: 输入数据, array
    :param input_dim: 输入数据的维度, int
    :param hash_dim: 输出数据的维度（大于input_dim），int
    :return: 稀疏哈希嵌入结果，array
    """
    # flyhash的工作时以维度为单位对数据进行随机，与文档信息或许不是很配合
    flyHash = FlyHash(input_dim, hash_dim)
    return flyHash(data)
