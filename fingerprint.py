"""
整合哈希指纹生成和相似度计算/距离计算方法。
"""
from hashes import *
from similarities import *

# simhash
class FP_SimHash:
    def __init__(self, f=64, hash_func=simhash._hashfunc):
        """
        :param f: 指纹的二进制位数，必须为8的倍数
        :param hash_func: 哈希函数，默认为hashlib的md5
        """
        self.f = f
        self.hash_func = hash_func

    def __call__(self, text):
        """
        计算指纹
        :param text: 文本值，str
        :return: simhash指纹，可使用simhash.value查看
        """
        return fp_with_simhash(text, self.f, self.hash_func)

    def compare(self, simhash1, simhash2):
        """
        计算汉明距离和相似度值
        :param simhash1: 第一个文本指纹
        :param simhash2: 第二个文本指纹
        :return: 汉明距离, int, 相似度, float
        """
        return hamming_distance(simhash1, simhash2, cal_simi=True)


# minhash
class FP_MinHash:
    def __init__(self, num_perm=128):
        """
        :param num_perm: minhash维度，影响其精度，默认为128
        """
        self.num_perm = num_perm

    def __call__(self, data):
        """
        计算指纹
        :param data: 集合数据，set
        :return: minhash指纹，可使用minhash.digest()查看
        """
        return fp_with_minhash(data, self.num_perm)

    def compare(self, minhash1, minhash2):
        """
        使用minhash估计的jaccard值
        :param minhash1: 第一个集合指纹
        :param minhash2: 第二个集合指纹
        :return: jaccard相似度，float
        """
        return jaccard_similarity(minhash1, minhash2)


# karp-rabin
class FP_KarpRabin:
    def __init__(self, window_size=5, base=256, prime=101):
        """
        :param window_size: 滑动窗口的大小
        :param base: 基数，用于计算哈希值
        :param prime: 大素数，用于计算哈希值
        """
        self.window_size = window_size
        self.base = base
        self.prime = prime

    def __call__(self, text):
        """
        计算指纹
        :param text: 文本值，str
        :return: 指纹，str
        """
        return fp_with_karprabin(text, self.window_size, self.base, self.prime)

    def compare(self, karprabin1, karprabin2):
        """
        使用汉明距离计算相似度
        :param karprabin1: 指纹1，str
        :param karprabin2: 指纹2，str
        :return: 汉明距离, int, 相似度, float
        """
        return hamming_distance(karprabin1, karprabin2, cal_simi=True)


# winnowing
class FP_Winnowing:
    def __init__(self, k=5, w=5):
        """
        :param k: 移动窗口的大小
        :param w: 选取指纹的窗口大小
        """
        self.k = k
        self.w = w

    def __call__(self, text):
        """
        计算指纹
        :param text: 文本值，str
        :return: 文本的指纹（哈希值）, list
        """
        return fp_with_winnowing(text, self.k, self.w)

    def compare(self, winnowing1, winnowing2):
        # 暂无相似度/距离计算方法
        pass


# fuzzy hash
class FP_FuzzyHash:
    def __init__(self):
        pass

    def __call__(self, data, is_path=False):
        """
        计算指纹
        :param data: 文本或文件路径（pdf、txt），str
        :param is_path: 是否是文件路径
        :return: 文档指纹，str
        """
        return fp_with_fuzzyhash(data, is_path)

    def compare(self, fuzzy1, fuzzy2):
        """
        计算指纹距离
        :param fuzzy1: 指纹1
        :param fuzzy2: 指纹2
        :return: 指纹距离，float
        """
        return pyssdeep.compare(fuzzy1, fuzzy2)


# flyhash
class FP_FlyHash:
    def __init__(self, input_dim, hash_dim):
        """
        :param input_dim: 输入数据的维度, int
        :param hash_dim: 输出数据的维度（大于input_dim），int
        """
        self.input_dim = input_dim
        self.hash_dim = hash_dim

    def __call__(self, data):
        """
        计算指纹
        :param data: 输入数据, array
        :return: 稀疏哈希嵌入结果，array
        """
        return fp_with_flyhash(data, self.input_dim, self.hash_dim)

    def compare(self, fly1, fly2):
        # 暂无相似度/距离计算方法
        pass