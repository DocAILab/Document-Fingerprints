"""
生成文档指纹的哈希算法集合
"""
import hashlib
import Levenshtein
import jieba
from simhash import Simhash
from nltk import ngrams
from datasketch import MinHash
import pyssdeep
import numpy as np
from flyhash import FlyHash
from sklearn.feature_extraction.text import TfidfVectorizer


# Simhash
def fp_with_simhash(text, f=64, hash_func=None):  # TODO：探究分词对simhash的影响
    """
    使用Simhash库生成文本指纹。
    :param text: 文本
    :param f: 指纹的二进制位数，必须为8的倍数, 默认为64
    :param hash_func: 哈希函数，默认为hashlib的md5
    :return: simhash指纹，01字符串的形式
    """
    # 替换hash_func
    # hash_func = hashlib.md5
    if hash_func is not None:
        simhash = Simhash(text, f, hash_func)
    else:
        simhash = Simhash(text, f)
    simhash_str = bin(simhash.value)[2:]
    if len(simhash_str) < f:  # 补前导0
        simhash_str = '0' * (f - len(simhash_str)) + simhash_str
    return simhash_str


def fp_with_simhash2(text, k=3, num_perm=64):
    """
    使用自实现的simhash生成文本指纹。
    :param text: 要生成Simhash的文本。
    :param k: Simhash的k-gram参数。
    :param num_perm: Simhash的排列数，也即最后指纹的二进制维度
    :return: simhash指纹，01字符串的形式
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
def fp_with_minhash(data, num_perm=128, n=3):
    """
    使用MinHash生成集合数据指纹，并可结合jaccard_similarity函数估计计算jaccard值。
    :param data: 要生是字符串成指纹的数据，可以，也可以是集合
    :param num_perm: minhash维度，影响其精度，默认为128
    :param n: n-gram参数，默认为3
    :return: minhash指纹，一个长度为num_perm的list
    """
    if isinstance(data, str):
        ngrams = []
        for i in range(len(data) - n + 1):
            ngrams.append(data[i:i + n])
        data = set(ngrams)
    minhash = MinHash(num_perm=num_perm)
    for item in data:
        minhash.update(str(item).encode('utf8'))
    return minhash.digest().tolist()


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
def fp_with_winnowing(text, n=5, w=5, hash_func=None):
    """
    使用Winnowing算法生成文本的哈希指纹。
    :param text: 要生成指纹的文本
    :param n: 截取文本的n-grams大小
    :param w: 选取指纹的窗口大小
    :param hash_func: 计算每个文本片段的哈希函数，默认为hashlib的sha1
    :return: 文本的指纹（哈希值）集合, list
    """
    # 生成n-grams列表
    ngrams = []
    for i in range(len(text) - n + 1):
        ngrams.append(text[i:i + n])

    # 计算n-grams的哈希值列表
    hashes = []
    for ngram in ngrams:
        if hash_func is None:
            hash_value = hashlib.sha1(ngram.encode('utf-8'))
            hash_value = hash_value.hexdigest()[-4:]
            hash_value = int(hash_value, 16)  # using last 16 bits of sha-1 digest
        else:  # 也可以换其他hash
            hash_value = hash_func(ngram)
        hashes.append(hash_value)
    # 使用Winnowing算法从哈希列表中生成文本的指纹
    min_hashes = []

    # 找到初始窗口中的最小哈希值(若有重复选最右边)
    min_pos, min_hash = 0, hashes[0]

    for i, x in enumerate(hashes[0:w]):
        if x <= min_hash:
            min_pos, min_hash = i, x
    min_hashes.append(min_hash)

    # 滑动窗口，选择局部最小哈希值
    for i in range(1, len(hashes) - w + 1):
        if min_pos < i:
            min_pos, min_hash = i, hashes[i]
            for pos, x in enumerate(hashes[i:w + i]):
                if x <= min_hash:
                    min_pos, min_hash = pos + i, x
            min_hashes.append(min_hash)
        elif hashes[i + w - 1] <= min_hash:
            min_pos = i + w - 1
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
    :return: 文档指纹，str(ascii码串的形式，每一位ascii码代表一个分片的结果)
    """
    if is_path:
        return pyssdeep.get_hash_file(data)
    return pyssdeep.get_hash_buffer(data)


# flyhash
def fp_with_flyhash(data, hash_dim, k=20, density=0.1, sparsity=0.05, tokenizer=None):
    """
        FlyHash 是一种 LSH 算法，它将输入数据映射到稀疏哈希嵌入， 其中哈希嵌入的维度远大于输入， 并将输入数据的位置保留在哈希嵌入中。
        FlyHash被设计为计算成本低廉，但内存效率低。它适用于散列中小型数据 （~10-1000）到大哈希嵌入（~100-10000）
        此函数采用tfidf把文本转化为向量，然后使用flyhash进行哈希
        :param data: 输入数据, 一个列表，可以一维或二维
        :param hash_dim: 输出数据的维度，int
        :param k: 把hash映射到高维的比例，int。需要保证hash_dim*k>tfidf_matrix的维度，论文中设置为20
        :param density: 决定投影矩阵的疏密。如果“density”是浮点数，则投影矩阵的每一列都有概率为“density”的非零条目;如果“density”是整数，则投影矩阵的每一列都恰好具有“density”非零条目。（论文采用0.1）
        :param sparsity: 决定hash结果的疏密（论文采用0.05）(理论上k*sparsity==1比较好)
        :param tokenizer: 分词函数，如果是中文需要自定义分词函数，否则使用默认的分词函数
        :return: 稀疏哈希嵌入结果，01字符串
    """

    def chinese_tokenizer(text):
        """
        中文分词函数，使用jieba分词
        :param text: 文本
        :return: 分词结果
        """
        return jieba.lcut(text)

    if tokenizer == "chinese":
        tokenizer = chinese_tokenizer
    # TF-IDF矩阵的每一行对应于一个文档，而每一列对应于一个单词
    vectorizer = TfidfVectorizer(use_idf=True, norm='l2', tokenizer=tokenizer)
    # 注意：如果data是中文需要另外进行分词处理！
    tfidf_matrix = vectorizer.fit_transform(data)
    flyHash = FlyHash(tfidf_matrix.shape[1], hash_dim * k, density, sparsity, seed=55)
    raw_answer = flyHash(tfidf_matrix.toarray())
    # 按照论文还有一个把高维度hash结果缩减到低维结果的步骤，此包中没有，在下方添加：
    answer = []
    if raw_answer.ndim == 1:
        raw_answer = [raw_answer]
    for sublist in raw_answer:
        # 缩减成1/k长度。每k个相加，如果大于0，则结果为1否则为0。存成01字符串
        sub_answer = [1 if (sum(sublist[i * k:i * k + k - 1]) > 0) else 0 for i in range(hash_dim)]
        answer.append("".join(map(str, sub_answer)))
    return str(answer)


if __name__ == "__main__":
    # 测试方法
    text = "ERK1/2 MAP kinases: structure, function, and regulation.ERK1 and ERK2 are related protein-serine/threonine kinases that participate in the Ras-Raf-MEK-ERK signal transduction cascade.This cascade participates in the regulation of a large variety of processes including cell adhesion, cell cycle progression, cell migration, cell survival, differentiation, metabolism, proliferation, and transcription.MEK1/2 catalyze the phosphorylation of human ERK1/2 at Tyr204/187 and then Thr202/185.The phosphorylation of both tyrosine and threonine is required for enzyme activation.Whereas the Raf kinase and MEK families have narrow substrate specificity, ERK1/2 catalyze the phosphorylation of hundreds of cytoplasmic and nuclear substrates including regulatory molecules and transcription factors.ERK1/2 are proline-directed kinases that preferentially catalyze the phosphorylation of substrates containing a Pro-Xxx-Ser/Thr-Pro sequence.Besides this primary structure requirement, many ERK1/2 substrates possess a D-docking site, an F-docking site, or both.A variety of scaffold proteins including KSR1/2, IQGAP1, MP1, β-Arrestin1/2 participate in the regulation of the ERK1/2 MAP kinase cascade.The regulatory dephosphorylation of ERK1/2 is mediated by protein-tyrosine specific phosphatases, protein-serine/threonine phosphatases, and dual specificity phosphatases.The combination of kinases and phosphatases make the overall process reversible.The ERK1/2 catalyzed phosphorylation of nuclear transcription factors including those of Ets, Elk, and c-Fos represents an important function and requires the translocation of ERK1/2 into the nucleus by active and passive processes involving the nuclear pore.These transcription factors participate in the immediate early gene response.The activity of the Ras-Raf-MEK-ERK cascade is increased in about one-third of all human cancers, and inhibition of components of this cascade by targeted inhibitors represents an important anti-tumor strategy.Thus far, however, only inhibition of mutant B-Raf (Val600Glu) has been found to be therapeutically efficacious."
    simhash = fp_with_simhash(text)
    print(simhash)
