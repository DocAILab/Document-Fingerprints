import pyssdeep
import numpy as np
from flyhash import FlyHash


def ssdeep():
    """
    ssdeep 是一个用来计算context triggered piecewise hashes(CTPH) 基于文本的分片哈希算法 ，同样也可以叫做模糊哈希 Fuzzy hashes。
    CTPH可以匹配同源文档（相似文档），这样的文档可能有一些顺序相同的字节，尽管这些字节可能在一个序列中长度和内容都不尽相同。
    SSDeep 对文本长度有要求，如果要生成有意义的结果，最好文本长度不小于 4096
    :return:
    """
    # ssdeep本身就内置了对文件hash的计算
    # 效果很好，对pdf文件仍有效
    result = pyssdeep.get_hash_buffer('The string for which you want to calculate a fuzzy hash')
    print(result)
    result1 = pyssdeep.get_hash_file(r'D:\OCR\OCR_test\OCR_test\data\a1.pdf')
    print(result1)
    result2 = pyssdeep.get_hash_file(r'D:\OCR\OCR_test\OCR_test\data\a2.pdf')
    print(result2)
    ro = pyssdeep.compare(result1, result2)
    print(ro)
    """
    ssdeep的主要原理是，使用一个弱哈希计算文件局部内容，
    在特定条件下对文件进行分片，然后使用一个强哈希对文件每片计算哈希值，取这些值的一部分并连接起来，与分片条件一起构成一个模糊哈希结果。
    使用一个字符串相似性对比算法判断两个模糊哈希值的相似度有多少，从而判断两个文件的相似程度。
    CTPH工作原理: https://www.sciencedirect.com/science/article/pii/S1742287606000764?via%3Dihub
    """


def flyhash():
    """
    FlyHash 是一种 LSH 算法，它将输入数据映射到稀疏哈希嵌入， 其中哈希嵌入的维度远大于输入， 并将输入数据的位置保留在哈希嵌入中。
    FlyHash被设计为计算成本低廉，但内存效率低。它适用于散列中小型数据 （~10-1000）到大哈希嵌入（~100-10000）
    :return:
    """
    # flyhash的工作时以维度为单位对数据进行随机，与文档信息或许不是很配合
    d = 10
    m = 100
    flyHash = FlyHash(d, m)
    data = np.random.randn(5, d)
    hashed_data = flyHash(data)
    print(hashed_data)


if __name__ == '__main__':
    flyhash()
    ssdeep()
