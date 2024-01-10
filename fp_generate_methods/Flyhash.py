from flyhash import FlyHash
import numpy as np


def fp_with_flyhash(data, hash_dim, k=20, density=0.1, sparsity=0.05):
    """
    FlyHash 是一种 LSH 算法，它将输入数据映射到稀疏哈希嵌入， 其中哈希嵌入的维度远大于输入， 并将输入数据的位置保留在哈希嵌入中。
    FlyHash被设计为计算成本低廉，但内存效率低。它适用于散列中小型数据 （~10-1000）到大哈希嵌入（~100-10000）
    :param data: 输入数据, 一个列表，可以一维或二维
    :param hash_dim: 输出数据的维度，int
    :param k: 把hash映射到高维的比例，int。需要保证mk>data维度，论文中设置为20
    :param density: 决定投影矩阵的疏密。如果“density”是浮点数，则投影矩阵的每一列都有概率为“density”的非零条目;如果“density”是整数，则投影矩阵的每一列都恰好具有“density”非零条目。（论文采用0.1）
    :param sparsity: 决定hash结果的疏密（论文采用0.05）(理论上k*sparsity==1比较好)
    :return: 稀疏哈希嵌入结果，01字符串
    """
    # 需要先将data映射为数值类型,且等长，且转成np类型
    max_len = max(map(len, data))
    num_data = []
    for sublist in data:
        inner_list = []
        for char in sublist:
            inner_list.append(ord(char))
        inner_list += [0] * (max_len - len(sublist))
        num_data.append(inner_list)
    input_data = np.array(num_data, dtype=int)
    flyHash = FlyHash(max_len, hash_dim * k, density, sparsity)
    raw_answer = flyHash(input_data)
    # 按照论文还有一个把高维度hash结果缩减到低维结果的步骤，此包中没有，在下方添加：
    answer = []
    for sublist in raw_answer:
        # 缩减成1/k长度。每k个相加，如果大于0，则结果为1否则为0。存成01字符串
        sub_answer = [1 if (sum(sublist[i * k:i * k + k - 1]) > 0) else 0 for i in range(hash_dim)]
        answer.append("".join(map(str, sub_answer)))
    return answer


def calculate_similarity(fingerprint1, fingerprint2):
    if len(fingerprint1) != len(fingerprint2):
        raise ValueError("input fingerprints in type of str must have same dimension!")
    hamming_dist = sum(bit1 != bit2 for bit1, bit2 in zip(list(fingerprint1), list(fingerprint2)))
    dim = len(fingerprint1)
    return 1 - (hamming_dist / dim)


if __name__ == "__main__":
    str1 = '北京增值税电子普通发票.pdf'
    str2 = '福建增值税电子普通发票.pdf'
    str3 = '福建工程学院计算机学院培养方案.pdf'

    # 生成文件指纹并打印
    data = [list(str1), list(str2), list(str3)]
    fingerprints = fp_with_flyhash(data, 16)
    fingerprint1 = fingerprints[0]
    fingerprint2 = fingerprints[1]
    fingerprint3 = fingerprints[2]
    print(f"文件指纹1: {fingerprint1}")
    print(f"文件指纹2: {fingerprint2}")
    print(f"文件指纹3: {fingerprint3}")

    similarity_12 = calculate_similarity(fingerprint1, fingerprint2)
    similarity_13 = calculate_similarity(fingerprint1, fingerprint3)
    similarity_23 = calculate_similarity(fingerprint2, fingerprint3)

    print(f"相似度1和2: {similarity_12}")
    print(f"相似度1和3: {similarity_13}")
    print(f"相似度2和3: {similarity_23}")
