from simhash import Simhash, SimhashIndex


def generate_simhash_fingerprint(text):
    simhash = Simhash(text)
    # 返回一个十六进制数
    return simhash
    # 返回Simhash值的哈希摘要（可以选择MD5、SHA-1等哈希算法）
    # return hashlib.md5(str(simhash.value).encode()).hexdigest()


def calculate_similarity(simhash1, simhash2):
    distance = simhash1.distance(simhash2)
    similarity = 1 - (distance / 64)  # 假设Simhash的num_perm为64
    return similarity


if __name__ == '__main__':
    str1 = '北京增值税电子普通发票.pdf'
    str2 = '福建增值税电子普通发票.pdf'
    str3 = '福建工程学院计算机学院培养方案.pdf'

    # 生成文件指纹并打印
    fingerprint1 = generate_simhash_fingerprint(str1)
    fingerprint2 = generate_simhash_fingerprint(str2)
    fingerprint3 = generate_simhash_fingerprint(str3)
    print(f"文件指纹1: {fingerprint1}")
    print(f"文件指纹2: {fingerprint2}")
    print(f"文件指纹3: {fingerprint3}")

    # 计算相似度
    similarity_12 = calculate_similarity(fingerprint1, fingerprint2)
    similarity_13 = calculate_similarity(fingerprint1, fingerprint3)
    similarity_23 = calculate_similarity(fingerprint2, fingerprint3)

    print(f"相似度1和2: {similarity_12}")
    print(f"相似度1和3: {similarity_13}")
    print(f"相似度2和3: {similarity_23}")
