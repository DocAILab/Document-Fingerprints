import hashlib

import Levenshtein
from simhash import Simhash


def dict_to_str(text_dict):
    """
    将dict类型文本转换为字符串。
    :param text_dict: 要转换的dict文本。
    :return: 转换后的字符串。
    """
    # 将dict按照键的顺序进行排序并拼接成字符串
    text_str = ''.join([f'{key}{text_dict[key]}' for key in sorted(text_dict.keys())])
    return text_str


def generate_dict_fingerprint(text_dict):
    """
    使用Simhash算法生成dict类型文本的指纹。
    :param text_dict: 要生成指纹的dict文本。
    :return: dict文本的指纹。
    """
    # 将dict转换为字符串
    text_str = dict_to_str(text_dict)

    # 创建Simhash对象并生成指纹
    simhash = Simhash(text_str)

    return simhash


def calculate_hamming_distance(simhash1, simhash2):
    """
    计算两个Simhash指纹的汉明距离。
    :param simhash1: 第一个Simhash指纹。
    :param simhash2: 第二个Simhash指纹。
    :return: 汉明距离。
    """
    # 使用内置函数计算汉明距离
    xor_result = simhash1.value ^ simhash2.value
    hamming_dist = bin(xor_result).count('1')
    return hamming_dist


def calculate_similarity_edit_distance(text1, text2):
    """
    计算两个文本之间的编辑距离相似度。
    :param text1: 第一个文本。
    :param text2: 第二个文本。
    :return: 编辑距离相似度。
    """
    distance = Levenshtein.distance(text1, text2)
    max_length = max(len(text1), len(text2))
    similarity = 1 - (distance / max_length)
    return similarity


if __name__ == '__main__':
    # 示例dict类型文本
    dict1 = {'name': 'John', 'age': 30, 'city': 'New York'}
    dict2 = {'name': 'Jane', 'age': 28, 'city': 'San Francisco'}
    # dict3 = {'name': 'John', 'age': 30, 'city': 'New York'}
    dict3 = {'name': 'Chen', 'school': 'buaa', 'major': 'CS', 'course': '计组'}

    # 生成文件指纹
    fingerprint1 = generate_dict_fingerprint(dict1)
    fingerprint2 = generate_dict_fingerprint(dict2)
    fingerprint3 = generate_dict_fingerprint(dict3)

    print("dict1的文件指纹:", fingerprint1)
    print("dict2的文件指纹:", fingerprint2)
    print("dict3的文件指纹:", fingerprint3)

    # 计算相似度（汉明距离）
    # similarity1_2 = 1 - calculate_hamming_distance(fingerprint1, fingerprint2) / 64
    # similarity1_3 = 1 - calculate_hamming_distance(fingerprint1, fingerprint3) / 64
    # similarity2_3 = 1 - calculate_hamming_distance(fingerprint2, fingerprint3) / 64

    # 计算相似度（编辑距离）
    similarity1_2 = calculate_similarity_edit_distance(str(dict1), str(dict2))
    similarity1_3 = calculate_similarity_edit_distance(str(dict1), str(dict3))
    similarity2_3 = calculate_similarity_edit_distance(str(dict2), str(dict3))

    print("dict1和dict2的相似度:", similarity1_2)
    print("dict1和dict3的相似度:", similarity1_3)
    print("dict2和dict3的相似度:", similarity2_3)
