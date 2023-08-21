"""
基础函数库
"""
from nltk.corpus import stopwords
from nltk import download

def dict_to_str(text_dict):
    """
    将dict类型文本转换为字符串。
    :param text_dict: 要转换的dict文本。
    :return: 转换后的字符串。
    """
    # 将dict按照键的顺序进行排序并拼接成字符串
    text_str = ''.join([f'{key}{text_dict[key]}' for key in sorted(text_dict.keys())])
    return text_str


def remove_stopwords(text):
    """
    句子切词并去除停用词。
    :param text: 输入文本，str
    :return: 切词列表，list
    """
    # 初始化停用词表
    try:
        stop_words = stopwords.words('english')
    except:
        download('stopwords')
        stop_words = stopwords.words('english')

    # 分词+去除停用词
    return [w for w in text.lower().split() if w not in stop_words]