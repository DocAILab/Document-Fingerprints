import json

from hashes import *
from similarities import calculate_similarity
import gensim.downloader as api
try:
    from FlagEmbedding import FlagModel
except ImportError:
    FlagModel = None
import numpy as np

class SimilarityModel:
    """
    SimilarityModel is a class that gets documents' fingerprints and calculates similarity between them.
    """

    def __init__(self, hash_name, similarity_name, args):
        self.args = args
        self.hash_name = hash_name
        self.similarity_name = similarity_name
        self.fps = dict()
        self.similarities = dict()
        if self.hash_name == 'BIGAE': # 提前用bigae的encoder测试得到按顺序的hash值
            if self.args.facet is None:
                with open(self.args.hash_path, 'r') as json_file:
                    self.hash_codes = json.load(json_file)
            else:
                self.hash_codes = {}
                with open(self.args.hash_path, 'r') as json_file_background:
                    self.hash_codes['background'] = json.load(json_file_background)
                with open(self.args.hash_path.replace('background', 'method'), 'r') as json_file_method:
                    self.hash_codes['method'] = json.load(json_file_method)
                with open(self.args.hash_path.replace('background', 'result'), 'r') as json_file_result:
                    self.hash_codes['result'] = json.load(json_file_result)
        # if self.similarity_name == 'wmd':
        #     self.model = api.load('word2vec-google-news-300')
        # if self.similarity_name == 'cosine' or 'manhattan' or 'mahalanobis':
        #     self.vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
        if "BGE" in self.hash_name:
            if FlagModel is None:
                raise ImportError(
                    "BGE-based methods require the FlagEmbedding package. "
                    "Install optional dependencies with: pip install FlagEmbedding"
                )
            self.dense_embedding_model = FlagModel(
                args.dense_embedding_model,
                query_instruction_for_retrieval="",
                use_fp16=True,
            )
        if self.hash_name == "fc-BGE":
            np.random.seed(123)
            input_dim = {
                "BAAI/bge-large-en-v1.5": 1024,
                "BAAI/bge-base-en-v1.5": 768,
                "BAAI/bge-small-en-v1.5": 384,
                "bge-large-en-v1.5": 1024,
                "bge-base-en-v1.5": 768,
                "bge-small-en-v1.5": 384
            }
            self.weight = np.random.uniform(-1, 1, size=(input_dim[args.dense_embedding_model], 128))

    def _cal_fp(self, text, pid, facet=None):
        """
        Calculate fingerprint of text(inner function)
        :param pid: the paper id of the document
        :param text: string to be fingerprinted
        :return: fingerprints
        """
        method_name = self.hash_name
        if method_name == 'SimHash':
            if self.args.hash_func is not None:
                return fp_with_simhash(text, self.args.hash_dim, self.args.hashfunc)
            return fp_with_simhash(text, self.args.hash_dim)
        elif method_name == 'MinHash':
            return fp_with_minhash(text, self.args.hash_dim, self.args.ngram)
        elif method_name == 'Winnowing':
            if self.args.hash_func is not None:
                return fp_with_winnowing(text, self.args.ngram, self.args.winnowing_window, self.args.hashfunc)
            return fp_with_winnowing(text, self.args.ngram, self.args.winnowing_window)
        elif method_name == 'FuzzyHash':
            return fp_with_fuzzyhash(text)
        elif method_name == 'FlyHash':
            return fp_with_flyhash([text], self.args.hash_dim)
        elif method_name == 'BIGAE':
            # 此处直接取出训练好的hash_codes（需要注意的事hash的json没有paper id，需要根据原数据集的文章对应的顺序来取）
            if facet is None:
                return self.hash_codes[pid]
            else:
                return self.hash_codes[facet][pid]
        elif method_name == 'EmbedBinHash':
            return fp_with_EmbedBinHash(text, self.dense_embedding_model)
        elif method_name in ['BGEHash']:
            return fp_with_BGEHash(text, self.dense_embedding_model)
        elif method_name in ["LSH-BGE", ]:
            return fp_with_LSH_BGEHash(text, self.dense_embedding_model, k=self.args.lsh_num_bucket, total_bits=self.args.lsh_hash_size)
        elif method_name in ["fc-BGE", ]:
            return fp_with_fc_bge(text, self.dense_embedding_model, self.weight)
        else:
            raise ValueError('Invalid fingerprint method name: {}'.format(method_name))

    def get_fp(self, pid, dataset):
        """
        Get fingerprint of a document
        :param pid: the paper id of the document
        :param dataset: the dataset of the document
        :return: fingerprint of the document
        """
        if pid not in self.fps:
            content = dataset.get(pid)
            content = content['TITLE'] + "".join(content['ABSTRACT'])
            fp = self._cal_fp(content, pid)
            self.fps[pid] = fp
        return self.fps[pid]

    def get_faced_fp(self, pid, dataset, facet):
        """
        Get fingerprint of a document(for CSFCube dataset only, which has facets)
        Compared with normal get_fp, this function select the sentence of the document according to the facet
        :param pid: the paper id of the document
        :param dataset: the dataset of the document
        :param facet: the facet of the document
        :return: fingerprint of the document
        """
        if pid not in self.fps:
            all_content = dataset.get(pid)
            # 提取facet的label，CSFCube数据集里pred_labels的取值有{background_label, objective_label, method_label,
            # result_label, other_label}，其中background_label和objective_label合并为background
            labels = ['background' if label == 'objective_label' else label[:-len('_label')]
                      for label in all_content['FACETS']]
            # 提取facet的内容
            facet_sentence_id = [i for i, k in enumerate(labels) if facet == k]
            facet_content = "".join([all_content['ABSTRACT'][i] for i in facet_sentence_id])
            if len(facet_content) <= 1 or facet_content is None:  # 如果对应facet的内容为空，则取整篇文章的内容
                facet_content = "".join(all_content['ABSTRACT'])
            fp = self._cal_fp(facet_content, pid, facet)
            self.fps[pid] = fp
        return self.fps[pid]

    def get_similarity(self, fp1, fp2):
        return calculate_similarity(fp1, fp2, self.similarity_name, self.args.text_to_vector_method)
        # if self.similarity_name == 'hamming':
        #     return hamming_distance(fp1, fp2, cal_simi=True)
        # elif self.similarity_name == 'jaccard':
        #     return jaccard_similarity(fp1, fp2)
        # elif self.similarity_name == 'multiset_jaccard':
        #     return multiset_jaccard_similarity(fp1, fp2)
        # elif self.similarity_name == 'levenshtein':
        #     return levenshtein_distance(fp1, fp2, cal_simi=True)
        # elif self.similarity_name == 'wmd':
        #     return wmd_distance(fp1, fp2, self.model)
        # elif self.similarity_name == 'cosine':
        #     return cosine_similarity(fp1, fp2, self.vectorizer)
        # elif self.similarity_name == 'manhattan':
        #     return manhattan_similarity(fp1, fp2)
        # elif self.similarity_name == 'mahalanobis':
        #     return mahalanobis_distance(fp1, fp2, cal_simi=True)
        # else:
        #     raise ValueError('Invalid similarity method name: {}'.format(self.similarity_name))
