from hashes import *
from similarities import calculate_similarity
import gensim.downloader as api


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
        # if self.similarity_name == 'wmd':
        #     self.model = api.load('word2vec-google-news-300')
        # if self.similarity_name == 'cosine' or 'manhattan' or 'mahalanobis':
        #     self.vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

    def _cal_fp(self, text):
        """
        Calculate fingerprint of text(inner function)
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
            fp = self._cal_fp(content)
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
            fp = self._cal_fp(facet_content)
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
