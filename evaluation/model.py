from hashes import *
from similarities import calculate_similarity
import gensim.downloader as api

class SimilarityModel:
    def __init__(self, hash_name, similarity_name, args):
        self.args = args
        self.hash_name = hash_name
        self.similarity_name = similarity_name
        if self.similarity_name == 'wmd':
            self.model = api.load('word2vec-google-news-300')
        if self.similarity_name == 'cosine' or 'manhattan' or 'mahalanobis':
            self.vectorizer = TfidfVectorizer(use_idf=True, norm='l2')

    def get_fp(self, text):
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
            return fp_with_flyhash(text, self.args.hash_dim, self.args.tokenizer)
        else:
            raise ValueError('Invalid fingerprint method name: {}'.format(method_name))

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
