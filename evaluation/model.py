import json

import numpy as np

from hashes import *
from similarities import calculate_similarity

try:
    from FlagEmbedding import FlagModel
except ImportError:
    FlagModel = None


class SimilarityModel:
    """Generate document fingerprints and compute pairwise similarities."""

    def __init__(self, hash_name, similarity_name, args):
        self.args = args
        self.hash_name = hash_name
        self.similarity_name = similarity_name
        self.fps = {}
        self.similarities = {}

        if self.hash_name == "BIGAE":
            if self.args.facet is None:
                with open(self.args.hash_path, "r", encoding="utf-8") as json_file:
                    self.hash_codes = json.load(json_file)
            else:
                self.hash_codes = {}
                with open(self.args.hash_path, "r", encoding="utf-8") as json_file_background:
                    self.hash_codes["background"] = json.load(json_file_background)
                with open(
                    self.args.hash_path.replace("background", "method"),
                    "r",
                    encoding="utf-8",
                ) as json_file_method:
                    self.hash_codes["method"] = json.load(json_file_method)
                with open(
                    self.args.hash_path.replace("background", "result"),
                    "r",
                    encoding="utf-8",
                ) as json_file_result:
                    self.hash_codes["result"] = json.load(json_file_result)

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
                "bge-small-en-v1.5": 384,
            }
            self.weight = np.random.uniform(
                -1,
                1,
                size=(input_dim[args.dense_embedding_model], 128),
            )

    def _cal_fp(self, text, pid, facet=None):
        method_name = self.hash_name
        if method_name == "SimHash":
            if self.args.hash_func is not None:
                return fp_with_simhash(text, self.args.hash_dim, self.args.hash_func)
            return fp_with_simhash(text, self.args.hash_dim)
        if method_name == "MinHash":
            return fp_with_minhash(text, self.args.hash_dim, self.args.ngram)
        if method_name == "Winnowing":
            if self.args.hash_func is not None:
                return fp_with_winnowing(
                    text,
                    self.args.ngram,
                    self.args.winnowing_window,
                    self.args.hash_func,
                )
            return fp_with_winnowing(text, self.args.ngram, self.args.winnowing_window)
        if method_name == "FuzzyHash":
            return fp_with_fuzzyhash(text)
        if method_name == "FlyHash":
            return fp_with_flyhash([text], self.args.hash_dim)
        if method_name == "BIGAE":
            if facet is None:
                return self.hash_codes[pid]
            return self.hash_codes[facet][pid]
        if method_name == "EmbedBinHash":
            return fp_with_EmbedBinHash(text, self.dense_embedding_model)
        if method_name == "BGEHash":
            return fp_with_BGEHash(text, self.dense_embedding_model)
        if method_name == "LSH-BGE":
            return fp_with_LSH_BGEHash(
                text,
                self.dense_embedding_model,
                k=self.args.lsh_num_bucket,
                total_bits=self.args.lsh_hash_size,
            )
        if method_name == "fc-BGE":
            return fp_with_fc_bge(text, self.dense_embedding_model, self.weight)
        raise ValueError(f"Invalid fingerprint method name: {method_name}")

    def get_fp(self, pid, dataset):
        """Get the full-document fingerprint for a paper."""
        if pid not in self.fps:
            content = dataset.get(pid)
            content = content["TITLE"] + "".join(content["ABSTRACT"])
            self.fps[pid] = self._cal_fp(content, pid)
        return self.fps[pid]

    def get_faced_fp(self, pid, dataset, facet):
        """Get a CSFCube facet-specific fingerprint for a paper."""
        cache_key = (pid, facet)
        if cache_key not in self.fps:
            all_content = dataset.get(pid)
            labels = [
                "background" if label == "objective_label" else label[:-len("_label")]
                for label in all_content["FACETS"]
            ]
            facet_sentence_id = [i for i, label in enumerate(labels) if facet == label]
            facet_content = "".join(
                all_content["ABSTRACT"][i] for i in facet_sentence_id
            )
            if facet_content is None or len(facet_content) <= 1:
                facet_content = "".join(all_content["ABSTRACT"])
            self.fps[cache_key] = self._cal_fp(facet_content, pid, facet)
        return self.fps[cache_key]

    def get_similarity(self, fp1, fp2):
        return calculate_similarity(
            fp1,
            fp2,
            self.similarity_name,
            self.args.text_to_vector_method,
        )
