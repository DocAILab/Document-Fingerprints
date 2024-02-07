from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagModel
from text2vec import SentenceModel

def get_MokaAI_embedding(text, model_name='moka-ai/m3e-base'):
    model = SentenceTransformer(model_name)
    embedding = model.encode([text])
    return embedding[0]


def get_BAAI_embedding(text, model_name='BAAI/bge-large-zh-v1.5'):
    model = FlagModel(model_name)
    embedding = model.encode([text])
    return embedding[0]


def get_text2vec_embedding(text, model_name):
    model = SentenceModel(model_name)
    embedding = model.encode([text])
    return embedding[0]

def get_sentence_embedding(embedding_model, text):
    moka_ai_models = ['moka-ai/m3e-base', 'moka-ai/m3e-small', 'moka-ai/m3e-large']

    baai_models = ['BAAI/bge-small-zh', 'BAAI/bge-base-zh', 'BAAI/bge-large-zh',
                   'BAAI/bge-small-zh-v1.5', 'BAAI/bge-base-zh-v1.5', 'BAAI/bge-large-zh-v1.5',
                   'BAAI/bge-large-zh-noinstruct', 'BAAI/bge-reranker-large', 'BAAI/bge-reranker-base']

    text2vec_models = ['shibing624/text2vec-base-chinese-sentence', 'shibing624/text2vec-base-chinese-paraphrase',
                       'shibing624/text2vec-base-multilingual', 'shibing624/text2vec-base-chinese',
                       'shibing624/text2vec-bge-large-chinese', 'GanymedeNil/text2vec-large-chinese']


    if embedding_model in moka_ai_models:
        return get_MokaAI_embedding(text, embedding_model)
    elif embedding_model in baai_models:
        return get_BAAI_embedding(text, embedding_model)
    elif embedding_model in text2vec_models:
        return get_text2vec_embedding(text, embedding_model)
    else:
        raise ValueError("Invalid embedding model specified.")


if __name__ == "__main__":
    sentence1 = "Your sentence here."

    print(get_sentence_embedding('shibing624/text2vec-base-chinese-sentence', sentence1))
