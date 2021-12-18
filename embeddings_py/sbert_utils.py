import json
import ijson
from itertools import islice
from os.path import splitext

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

def read_authors(fname):
    _, file_extension = splitext(fname)
    if file_extension == '.json':
        with open(fname, encoding="utf8") as json_file:
            auth_dict = json.load(json_file)
    return auth_dict


def read_authors2(fname):
    _, file_extension = splitext(fname)
    auth_dict = []

    if file_extension == '.json':
        with open(fname) as f:
            objects = ijson.items(f, 'item')
            objects = islice(objects, 500)
            for author in objects:
                auth_dict.append(author)
    return auth_dict


def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    return model(**inputs).last_hidden_state[:, 0, :]


def get_specter_model():
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter')
    return model, tokenizer


def sbert_check_test():
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    sentences1 = ['The cat sits outside',
                  'A man is playing guitar',
                  'The new movie is awesome']

    sentences2 = ['The dog plays in the garden',
                  'A woman watches TV',
                  'The new movie is so great']

    # Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    # Output the pairs with their score
    for i in range(len(sentences1)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
