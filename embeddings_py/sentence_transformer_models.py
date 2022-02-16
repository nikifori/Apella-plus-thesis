import os.path
from itertools import islice
import ijson
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from embeddings.utils import get_underscored_name, mkdirs


def get_embedding(text, model, tokenizer, model_type="specter"):
    if model_type.startswith("specter_simcse") : return torch.tensor(model.encode(text))
    else:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        return model(**inputs).last_hidden_state[:, 0, :]


def get_scibert_model():
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    return model, tokenizer


def get_specter_model():
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter')
    return model, tokenizer


def get_custom_model(model_dir: str):
    if not os.path.exists(model_dir):
        print(f"Error while parsing: {model_dir}. Model path directory with this name does not exist!")
        return None

    model = SentenceTransformer(model_dir)
    tokenizer = None
    return model, tokenizer


def get_model(model_type: str,
              custom_model_dir=""):
    if model_type == "specter": return get_specter_model()
    if model_type in ['scibert_average', 'scibert_cls']: return get_scibert_model()
    if model_type.startswith("specter_simcse"): return get_custom_model(custom_model_dir)
    print("Error!! Invalid model name in get_model()")
    return None, None


def create_author_embeddings(author, model_name="specter", model=[], tokenizer=[], in_or_out="in"):
    if "Publications" not in author: return

    auth_underscore_name = get_underscored_name(author['romanize name'])
    fname_out = f'./author_embeddings/{model_name}_embeddings/{in_or_out}'
    emb_total = []
    mkdirs(fname_out)
    publication_texts = []

    print(f"{auth_underscore_name}, total papers:{len(author['Publications'])}")

    for paper in author['Publications']:
        try: title_abs = paper['Title'] + " [SEP] " + paper['Abstract'] if ("Abstract" in paper and paper["Abstract"]) else paper['Title']
        except: title_abs = paper['title'] + " [SEP] " + paper['Abstract'] if ("Abstract" in paper and paper["Abstract"]) else paper['title']
        publication_texts.append(title_abs)

    if model_name == "specter":
        for title_abs in publication_texts:
            emb_total.append(get_embedding(title_abs, model, tokenizer, model_name))
    else:   emb_total = model.encode(publication_texts)

    pd.DataFrame(emb_total).to_csv(fname_out + f'/{auth_underscore_name}.csv', header=False, index=False)


if __name__ == '__main__':
    pass
