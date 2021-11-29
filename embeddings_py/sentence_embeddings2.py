import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

from sbert_utils import chunks, batching, read_authors
from transformers import AutoTokenizer, AutoModel


def sentence_bert_model(model_name):
    # Get a sentence bert model in general
    model = SentenceTransformer(model_name)
    model.max_seq_length = 512
    return model


def get_specter_model():
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter')
    return model, tokenizer


def tokenize_papers(papers):
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    title_abs = [paper['Title'] + tokenizer.sep_token + paper['Abstract'] if "Abstract" in paper else paper['Title']  for paper in papers]
    return title_abs


def read_author_embedding(author, csd_in=True):
    if csd_in:
        subfolder = "/csd_in/"
    else:
        subfolder = "/csd_out/"

    fname = "author_embeddings" + subfolder + author['romanize name'].replace(" ", "_") + "_embeddings.csv"
    author_embedding = np.genfromtxt(fname, delimiter=',')
    return author_embedding


def rank_candidates_by_relevant_article(auth_dict, title):

    model, tokenizer = get_specter_model()

    inputs = tokenizer(title, padding=True, truncation=True, return_tensors="pt", max_length=512)
    result = model(**inputs)

    # take the first token in the batch as the embedding
    embeddings = np.array(result.last_hidden_state[:, 0, :].tolist()[0])

    similarity_score = []
    roman_name = []

    for author in auth_dict:
        fname = "author_embeddings/csd_in/" + author['romanize name'].replace(" ","_") + "_embeddings.csv"
        try:
            data = np.genfromtxt(fname, delimiter=',')
            max_sim_val = max(util.pytorch_cos_sim(torch.tensor(data), torch.tensor(embeddings)))
            roman_name.append(author['romanize name'])
            similarity_score.append(float(max_sim_val))
        except:
            print(author['romanize name'] + " NOT FOUND")

    result = pd.DataFrame({'roman names' : roman_name,
                           'Cosine score' : similarity_score
                           })
    print(result.sort_values(by=['Cosine score'], ascending=False))


def rank_candidates_by_relevant_profile(auth_dict, title):

    model, tokenizer = get_specter_model()

    inputs = tokenizer(title, padding=True, truncation=True, return_tensors="pt", max_length=512)
    result = model(**inputs)

    # take the first token in the batch as the embedding
    embeddings = result.last_hidden_state[:, 0, :]

    similarity_score = []
    roman_name = []

    for author in auth_dict:
        fname = "author_embeddings/csd_in/" + author['romanize name'].replace(" ", "_") + "_embeddings.csv"
        try:
            author_embeddings = torch.tensor(np.genfromtxt(fname, delimiter=','))
            mean_embedding = torch.mean(author_embeddings, dim=0)

            sim_val = util.pytorch_cos_sim(mean_embedding, embeddings.double())
            roman_name.append(author['romanize name'])
            similarity_score.append(float(sim_val))
        except Exception as e:
            print(author['romanize name'] + " NOT FOUND")

    result = pd.DataFrame({'roman names': roman_name,
                           'Cosine score': similarity_score})
    print(result.sort_values(by=['Cosine score'], ascending=False))
    return result


def compare_authors_similarity(auth1, auth2, type='cosine'):
    papers1 = auth1['Publications']
    papers2 = auth2['Publications']

    print("Author1: {}, len={}".format(auth1['name'], len(papers1)))
    print("Author2: {}, len={}".format(auth2['name'], len(papers2)))

    file1_path = Path("author_embeddings/csd_in/" + auth1['romanize name'].replace(" ", "_") + "_embeddings.csv")
    file2_path = Path("author_embeddings/csd_in/" + auth2['romanize name'].replace(" ", "_") + "_embeddings.csv")

    model, tokenizer = get_specter_model()
    embeddings1 = read_author_embedding(auth1) if file1_path.is_file() else simple_specter(papers1, model, tokenizer)
    embeddings2 = read_author_embedding(auth2) if file2_path.is_file() else simple_specter(papers2, model, tokenizer)

    # Mean embeddings
    auth1_embedding = np.asarray(np.mean(embeddings1, axis=0))
    auth2_embedding = np.asarray(np.mean(embeddings2, axis=0))

    if type == 'cosine':
        res = util.pytorch_cos_sim(torch.tensor(auth1_embedding), torch.tensor(auth2_embedding))
        print("Authors Cosine Score:{}".format(float(res)))
    if type == 'euclidean':
        res = np.linalg.norm(auth1_embedding-auth2_embedding)
        print("Authors Euclidean Distance:{}".format(res))

    # # Find the most relevant paper of Author 2 for each paper Author 1 has
    # cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    # for i in range(len(papers1)):
    #     i_max = max(range(len(cosine_scores[i, :])), key=cosine_scores[i, :].__getitem__)
    #     if cosine_scores[i, i_max] > 0.5:
    #         print("{} \t\t {} \t\t Score: {:.4f}".format(papers1[i], papers2[i_max], cosine_scores[i][i_max]))


def sbert_specter(papers, model=""):
    if model == "":
        model = sentence_bert_model('allenai/specter')

    tokenized_papers = tokenize_papers(papers)
    embeddings = model.encode(tokenized_papers, convert_to_tensor=True)
    return embeddings


def simple_specter(papers, model="", tokenizer=""):

    total_embeddings = []

    # concatenate title and abstract
    for batch in batching(papers):
        print("Entering batch")
        title_abs = [d['Title'] + tokenizer.sep_token + d['Abstract'] for d in batch]

        # preprocess the input
        for paper in title_abs:
            inputs = tokenizer(paper, padding=True, truncation=True, return_tensors="pt", max_length=512)
            result = model(**inputs)
            print('---' + paper)

            # take the first token in the batch as the embedding
            embeddings = result.last_hidden_state[:, 0, :]
            total_embeddings.append(embeddings.tolist()[0])

    print("Exiting")
    return np.matrix(total_embeddings)


def create_author_embeddings(author, model="", tokenizer=""):
    if model == "" or tokenizer == "":
        model, tokenizer = get_specter_model()

    papers = author['Publications'] if "Publications" in author else []
    print("Author:{}, total papers:{}".format(author['name'], len(papers)))
    emb_total = []

    for paper in papers:
        title_abs = paper['Title'] + tokenizer.sep_token + paper['Abstract'] if "Abstract" in paper else paper['Title']
        inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        result = model(**inputs)
        print('---' + paper['Title'])
        emb_total.append(result.last_hidden_state[:,0,:].tolist()[0])
        # print(paper['Title'])

    pd.DataFrame(np.matrix(emb_total)).to_csv("author_embeddings/" + author['romanize name'].replace(" ","_") + "_embeddings.csv", header=False, index=False)


if __name__ == '__main__':
    # freeze_support()

    # Get input data - preprocessing
    authors_dict = read_authors(r'..\json_files\csd_in_with_abstract\csd_in_with_abstracs_db.json')
    # authors_dict = read_authors(r'..\json_files\csd_out_with_abstract\csd_out_with_abstracts_db.json')

    # create_author_embeddings(authors_dict[23])

    compare_authors_similarity(authors_dict[20], authors_dict[23], type='cosine')
    compare_authors_similarity(authors_dict[20], authors_dict[23], type='euclidean')

    # # Calculate Author Embeddings and store them to CSV
    # model, tokenizer = get_specter_model()
    # for author in authors_dict[12:]:
    #     create_author_embeddings(author, model, tokenizer)

    # create_author_embeddings_complete(authors_dict[1:26], thread_num=4)

    title = "Robotics and AI systems"
    rank_candidates_by_relevant_article(authors_dict, title)
    rank_candidates_by_relevant_profile(authors_dict, title)
