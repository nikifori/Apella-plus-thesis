# Specter Embeddings Csd_in and Csd_out at "https://drive.google.com/drive/folders/1SKbwnNQR3aa94far3JcBtbQ1DKZcVE2v?usp=sharing"
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

from sbert_utils import read_authors
from transformers import AutoTokenizer, AutoModel

from emb_clustering import embeddings_clustering


def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    return model(**inputs).last_hidden_state[:, 0, :]


def get_specter_model():
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter')
    return model, tokenizer


def read_author_embedding_csv(author, csd_in=True):
    subfolder = "/csd_in/" if csd_in else "/csd_out/"

    fname = "author_embeddings" + subfolder + author['romanize name'].replace(" ", "_") + "_embeddings.csv"
    author_embedding = np.genfromtxt(fname, delimiter=',')
    return author_embedding


def read_author_embedding_dict(author):

    author_embedding = []

    for pub in author['Publications']:
        if "Specter embedding" in pub:
            specter_emb = np.array(pub['Specter embedding'][0])
            author_embedding.append(specter_emb)
        else:
            print("SPECTER EMBEDDING IS MISSING FROM AUTHOR{}".format(author['Name']))
            return np.empty()

    author_embedding = np.matrix(author_embedding)
    return author_embedding


def rank_candidates(auth_dict, title, description, mode='mean', clustering_type='agglomerative',csd_in=True,input_type='csv'):
    model, tokenizer = get_specter_model()

    title_embedding = get_embedding(title + tokenizer.sep_token + description, model, tokenizer)

    similarity_score = []
    roman_name = []

    for author in auth_dict:
        try:
            if input_type == "json":
                author_embeddings_np = read_author_embedding_dict(author)
            else:
                author_embeddings_np = read_author_embedding_csv(author, csd_in)

            author_embeddings = torch.tensor(author_embeddings_np)
            if not author_embeddings_np.size: continue  # No publications found for this author
        except Exception as e:
            print("File for {} NOT found...".format(author['romanize name']))
            continue

        if mode == 'mean':  # average of all paper embeddings (title + abstract, for each paper)
            aggregated_embeddings = torch.mean(author_embeddings, dim=0)
        if mode == 'clustering':  # Creates paper cluster (after dimensionality reduction) for each author
            # and computes the cosine score for the most similar cluster to the title
            n_clusters = 5
            cluster_centroids = embeddings_clustering(author_embeddings, type=clustering_type, n_clusters=n_clusters)
            aggregated_embeddings = torch.Tensor(np.matrix(cluster_centroids)).double()
        if mode == 'max_articles':  # Average of N most relevant papers (N=10 by default)
            N_articles = 10
            aggregated_embeddings = author_embeddings
            cos_scores = util.pytorch_cos_sim(aggregated_embeddings, title_embedding.double()).detach().cpu().numpy()
            cos_scores = np.sort(cos_scores, axis=0,)[-N_articles:]
            roman_name.append(author['romanize name'])
            similarity_score.append(np.mean(cos_scores))
            continue

        sim_val = max(util.pytorch_cos_sim(aggregated_embeddings, title_embedding.double()))
        roman_name.append(author['romanize name'])
        similarity_score.append(float(sim_val))

    result = pd.DataFrame({'Name_roman': roman_name,
                           'Cosine_score': similarity_score}).sort_values(by=['Cosine_score'],
                                                                          ascending=False)
    print("Title:{}".format(title))
    print(result)
    return result


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
        emb_total.append(result.last_hidden_state[:, 0, :].tolist()[0])

    pd.DataFrame(np.matrix(emb_total)).to_csv(
        "author_embeddings/" + author['romanize name'].replace(" ", "_") + "_embeddings.csv", header=False, index=False)


if __name__ == '__main__':

    # compare_authors_similarity(authors_dict[20], authors_dict[23], type='cosine')

    ############################################################################
    # # Calculate Author Embeddings and store them to CSV
    # model, tokenizer = get_specter_model()
    # for author in authors_dict[36:]:
    #     create_author_embeddings(author, model, tokenizer)

    ############################################################################
    # Compute Author Rankings with respect to similarity with a title (+description)

    # Get input data - preprocessing
    input_type = 'csv'  # or json
    csd_in = False
    # authors_dict = read_authors(r'..\json_files\csd_in_with_abstract\csd_in_with_abstracs_db.json')
    authors_dict = read_authors(r'..\json_files\csd_out_with_abstract\csd_out_with_abstracts_db.json')
    # authors_dict = read_authors(r'..\json_files\csd_out_with_abstract\csd_out_specter.json')
    # authors_dict = read_authors(r'..\json_files\csd_out_with_abstract\data.json')

    titles = ['Intelligent Systems - Symbolic Artificial Intelligence']
    description = 'Development of intelligent systems using a combination of methodologies of symbolic Artificial Intelligence, such as Representation of Knowledge and Reasoning, Multiagent Systems, Machine Learning, Intelligent Autonomous Systems, Planning and Scheduling of Actions, Satisfaction'

    # ranking_mode = 'max_articles'  # Average of N most relevant papers (N=10 by default)
    # ranking_mode = 'mean'         # Average of all paper embeddings (title + abstract, for each paper)
    ranking_mode = 'clustering'   # Creates paper cluster (after dimensionality reduction) for each author
                                    # and computes the cosine score for the most similar cluster to the title

    # clustering_options = ['agglomerative', 'kmeans', 'dbscan']
    clustering_type = 'agglomerative'

    for title in titles:
        res = rank_candidates(auth_dict=authors_dict,
                              title=title,
                              description=description,
                              mode=ranking_mode,
                              clustering_type=clustering_type,
                              csd_in=csd_in,
                              input_type=input_type)

        # Format Csv Name
        fname = './specter_rankings/{}___{}_{}'.format(title, 'in' if csd_in else 'out', ranking_mode)
        if ranking_mode == 'clustering': fname += '_{}'.format(clustering_type)
        fname += '.csv'

        res.to_csv(fname, encoding='utf-8', index=False)









# from pathlib import Path

# def simple_specter(papers, model="", tokenizer=""):
#     total_embeddings = []
#
#     # concatenate title and abstract
#     for batch in batching(papers):
#         print("Entering batch")
#         title_abs = [d['Title'] + tokenizer.sep_token + d['Abstract'] for d in batch]
#
#         # preprocess the input
#         for paper in title_abs:
#             inputs = tokenizer(paper, padding=True, truncation=True, return_tensors="pt", max_length=512)
#             result = model(**inputs)
#             print('---' + paper)
#
#             # take the first token in the batch as the embedding
#             embeddings = result.last_hidden_state[:, 0, :]
#             total_embeddings.append(embeddings.tolist()[0])
#
#     print("Exiting")
#     return np.matrix(total_embeddings)

# def get_embedding_model(model_name):
#     tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
#     model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
#     return model, tokenizer
#
#
# def get_sbert_model(model_name):
#     # Get a sentence bert model in general
#     model = SentenceTransformer(model_name)
#     model.max_seq_length = 512
#     return model
# def compare_authors_similarity(auth1, auth2, type='cosine'):
#     papers1 = auth1['Publications']
#     papers2 = auth2['Publications']
#
#     print("Author1: {}, len={}".format(auth1['name'], len(papers1)))
#     print("Author2: {}, len={}".format(auth2['name'], len(papers2)))
#
#     file1_path = Path("author_embeddings/csd_in/" + auth1['romanize name'].replace(" ", "_") + "_embeddings.csv")
#     file2_path = Path("author_embeddings/csd_in/" + auth2['romanize name'].replace(" ", "_") + "_embeddings.csv")
#
#     model, tokenizer = get_specter_model()
#     embeddings1 = read_author_embedding_csv(auth1) if file1_path.is_file() else simple_specter(papers1, model, tokenizer)
#     embeddings2 = read_author_embedding_csv(auth2) if file2_path.is_file() else simple_specter(papers2, model, tokenizer)
#
#     # Mean embeddings
#     auth1_embedding = np.asarray(np.mean(embeddings1, axis=0))
#     auth2_embedding = np.asarray(np.mean(embeddings2, axis=0))
#
#     if type == 'cosine':
#         res = util.pytorch_cos_sim(torch.tensor(auth1_embedding), torch.tensor(auth2_embedding))
#         print("Authors Cosine Score:{}".format(float(res)))
#     if type == 'euclidean':
#         res = np.linalg.norm(auth1_embedding - auth2_embedding)
#         print("Authors Euclidean Distance:{}".format(res))
#
#     # # Find the most relevant paper of Author 2 for each paper Author 1 has
#     # cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
#     # for i in range(len(papers1)):
#     #     i_max = max(range(len(cosine_scores[i, :])), key=cosine_scores[i, :].__getitem__)
#     #     if cosine_scores[i, i_max] > 0.5:
#     #         print("{} \t\t {} \t\t Score: {:.4f}".format(papers1[i], papers2[i_max], cosine_scores[i][i_max]))
