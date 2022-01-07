import re

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

from sentence_transformers import util
from emb_clustering import embeddings_clustering
from utils import *
from metrics import *


def get_embedding(text, model, tokenizer):
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


def read_author_embedding_csv(author, csd_in=True):
    subfolder = "/csd_in/" if csd_in else "/csd_out/"

    fname = "author_embeddings" + subfolder + author['romanize name'].replace(" ", "_") + "_embeddings.csv"
    author_embedding = np.genfromtxt(fname, delimiter=',')
    return author_embedding


def read_author_embedding_dict(author):
    author_embedding = []

    if "Publications" not in author: return np.empty([0, 0])

    for pub in author['Publications']:
        if "Specter embedding" in pub:
            specter_emb = np.array([float(f) for f in pub['Specter embedding'][0]])
            author_embedding.append(specter_emb)
        else:
            print("SPECTER EMBEDDING IS MISSING FROM AUTHOR{}".format(author['Name']))
            return np.empty()

    author_embedding = np.matrix(author_embedding)
    return author_embedding


def calculate_similarities(title_emb: torch.tensor, author_embeddings, similarity_mode, clustering_parameters: dict, success=False):
    sim_val = 0.0
    if not author_embeddings.size: return sim_val  # No publications found for this author

    if similarity_mode == "mean":
        if not success: aggregated_embeddings = torch.mean(torch.tensor(author_embeddings), dim=0)
        else:           aggregated_embeddings = torch.tensor(author_embeddings)
        sim_val = util.pytorch_cos_sim(aggregated_embeddings, title_emb.double()).detach().cpu().numpy()
    elif similarity_mode == "clustering":
        if not success:
            author_embeddings = torch.tensor(author_embeddings)
            cluster_centroids = embeddings_clustering(author_embeddings, type=clustering_parameters["clustering_type"],
                                                  reduction_type=clustering_parameters["reduction_type"],
                                                  n_clusters=clustering_parameters["n_clusters"])
            aggregated_embeddings = torch.Tensor(np.matrix(cluster_centroids)).double()
        else: aggregated_embeddings = torch.tensor(author_embeddings)
        sim_val = max(util.pytorch_cos_sim(aggregated_embeddings, title_emb.double()))
    elif similarity_mode == "max_articles":
        N_articles = 10
        cos_scores = util.pytorch_cos_sim(torch.tensor(author_embeddings), title_emb.double()).detach().cpu().numpy()
        cos_scores = np.sort(cos_scores, axis=0, )[-N_articles:]
        sim_val = np.mean(cos_scores)
    else:
        print("ERROR @calculate similarity, GIVE correct similarity mode ('mean','clustering','max_articles'")

    return round(float(sim_val), 4)


def rank_candidates(fname, in_or_out, title_embedding, position_rank, mode, clustering_type,
                     n_clusters, reduction_type, reduced_dims):

    similarity_score = []
    roman_name = []
    clustering_params = {"clustering_type":clustering_type, "n_clusters":n_clusters, "reduction_type":reduction_type, "reduced_dims":reduced_dims}

    with open(fname, encoding='utf-8') as f:
        objects = ijson.items(f, 'item')
        objects = islice(objects, 500)
        for i, author in enumerate(objects):
            author_rank = find_author_rank(author)
            sim_val = 0.0
            print("{}.{}".format(i, author['romanize name']))
            fname_emb = f"./author_embeddings/specter_embeddings/aggregations/{in_or_out}/" + author['romanize name'].replace(" ", "_")

            if author_rank <= position_rank:
                try:
                    if mode == "mean": fname_emb += "/mean.csv"
                    elif mode == "clustering": fname_emb += f"/{clustering_type}_{n_clusters}_{reduction_type}_{reduced_dims}.csv"
                    author_embeddings_np = np.genfromtxt(fname_emb, delimiter=',')
                    sim_val = calculate_similarities(title_embedding, author_embeddings_np, mode, clustering_params,
                                                     success=True)
                except:
                    try:
                        author_embeddings_np = read_author_embedding_dict(author)
                        sim_val = calculate_similarities(title_embedding, author_embeddings_np, mode, clustering_params,success=False)
                    except: pass
            roman_name.append(author['romanize name'])
            similarity_score.append(sim_val)

    result = pd.DataFrame({'Name_roman': roman_name,
                           'Cosine_score': similarity_score}).sort_values(by=['Cosine_score'],
                                                                          ascending=False)
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


def create_author_aggregations(author, in_or_out="in"):
    papers = author['Publications'] if "Publications" in author else []
    print("Author:{}, total papers:{}".format(author['name'], len(papers)))
    auth_underscore_name = re.sub('/', '_', author["romanize name"])
    auth_underscore_name = re.sub(' ', '_', auth_underscore_name)
    author_embeddings = []
    if not len(papers): return

    for pub in author['Publications']:
        if "Specter embedding" in pub:
            # specter_emb = np.array(pub['Specter embedding'][0])
            specter_emb = np.array([float(f) for f in pub['Specter embedding'][0]])
            author_embeddings.append(specter_emb)
        else:
            print("SPECTER EMBEDDING IS MISSING FROM AUTHOR{}".format(author['Name']))
            return

    author_embeddings = torch.tensor(np.matrix(author_embeddings))

    # n_clusters = [2, 3, 4, 5, 7]
    # dims_reduced = [2, 3, 5, 7, 10, 768]
    # clustering_types = ['agglomerative', 'kmeans', 'spectral']
    # reduction_types = ['PCA', 'isomap']

    n_clusters = [2, 3, 4, 5, 7]
    dims_reduced = [2, 3, 5, 7, 10, 768]
    clustering_types = ['agglomerative_cosine', 'kmeans_cosine']
    reduction_types = ['PCA', 'isomap']

    mkdirs(f'./author_embeddings/specter_embeddings/aggregations/{in_or_out}/{auth_underscore_name}')
    fname_base = f'./author_embeddings/specter_embeddings/aggregations/{in_or_out}/{auth_underscore_name}/'

    for clustering_type in clustering_types:
        for reduction_type in reduction_types:
            for n_cluster in n_clusters:
                for d in dims_reduced:
                    try:
                        print("{}-{}-{}-{}".format(author['romanize name'], clustering_type, n_cluster, reduction_type,
                                                   d))
                        author_centroids = embeddings_clustering(author_embeddings, clustering_type, reduction_type,
                                                                 n_cluster, dimensions_reduced=d)
                        fname_out = fname_base + "{}_{}_{}_{}.csv".format(clustering_type, n_cluster, reduction_type, d)
                        pd.DataFrame(np.matrix(author_centroids)).to_csv(fname_out, header=False, index=False)
                    except Exception as e:
                        print("Problem for author:{}, {}-{}-{}-{}".format(author['romanize name'], clustering_type,
                                                                          n_cluster, reduction_type, d))
                        print(e)

    fname_out = fname_base + "mean.csv"
    pd.DataFrame(np.matrix(torch.mean(author_embeddings, dim=0))).to_csv(fname_out, header=False, index=False)


def main_ranking_authors(fname, titles, descriptions, authors_targets, authors_targets_standby, position_ranks,
                         ranking_mode, clustering_type, reduction_type, csd_in):
    n_clusters = 5
    reduced_dims = 3
    model, tokenizer = get_specter_model()

    for i, title in enumerate(titles):
        in_or_out = 'in' if csd_in else 'out'
        print("Title:{}".format(title))
        title_embedding = get_embedding(title + tokenizer.sep_token + descriptions[i], model, tokenizer)
        res = rank_candidates(fname=fname,
                               in_or_out=in_or_out,
                               title_embedding=title_embedding,
                               position_rank=position_ranks[i],
                               mode=ranking_mode,
                               clustering_type=clustering_type,
                               n_clusters=n_clusters,
                               reduction_type=reduction_type,
                               reduced_dims=reduced_dims)

        mkdirs(f'./results/specter/{title}/{in_or_out}')
        fname_output = './results/specter/{}/{}/{}'.format(title, in_or_out, ranking_mode)
        if ranking_mode == 'clustering':
            fname_output += '_{}_{}'.format(clustering_type, reduction_type)
            version = f'{clustering_type}_{n_clusters}_{reduction_type}_{reduced_dims}'
        else:
            version = f'{ranking_mode}'

        res_target = find_author_relevance(title, version, in_or_out, authors_targets[i], authors_targets_standby[i],
                                           res)

        res.to_csv(fname_output + '.csv', encoding='utf-8', index=False)
        res_target.to_csv('{}_target.csv'.format(fname_output), encoding='utf-8', index=False)


if __name__ == '__main__':
    fname_in = r'..\json_files\csd_in_with_abstract\csd_in_specter_no_greek_rank.json'
    fname_out = r'..\json_files\csd_out_with_abstract\csd_out_specter_out.json'

    ##### SET PARAMETERS ######
    # ranking_mode = 'max_articles'  # Average of N most relevant papers (N=10 by default)
    # ranking_mode = 'mean'  # Average of all paper embeddings (title + abstract, for each paper)
    ranking_mode = 'clustering'    # Creates paper cluster (after dimensionality reduction) for each author
    # and computes the cosine score for the most similar cluster to the title
    # clustering_options = ['agglomerative', 'kmeans', 'dbscan']
    # reduction_options = ['PCA', 'SVD', 'isomap', 'LLE']

    clustering_params = {
        "clustering_type": "agglomerative",
        "n_clusters": 5,
        "reduction_type": "PCA",
        "reduced_dims": 3
    }

    titles, descriptions, authors_targets_in, authors_targets_in_standby, authors_targets_out, authors_targets_out_standby, position_ranks = get_positions(
        r'.\positions\test_apella_data.json')

    ##### CALCULATE RANKINGS ######
    # main_ranking_authors(fname_in, titles, descriptions, authors_targets_in, authors_targets_in_standby, position_ranks,
    #                      ranking_mode,
    #                      clustering_params["clustering_type"], clustering_params["reduction_type"], csd_in=True)
    # main_ranking_authors(fname_out, titles, descriptions, authors_targets_out, authors_targets_out_standby,
    #                      position_ranks,
    #                      ranking_mode, clustering_params["clustering_type"], clustering_params["reduction_type"],
    #                      csd_in=False)

    # print_final_results(titles)
