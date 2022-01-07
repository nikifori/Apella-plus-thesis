import torch
import numpy as np
from metrics import *

from itertools import islice
import ijson
import re

from embeddings.emb_clustering import embeddings_clustering
from embeddings.metrics import find_author_relevance, average_precision
from embeddings.utils import find_author_rank, open_json, mkdirs, get_positions
from sentence_transformers import util
from embeddings.sentence_embeddings2 import calculate_similarities, get_scibert_model, get_embedding


def roman_name_to_fname(roman_name:str, in_or_out, aggregation_type="average"):
    auth_underscore_name = re.sub('/', '_', roman_name)
    auth_underscore_name = re.sub(' ', '_', auth_underscore_name)
    fname_emb = fr"./author_embeddings/scibert_embeddings/{aggregation_type}/{in_or_out}/{auth_underscore_name}.csv"
    return fname_emb


def rank_candidates(fname, in_or_out, title, description, position_rank, mode, clustering_params:dict):
    model, tokenizer = get_scibert_model()
    title_embedding = get_embedding(title + tokenizer.sep_token + description, model, tokenizer)

    similarity_score = []
    roman_name = []

    with open(fname, encoding='utf-8') as f:
        objects = ijson.items(f, 'item')
        objects = islice(objects, 500)
        for i, author in enumerate(objects):
            print(f"{i}.",author['romanize name'])
            author_rank = find_author_rank(author)
            sim_val = 0.0

            if author_rank <= position_rank:
                try:
                    fname_emb = roman_name_to_fname(author["romanize name"], in_or_out, aggregation_type="average")
                    author_embeddings = np.genfromtxt(fname_emb, delimiter=',')
                    sim_val = calculate_similarities(title_embedding, author_embeddings, mode, clustering_params, success=True)
                except:
                    pass
            roman_name.append(author['romanize name'])
            similarity_score.append(sim_val)

    result = pd.DataFrame({'Name_roman': roman_name,
                           'Cosine_score': similarity_score}).sort_values(by=['Cosine_score'],
                                                                          ascending=False)
    print("Title:{}".format(title))
    print(result)
    return result


def main_ranking_authors(fname, in_or_out, titles, descriptions, authors_targets, authors_targets_standby, position_ranks, ranking_mode, clustering_params):

    for i, title in enumerate(titles):
        version = f'scibert_{ranking_mode}'
        if ranking_mode == "random":
            random_rankings(fname, title, in_or_out, position_ranks[i], authors_targets[i], authors_targets_standby[i])
            continue
        else:
            res = rank_candidates(fname, in_or_out, title, descriptions[i], position_ranks[i], ranking_mode, clustering_params)

        mkdirs(f"./results/scibert/{title}/{in_or_out}")
        fname_output = './results/scibert/{}/{}/{}'.format(title, in_or_out, ranking_mode)
        if ranking_mode == 'clustering':
            fname_output += '_{}_{}'.format(clustering_params["clustering_type"], clustering_params["reduction_type"])
            version += f'_{clustering_params["clustering_type"]}_{clustering_params["n_clusters"]}_{clustering_params["reduction_type"]}_{clustering_params["reduced_dims"]}'

        res_target = find_author_relevance(title, version, in_or_out, authors_targets[i], authors_targets_standby[i],
                                           res)

        res.to_csv(fname_output + '.csv', encoding='utf-8', index=False)
        res_target.to_csv('{}_target.csv'.format(fname_output), encoding='utf-8', index=False)


def create_scibert_embeddings(fname: str):

    model, tokenizer = get_scibert_model()
    mkdirs("./author_embeddings/scibert_embeddings/average")

    with open(fname, encoding='utf-8') as f:
        authors = ijson.items(f, 'item')
        authors = islice(authors, 500)
        for i, author in enumerate(authors):
            print(f"{i}.",author['romanize name'])
            auth_underscore_name = re.sub('/', '_', author['romanize name'])
            auth_underscore_name = re.sub(' ', '_', auth_underscore_name)
            if "Publications" not in author:
                open(fr"./author_embeddings/scibert_embeddings/cls/{auth_underscore_name}.csv", 'w')
                open(fr"./author_embeddings/scibert_embeddings/average/{auth_underscore_name}.csv", 'w')
                continue

            cls_embeddings = []
            average_embeddings = []
            for pub in author["Publications"]:
                try:
                    text = pub['Title'] + ". " + pub["Abstract"]
                except:
                    text = pub['Title']
                model_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
                result = model(**model_input)
                embeddings = result.last_hidden_state[0, :, :].detach().cpu().numpy()

                cls_embeddings.append(embeddings[0, :])
                average_embeddings.append(np.mean(embeddings, axis=0))
            pd.DataFrame(np.matrix(np.matrix(cls_embeddings))).to_csv(fr"./author_embeddings/scibert_embeddings/cls/{auth_underscore_name}.csv", header=False,index=False)
            pd.DataFrame(np.matrix(np.matrix(average_embeddings))).to_csv(fr"./author_embeddings/scibert_embeddings/average/{auth_underscore_name}.csv", header=False,index=False)


def create_author_aggregations(author, in_or_out="in", aggregation_type="average"):
    papers = author['Publications'] if "Publications" in author else []
    print("Author:{}, total papers:{}".format(author['name'], len(papers)))
    auth_underscore_name = re.sub('/', '_', author["romanize name"])
    auth_underscore_name = re.sub(' ', '_', auth_underscore_name)
    author_embeddings = []

    fname_emb = roman_name_to_fname(author["romanize name"], in_or_out, aggregation_type=aggregation_type)
    author_embeddings = np.genfromtxt(fname_emb, delimiter=',')
    print(author_embeddings.shape)
    author_embeddings = torch.tensor(author_embeddings)

    n_clusters = [2, 3, 4, 5, 7]
    dims_reduced = [2, 3, 5, 7, 10]
    clustering_types = ['agglomerative', 'kmeans']
    reduction_types = ['PCA', 'isomap']

    mkdirs(f'./author_embeddings/scibert_embeddings/aggregations/{aggregation_type}/{in_or_out}/{auth_underscore_name}')
    fname_base = f'./author_embeddings/scibert_embeddings/aggregations/{aggregation_type}/{in_or_out}/{auth_underscore_name}/'

    for clustering_type in clustering_types:
        for reduction_type in reduction_types:
            for n_cluster in n_clusters:
                for d in dims_reduced:
                    try:
                        # print("{}-{}-{}-{}".format(author['romanize name'], clustering_type, n_cluster, reduction_type, d))
                        author_centroids = embeddings_clustering(author_embeddings,clustering_type,reduction_type,n_cluster,dimensions_reduced=d)
                        fname_out = fname_base + "{}_{}_{}_{}.csv".format(clustering_type, n_cluster, reduction_type, d)
                        pd.DataFrame(np.matrix(author_centroids)).to_csv(fname_out, header=False, index=False)
                    except Exception as e:
                        print("Problem for author:{}, {}-{}-{}-{}".format(author['romanize name'], clustering_type, n_cluster, reduction_type, d))
                        print(e)

    fname_out = fname_base + "mean.csv"
    pd.DataFrame(np.matrix(torch.mean(author_embeddings, dim=0))).to_csv(fname_out, header=False, index=False)



if __name__ == '__main__':

    ranking_mode = "max_articles"
    in_or_out = "in"
    clustering_params = {
        "clustering_type": "agglomerative",
        "n_clusters"     : 5,
        "reduction_type" : "PCA",
        "reduced_dims"   : 3
    }

    fname_in = r'..\json_files\csd_in_with_abstract\csd_in_specter.json'
    fname_out = r'..\json_files\csd_out_with_abstract\csd_out_completed_missing_2_no_greek_rank2.json'
    fname = fname_in if in_or_out == "in" else fname_out

    titles, descriptions, authors_targets_in, authors_targets_in_standby, authors_targets_out, authors_targets_out_standby, position_ranks = get_positions(
        r'.\positions\test_apella_data.json')

    if in_or_out == "in":
        main_ranking_authors(fname, in_or_out, titles, descriptions, authors_targets_in, authors_targets_in_standby,
                             position_ranks, ranking_mode, clustering_params)
    else:
        main_ranking_authors(fname, in_or_out, titles, descriptions, authors_targets_out, authors_targets_out_standby,
                             position_ranks, ranking_mode, clustering_params)
