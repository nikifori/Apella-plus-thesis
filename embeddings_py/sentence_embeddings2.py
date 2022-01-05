# Specter Embeddings Csd_in and Csd_out at "https://drive.google.com/drive/folders/1SKbwnNQR3aa94far3JcBtbQ1DKZcVE2v?usp=sharing"
import re

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

from sentence_transformers import util
from emb_clustering import embeddings_clustering
from utils import *
from metrics import find_author_relevance, print_sorted_metrics


def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    return model(**inputs).last_hidden_state[:, 0, :]


def get_specter_model():
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter')
    return model, tokenizer


def create_position_object(title, description, targets_in, targets_in_standby, targets_out, targets_out_standby,
                           position_rank):
    position_dict = {
        "title": title,
        "description": description,
        "rank": position_rank,
        "targets_in": targets_in,
        "targets_in_standby": targets_in_standby,
        "targets_out": targets_out,
        "targets_out_standby": targets_out_standby

    }

    my_mkdir(r"./positions/")
    save2json(position_dict, path2save=r"./positions/{}.json".format(title))


def read_author_embedding_csv(author, csd_in=True):
    subfolder = "/csd_in/" if csd_in else "/csd_out/"

    fname = "author_embeddings" + subfolder + author['romanize name'].replace(" ", "_") + "_embeddings.csv"
    author_embedding = np.genfromtxt(fname, delimiter=',')
    return author_embedding


def read_author_embedding_dict(author):

    author_embedding = []

    if "Publications" not in author:
        return np.empty([0,0])

    for pub in author['Publications']:
        if "Specter embedding" in pub:
            # specter_emb = np.array(pub['Specter embedding'][0])
            specter_emb = np.array([float(f) for f in pub['Specter embedding'][0]])
            author_embedding.append(specter_emb)
        else:
            print("SPECTER EMBEDDING IS MISSING FROM AUTHOR{}".format(author['Name']))
            return np.empty()

    author_embedding = np.matrix(author_embedding)
    return author_embedding


def rank_candidates(fname, in_or_out, title, description, position_rank, mode='mean', clustering_type='agglomerative',n_clusters=5, reduction_type='PCA', reduced_dims=3):
    model, tokenizer = get_specter_model()
    title_embedding = get_embedding(title + tokenizer.sep_token + description, model, tokenizer)

    similarity_score = []
    roman_name = []

    with open(fname, encoding='utf-8') as f:
        objects = ijson.items(f, 'item')
        objects = islice(objects, 500)
        for author in objects:
            author_rank = find_author_rank(author)
            if author_rank <= position_rank and "Publications" in author:
                print(author['romanize name'])
                fname_base = f"./author_embeddings/specter_embeddings/aggregations/{in_or_out}/" + author['romanize name'].replace(" ", "_") + "/"

                if mode == 'mean':  # average of all paper embeddings (title + abstract, for each paper)
                    fname_in = fname_base + "aggregations.json"
                    try:
                        fname_in = fname_base + 'mean.csv'
                        aggregated_embeddings = np.genfromtxt(fname_in, delimiter=',')
                    except:
                        print("Except")
                        author_embeddings_np = read_author_embedding_dict(author)
                        author_embeddings = torch.tensor(author_embeddings_np)
                        if not author_embeddings_np.size: continue  # No publications found for this author
                        aggregated_embeddings = torch.mean(author_embeddings, dim=0)
                if mode == 'clustering':  # Creates paper cluster (after dimensionality reduction) for each author
                                          # and computes the cosine score for the most similar cluster to the title
                    try:
                        fname_in = fname_base + f'{clustering_type}_{n_clusters}_{reduction_type}_{reduced_dims}.csv'
                        aggregated_embeddings = np.genfromtxt(fname_in, delimiter=',')
                    except:
                        print("EXCEPT------------------")
                        try:
                            author_embeddings_np = read_author_embedding_dict(author)
                            author_embeddings = torch.tensor(author_embeddings_np)
                            if not author_embeddings_np.size: continue  # No publications found for this author
                            cluster_centroids = embeddings_clustering(author_embeddings, type=clustering_type,
                                                                          reduction_type=reduction_type,
                                                                          n_clusters=n_clusters)
                            aggregated_embeddings = torch.Tensor(np.matrix(cluster_centroids)).double()
                        except:
                            print("EXCEPT2------------------")
                            continue
                if mode == 'max_articles':  # Average of N most relevant papers (N=10 by default)
                    author_embeddings_np = read_author_embedding_dict(author)
                    author_embeddings = torch.tensor(author_embeddings_np)
                    if not author_embeddings_np.size: continue  # No publications found for this author
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
            else:
                sim_val = 0
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

    n_clusters = [2, 3, 4, 5, 7]
    dims_reduced = [2, 3, 5, 7, 10]
    clustering_types = ['agglomerative', 'kmeans']
    reduction_types = ['PCA', 'isomap']

    my_mkdir('./author_embeddings')
    my_mkdir('./author_embeddings/specter_embeddings')
    my_mkdir('./author_embeddings/specter_embeddings/aggregations')
    my_mkdir(f'./author_embeddings/specter_embeddings/aggregations/{in_or_out}')
    my_mkdir(f'./author_embeddings/specter_embeddings/aggregations/{in_or_out}/{auth_underscore_name}')

    fname_base = f'./author_embeddings/specter_embeddings/aggregations/{in_or_out}/{auth_underscore_name}/'

    for clustering_type in clustering_types:
        for reduction_type in reduction_types:
            for n_cluster in n_clusters:
                for d in dims_reduced:
                    try:
                        print("{}-{}-{}-{}".format(author['romanize name'], clustering_type, n_cluster, reduction_type, d))
                        author_centroids = embeddings_clustering(author_embeddings,clustering_type,reduction_type,n_cluster,dimensions_reduced=d)
                        fname_out = fname_base + "{}_{}_{}_{}.csv".format(clustering_type, n_cluster, reduction_type, d)
                        pd.DataFrame(np.matrix(author_centroids)).to_csv(fname_out, header=False, index=False)
                    except Exception as e:
                        print("Problem for author:{}, {}-{}-{}-{}".format(author['romanize name'], clustering_type, n_cluster, reduction_type, d))
                        print(e)

    fname_out = fname_base + "mean.csv"
    pd.DataFrame(np.matrix(torch.mean(author_embeddings, dim=0))).to_csv(fname_out, header=False, index=False)


def main_ranking_authors(fname, titles, descriptions, authors_targets, authors_targets_standby, position_ranks, ranking_mode, clustering_type, reduction_type, csd_in):

    n_clusters = 5
    reduced_dims = 3

    for i, title in enumerate(titles):
        in_or_out = 'in' if csd_in else 'out'
        res = rank_candidates(fname=fname,
                              in_or_out=in_or_out,
                              title=title,
                              description=descriptions[i],
                              position_rank=position_ranks[i],
                              mode=ranking_mode,
                              clustering_type=clustering_type,
                              n_clusters=n_clusters,
                              reduction_type=reduction_type,
                              reduced_dims=reduced_dims)

        # Format Csv Name
        my_mkdir('./results')
        my_mkdir('./results/specter')
        my_mkdir('./results/specter/{}'.format(title))
        my_mkdir('./results/specter/{}/{}'.format(title, in_or_out))

        fname_output = './results/specter/{}/{}/{}'.format(title, in_or_out, ranking_mode)
        if ranking_mode == 'clustering':
            fname_output += '_{}_{}'.format(clustering_type, reduction_type)
            version = f'{clustering_type}_{n_clusters}_{reduction_type}_{reduced_dims}'
        else:
            version = f'{ranking_mode}'

        res_target = find_author_relevance(title, version, in_or_out, authors_targets[i], authors_targets_standby[i], res)

        res.to_csv(fname_output + '.csv', encoding='utf-8', index=False)
        res_target.to_csv('{}_target.csv'.format(fname_output), encoding='utf-8', index=False)


if __name__ == '__main__':

    ##### SET PARAMETERS ######
    ranking_mode = 'max_articles'  # Average of N most relevant papers (N=10 by default)
    # ranking_mode = 'mean'  # Average of all paper embeddings (title + abstract, for each paper)
    # ranking_mode = 'clustering'    # Creates paper cluster (after dimensionality reduction) for each author
    # and computes the cosine score for the most similar cluster to the title

    clustering_type = 'kmeans'  # clustering_options = ['agglomerative', 'kmeans', 'dbscan']
    reduction_type = 'PCA'  # reduction_options = ['PCA', 'SVD', 'isomap', 'LLE']
    input_type = 'json'  # csv or json

    ##### SET TITLES ######
    titles = []
    descriptions = []
    authors_targets_in = []
    authors_targets_in_standby = []
    authors_targets_out = []
    authors_targets_out_standby = []
    position_ranks = []

    data = open_json(r'.\positions\test_apella_data.json')

    for i in data[:-1]:  # ignore last one without target_lists
        titles.append(i.get("title"))
        descriptions.append(i.get("description"))
        authors_targets_in.append(i.get("targets_in"))
        authors_targets_in_standby.append(i.get("targets_in_standby"))
        authors_targets_out.append(i.get("targets_out"))
        authors_targets_out_standby.append(i.get("targets_out_standby"))
        position_ranks.append(i.get("rank"))

    for i, title in enumerate(titles):
        create_position_object(title, descriptions[i],
                               targets_in=authors_targets_in[i],
                               targets_in_standby=authors_targets_in_standby[i],
                               targets_out=authors_targets_out[i],
                               targets_out_standby=authors_targets_out_standby[i],
                               position_rank=position_ranks[i])

    ##### CALCULATE RANKINGS ######
    fname_in = r'..\json_files\csd_in_with_abstract\csd_in_specter_no_greek_rank.json'
    fname_out = r'..\json_files\csd_out_with_abstract\csd_out_specter_out.json'

    main_ranking_authors(fname_in, titles, descriptions, authors_targets_in, authors_targets_in_standby, position_ranks, ranking_mode,
                         clustering_type, reduction_type, csd_in=True)
    main_ranking_authors(fname_out, titles, descriptions, authors_targets_out, authors_targets_out_standby, position_ranks,
                         ranking_mode, clustering_type, reduction_type, csd_in=False)

    # print_sorted_metrics(titles, metric='Average_Precision', ascending=False, in_or_out='in')
    # print_sorted_metrics(titles, metric='Average_Precision', ascending=False, in_or_out='out')
