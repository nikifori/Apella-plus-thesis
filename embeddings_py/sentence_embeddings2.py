import torch
import numpy as np
import pandas as pd

from sentence_transformers import util
from emb_clustering import embeddings_clustering
from embeddings.sentence_transformer_models import get_specter_model, get_embedding, get_model
from utils import *
from metrics import *


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


def load_auth_embeddings(author, model_name="specter", in_or_out="in"):
    auth_underscore_name = get_underscored_name(author['romanize name'])

    if model_name == "specter": return read_author_embedding_dict(author)
    elif model_name in ['scibert_average', 'scibert_cls']:
        fname = f'./author_embeddings/{model_name}_embeddings/{in_or_out}/{auth_underscore_name}.csv'
        author_embedding = np.genfromtxt(fname, delimiter=',')
        return author_embedding
    else:
        return np.empty([0,0])


def calculate_similarities(title_emb: torch.tensor, author_embeddings:np.ndarray, similarity_mode, clustering_parameters: dict, success=False):
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


def rank_candidates(fname, model_name, in_or_out, title_embedding, position_rank, mode, clustering_type,
                    n_clusters, reduction_type, reduced_dims):

    similarity_score = []
    roman_name = []
    clustering_params = {"clustering_type":clustering_type, "n_clusters":n_clusters, "reduction_type":reduction_type, "reduced_dims":reduced_dims}

    with open(fname, encoding='utf-8') as f:
        objects = ijson.items(f, 'item')
        objects = islice(objects, 500)
        for i, author in enumerate(objects):
            author_rank = find_author_rank(author)
            auth_underscore_name = get_underscored_name(author['romanize name'])
            sim_val = 0.0
            print("{}.{}".format(i, author['romanize name']))
            fname_emb = f"./author_embeddings/{model_name}_embeddings/aggregations/{in_or_out}/" + author['romanize name'].replace(" ", "_")

            if author_rank <= position_rank:
                try:
                    if mode == "mean": fname_emb += "/mean.csv"
                    elif mode == "clustering": fname_emb += f"/{clustering_type}_{n_clusters}_{reduction_type}_{reduced_dims}.csv"
                    author_embeddings_np = np.genfromtxt(fname_emb, delimiter=',')
                    sim_val = calculate_similarities(title_embedding, author_embeddings_np, mode, clustering_params,
                                                     success=True)
                except:
                    try:
                        if model_name == "specter":
                            author_embeddings_np = read_author_embedding_dict(author)
                        else:
                            fname_embeddings = f"./author_embeddings/{model_name}_embeddings/{in_or_out}/{auth_underscore_name}.csv"
                            author_embeddings_np = np.genfromtxt(fname_embeddings, delimiter=',')
                        sim_val = calculate_similarities(title_embedding, author_embeddings_np, mode, clustering_params,success=False)
                    except: print("Couldn't read embeddings for {}".format(author['romanize name']))
            roman_name.append(author['romanize name'])
            similarity_score.append(sim_val)

    result = pd.DataFrame({'Name_roman': roman_name,
                           'Cosine_score': similarity_score}).sort_values(by=['Cosine_score'],
                                                                          ascending=False)
    print(result)
    return result


def create_author_aggregations(author, model_name="specter", in_or_out="in"):
    if "Publications" not in author: return

    auth_underscore_name = get_underscored_name(author['romanize name'])
    print(f"Author:{auth_underscore_name}")
    model_dir = f"{model_name}_embeddings"
    fname_base = f'./author_embeddings/{model_dir}/aggregations/{in_or_out}/{auth_underscore_name}/'
    auth_embeddings_path = f'./author_embeddings/{model_dir}/{in_or_out}/{auth_underscore_name}.csv'
    if model_name == "specter": author_embeddings = torch.tensor(load_auth_embeddings(author, model_name=model_name, in_or_out=in_or_out))
    else:                       author_embeddings = torch.tensor(np.genfromtxt(auth_embeddings_path, delimiter=','))
    mkdirs(fname_base)

    clustering_types = ['kmeans', 'kmeans_euclidean_constr']
    reduction_types = ['PCA', 'isomap']
    n_clusters = [5, 7]
    dims_reduced = [7, 10]

    for clustering_type in clustering_types:
        for reduction_type in reduction_types:
            for n_cluster in n_clusters:
                for d in dims_reduced:
                    try:
                        # print(f'{auth_underscore_name}-{clustering_type}-{n_cluster}-{reduction_type}-{d}')
                        author_centroids = embeddings_clustering(author_embeddings, clustering_type, reduction_type, n_cluster, dimensions_reduced=d)
                        fname_out = fname_base + "{}_{}_{}_{}.csv".format(clustering_type, n_cluster, reduction_type, d)
                        pd.DataFrame(np.matrix(author_centroids)).to_csv(fname_out, header=False, index=False)
                    except Exception as e:
                        print(f'Problem for: {auth_underscore_name}-{clustering_type}-{n_cluster}-{reduction_type}-{d} \n{e}')

    fname_out = fname_base + "mean.csv"
    pd.DataFrame(np.matrix(torch.mean(author_embeddings, dim=0))).to_csv(fname_out, header=False, index=False)


def main_ranking_authors(fname, model_name, titles, descriptions, authors_targets, authors_targets_standby, position_ranks,
                         ranking_mode, clustering_type, reduction_type, csd_in, custom_model_dir=""):
    n_clusters = 5
    reduced_dims = 3
    model, tokenizer = get_model(model_name, custom_model_dir)

    for i, title in enumerate(titles):
        in_or_out = 'in' if csd_in else 'out'
        print("Title:{}".format(title))
        try: title_embedding = get_embedding(title + tokenizer.sep_token + descriptions[i], model, tokenizer, model_name)
        except: title_embedding = get_embedding(title + " [SEP] " + descriptions[i], model, tokenizer, model_name)
        res = rank_candidates(fname=fname,
                              model_name=model_name,
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
            version = f'specter_{clustering_type}_{n_clusters}_{reduction_type}_{reduced_dims}'
        else: version = f'specter_{ranking_mode}'

        res_target = find_author_relevance(title, version, in_or_out, authors_targets[i], authors_targets_standby[i],
                                           res)

        res.to_csv(fname_output + '.csv', encoding='utf-8', index=False)
        res_target.to_csv('{}_target.csv'.format(fname_output), encoding='utf-8', index=False)


def read_author_aggregations(author, model_name, in_or_out, version):
    author_underscore_name = get_underscored_name(author["romanize name"])
    fname_emb = f"./author_embeddings/{model_name}_embeddings/{in_or_out}/{author_underscore_name}.csv"
    fname_aggr = f"./author_embeddings/{model_name}_embeddings/aggregations/{in_or_out}/{author_underscore_name}/" + version["version_str"] + ".csv"

    ################# MAX ARTICLES
    if version["version_str"] == "max_articles":
        if model_name == "specter":
            return torch.tensor(read_author_embedding_dict(author))
        else:
            aggregated_embeddings_np = np.genfromtxt(fname_emb, delimiter=',')
            return aggregated_embeddings_np
    ################# MEAN
    elif version["version_str"] == "mean":
        if model_name == "specter":
            aggregated_embeddings = torch.mean(torch.tensor(read_author_embedding_dict(author)), dim=0)
            return np.matrix(aggregated_embeddings)
        else:
            aggregated_embeddings_np = np.genfromtxt(fname_emb, delimiter=',')
            aggregated_embeddings = torch.mean(torch.tensor(aggregated_embeddings_np), dim=0)
            return np.matrix(aggregated_embeddings)
    ################# CLUSTERING
    else:
        try:
            aggregated_embeddings_np = np.genfromtxt(fname_aggr, delimiter=',')
            return aggregated_embeddings_np
        except:
            print("ERROR parsing aggregations for {}".format(version["version_str"]))
            if model_name == "specter":
                author_embeddings_np = read_author_embedding_dict(author)
                author_embeddings = torch.tensor(author_embeddings_np)
            else:
                try:
                    author_embeddings_np = np.genfromtxt(fname_emb, delimiter=',')
                    author_embeddings = torch.tensor(author_embeddings_np)
                except:
                    print("Double ERRROR")
                    return None
            if not author_embeddings_np.size: return None
            try:
                cluster_centroids = embeddings_clustering(author_embeddings, type=version["clustering_type"],
                                                          reduction_type=version["reduction_type"],
                                                          n_clusters=version["n_cluster"],
                                                          dimensions_reduced=version["d"])
                aggregated_embeddings_np = np.matrix(cluster_centroids)
                return aggregated_embeddings_np
            except:
                print("Double ERRROR")
                return None


def get_clustering_params_dict(version):
    if not version["general_type"] == "clustering": return {}

    clustering_params = {
        "clustering_type": version["clustering_type"],
        "n_clusters": version["n_cluster"],
        "reduction_type": version["reduction_type"],
        "reduced_dims": version["d"]
    }
    return clustering_params


def main_ranking_authors_aggregations(fname, model_name, titles, descriptions, authors_targets, authors_targets_standby, position_ranks,
                          in_or_out, custom_model_dir=""):

    clustering_types = ['kmeans', 'kmeans_euclidean_constr']
    reduction_types = ['PCA', 'isomap']
    n_clusters = [5, 7]
    dims_reduced = [7, 10]

    model, tokenizer = get_model(model_type=model_name, custom_model_dir=custom_model_dir)
    similarity_scores, roman_names = ({}, {})
    positions, versions = ([], [])

    for i, title in enumerate(titles):
        title_embedding = get_embedding(title + " " + descriptions[i], model, tokenizer, model_type=model_name)
        try:    position = {'title': title, 'title_embedding': np.double(title_embedding), 'rank': position_ranks[i]}
        except: position = {'title': title, 'title_embedding': np.double(title_embedding.detach().numpy()), 'rank': position_ranks[i]}
        positions.append(position)

        versions.append({'general_type': "max_articles", 'pos_id': i, 'version_str': 'max_articles', 'title': title})
        similarity_scores[f'max_articles_{title}'] = []
        roman_names[f'max_articles_{title}'] = []

    for clustering_type in clustering_types:
        for reduction_type in reduction_types:
            for n_cluster in n_clusters:
                for d in dims_reduced:
                    for i, position in enumerate(positions):
                        title = position['title']
                        version_str = f'{clustering_type}_{n_cluster}_{reduction_type}_{d}'
                        similarity_scores[f'{version_str}_{title}'] = []
                        roman_names[f'{version_str}_{title}'] = []
                        version = {'general_type': "clustering", 'clustering_type': clustering_type , 'reduction_type': reduction_type,
                                   'n_cluster':n_cluster, 'd': d, 'pos_id': i, 'version_str': version_str, 'title': title}
                        versions.append(version)

    with open(fname, encoding='utf-8') as f:
        objects = ijson.items(f, 'item')
        objects = islice(objects, 500)
        for i, author in enumerate(objects):
            print("{}.{}".format(i, author['romanize name']))
            for version in versions:
                sim_val = 0.0
                pos_id = version["pos_id"]
                version_str_full = version["version_str"] + "_" + version["title"]
                if find_author_rank(author) <= positions[pos_id]["rank"]:
                    try:
                        aggregated_embeddings = read_author_aggregations(author, model_name, in_or_out, version)
                        sim_val = calculate_similarities(torch.tensor(positions[pos_id]['title_embedding']),
                                                         aggregated_embeddings, similarity_mode= version["general_type"],
                                                         clustering_parameters=get_clustering_params_dict(version), success=True)
                    except: print("Couldn't calculate similarity for {}".format(version["version_str"]))
                roman_names[version_str_full].append(author['romanize name'])
                similarity_scores[version_str_full].append(sim_val)

    for version in versions:
        pos_id = version["pos_id"]
        title = positions[pos_id]['title']
        version_str = version["version_str"]
        version_str_full = version["version_str"] + "_" + title

        result = pd.DataFrame( {'Name_roman': roman_names[version_str_full],
                                'Cosine_score': similarity_scores[version_str_full]}).sort_values(by=['Cosine_score'],ascending=False)
        print(f"Title:{title}")
        print(result)

        mkdirs(f"./results/{model_name}/{title}/all/{in_or_out}")
        fname_output = f"./results/{model_name}/{title}/all/{in_or_out}/{version_str}"

        res_target = find_author_relevance(title, f"{model_name}_{version_str}", in_or_out, authors_targets[pos_id], authors_targets_standby[pos_id], result)
        result.to_csv(fname_output + '.csv', encoding='utf-8', index=False)
        res_target.to_csv('{}_target.csv'.format(fname_output), encoding='utf-8', index=False)


if __name__ == '__main__':
    fname_in = r'..\json_files\csd_in_with_abstract\csd_in_specter_no_greek_rank.json'
    fname_out = r'..\json_files\csd_out_with_abstract\csd_out_specter_rank_no_nan.json'

    ##### SET PARAMETERS ######
    # ranking_mode = 'max_articles'  # Average of N most relevant papers (N=10 by default)
    # ranking_mode = 'mean'  # Average of all paper embeddings (title + abstract, for each paper)
    ranking_mode = 'clustering'    # Creates paper cluster (after dimensionality reduction) for each author and computes the cosine score for the most similar cluster to the title
    model_name = "specter"
    # clustering_params = {
    #     "clustering_type": "agglomerative",  #clustering_options = ['agglomerative', 'kmeans', 'dbscan']
    #     "n_clusters": 5,
    #     "reduction_type": "PCA",             # reduction_options = ['PCA', 'SVD', 'isomap', 'LLE']
    #     "reduced_dims": 3
    # }

    titles, descriptions, authors_targets_in, authors_targets_in_standby, authors_targets_out, authors_targets_out_standby, position_ranks = get_positions(r'.\positions\test_apella_data.json')

    # ##### CALCULATE RANKINGS ######
    # main_ranking_authors_aggregations(fname_in, model_name, titles, descriptions, authors_targets_in, authors_targets_in_standby, position_ranks, "in")
    main_ranking_authors_aggregations(fname_out, model_name, titles, descriptions, authors_targets_out, authors_targets_out_standby, position_ranks, "out")

    print_final_results(titles)
