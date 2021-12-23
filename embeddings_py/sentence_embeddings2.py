# Specter Embeddings Csd_in and Csd_out at "https://drive.google.com/drive/folders/1SKbwnNQR3aa94far3JcBtbQ1DKZcVE2v?usp=sharing"
import torch
import numpy as np
import pandas as pd

from sentence_transformers import util
from sbert_utils import get_embedding, get_specter_model
from emb_clustering import embeddings_clustering, average_precision
from utils import *


def create_position_object(title, description, targets_in, targets_in_standby, targets_out, targets_out_standby):
    position_dict = {
        "title": title,
        "description": description,
        "targets_in": targets_in,
        "targets_in_standby": targets_in_standby,
        "targets_out": targets_out,
        "targets_out_standby": targets_out_standby
    }

    my_mkdir(r"./specter_rankings/")
    my_mkdir(r"./specter_rankings/{}".format(title))
    my_mkdir(r"./specter_rankings/{}/{}".format(title, title))
    save2json(position_dict, path2save=r"./specter_rankings/{}/{}.json".format(title, title))


def find_author_relevance(authors_target, authors_target_standby, result):
    print("Selected Authors ranking:")
    result_names = list(result[['Name_roman']].values.flatten())
    total_authors = result[['Name_roman']].size
    n_authors_target_all = len(authors_target) + len(authors_target_standby)

    k = int(1.5 * n_authors_target_all)
    top_k = []
    target_result = []

    if len(authors_target) == 0:
        return pd.DataFrame({'target_result': []})

    sum_of_ranking = 0
    for i in range(total_authors):
        if result_names[i] in authors_target:
            print('{}/{}:{}'.format(i + 1, total_authors, result_names[i]))
            sum_of_ranking += i + 1
            target_result.append('{}/{}:{}'.format(i + 1, total_authors, result_names[i]))
            if i <= k: top_k.append(i + 1)
        elif result_names[i] in authors_target_standby:
            print('{}/{} ({}):{}'.format(i + 1, total_authors, "standby", result_names[i]))
            sum_of_ranking += i + 1
            target_result.append('{}/{} ({}):{}'.format(i + 1, total_authors, "standby", result_names[i]))
            if i <= k: top_k.append(i + 1)

    # metric = (n_authors_target_all) * (n_authors_target_all + 1) / (2 * sum_of_ranking)
    # top_k = len(top_k) / n_authors_target_all * 100
    # print("Metric1:{}, top_k={}% (top {} of {})".format(metric, top_k, k, total_authors))
    # target_result.append(f"Metric1: {metric}")
    # target_result.append(f'top_k={top_k}% (top {k} of {total_authors})')
    averageP = average_precision(authors_target, authors_target_standby, result_names)
    target_result.append(f"Average Precision: {averageP}")
    print('Average Precision: {}'.format(averageP))
    
    return pd.DataFrame({'target_result': target_result})


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


def rank_candidates(fname, title, description, mode='mean', clustering_type='agglomerative', reduction_type='PCA'):
    model, tokenizer = get_specter_model()

    title_embedding = get_embedding(title + tokenizer.sep_token + description, model, tokenizer)

    similarity_score = []
    roman_name = []

    with open(fname) as f:
        objects = ijson.items(f, 'item')
        objects = islice(objects, 500)
        for author in objects:
            print(author['romanize name'])
            fname_base = "./author_embeddings/aggregations/" + author['romanize name'].replace(" ", "_") + "/"

            if mode == 'mean':  # average of all paper embeddings (title + abstract, for each paper)
                fname_in = fname_base + "mean.csv"
                try:
                    aggregated_embeddings = np.genfromtxt(fname_in, delimiter=',')
                except:
                    author_embeddings_np = read_author_embedding_dict(author)
                    author_embeddings = torch.tensor(author_embeddings_np)
                    if not author_embeddings_np.size: continue  # No publications found for this author
                    aggregated_embeddings = torch.mean(author_embeddings, dim=0)
            if mode == 'clustering':  # Creates paper cluster (after dimensionality reduction) for each author
                                      # and computes the cosine score for the most similar cluster to the title
                n_clusters = 5
                try:
                    fname_in = fname_base + "{}_{}_{}_{}.csv".format(clustering_type, n_clusters, reduction_type, 3)
                    aggregated_embeddings = np.genfromtxt(fname_in, delimiter=',')
                except:
                    try:
                        author_embeddings_np = read_author_embedding_dict(author)
                        author_embeddings = torch.tensor(author_embeddings_np)
                        if not author_embeddings_np.size: continue  # No publications found for this author
                        cluster_centroids = embeddings_clustering(author_embeddings, type=clustering_type,
                                                                      reduction_type=reduction_type,
                                                                      n_clusters=n_clusters)
                        aggregated_embeddings = torch.Tensor(np.matrix(cluster_centroids)).double()
                    except:
                        print("Except")
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


def main_ranking_authors(fname, titles, descriptions, authors_targets, authors_targets_standby, ranking_mode, clustering_type, reduction_type, csd_in):

    for i, title in enumerate(titles):
        res = rank_candidates(fname=fname,
                              title=title,
                              description=descriptions[i],
                              mode=ranking_mode,
                              clustering_type=clustering_type,
                              reduction_type=reduction_type)

        # Format Csv Name
        in_or_out = 'in' if csd_in else 'out'
        my_mkdir('./specter_rankings')
        my_mkdir('./specter_rankings/{}'.format(title))
        my_mkdir('./specter_rankings/{}/{}'.format(title, in_or_out))

        fname_output = './specter_rankings/{}/{}/{}'.format(title, in_or_out, ranking_mode)
        if ranking_mode == 'clustering': fname_output += '_{}_{}'.format(clustering_type, reduction_type)

        res_target = find_author_relevance(authors_targets[i], authors_targets_standby[i], res)

        res.to_csv(fname_output + '.csv', encoding='utf-8', index=False)
        res_target.to_csv('{}_target.csv'.format(fname_output), encoding='utf-8', index=False)

    
if __name__ == '__main__':

    ##### SET PARAMETERS ######
    ranking_mode = 'max_articles'  # Average of N most relevant papers (N=10 by default)
    # ranking_mode = 'mean'          # Average of all paper embeddings (title + abstract, for each paper)
    # ranking_mode = 'clustering'    # Creates paper cluster (after dimensionality reduction) for each author
                                     # and computes the cosine score for the most similar cluster to the title

    clustering_type = 'agglomerative'  # clustering_options = ['agglomerative', 'kmeans', 'dbscan']
    reduction_type = 'PCA'             # reduction_options = ['PCA', 'SVD', 'isomap', 'LLE']
    input_type = 'json'                # csv or json

    ##### SET TITLES ######
    titles = []
    descriptions = []
    authors_targets_in = []
    authors_targets_in_standby = []
    authors_targets_out = []
    authors_targets_out_standby = []
    
    data = open_json(r'.\specter_rankings\test_apella_data.json')
    
    for i in data[:-1]: # ignore last one without target_lists
        titles.append(i.get("title"))
        descriptions.append(i.get("description"))
        authors_targets_in.append(i.get("targets_in"))
        authors_targets_in_standby.append(i.get("targets_in_standby"))
        authors_targets_out.append(i.get("targets_out"))
        authors_targets_out_standby.append(i.get("targets_out_standby"))
          
    for i, title in enumerate(titles):
        create_position_object(title, descriptions[i], 
                               targets_in=authors_targets_in[i],
                               targets_in_standby=authors_targets_in_standby[i],
                               targets_out=authors_targets_out[i],
                               targets_out_standby=authors_targets_out_standby[i])


    ##### CALCULATE RANKINGS ######
    fname_in = r'..\json_files\csd_in_with_abstract\csd_in_specter_no_greeks.json'
    fname_out = r'..\json_files\csd_out_with_abstract\csd_out_specter.json'

    main_ranking_authors(fname_in, titles, descriptions, authors_targets_in, authors_targets_in_standby, ranking_mode, clustering_type, reduction_type, csd_in=True)
    main_ranking_authors(fname_out, titles, descriptions, authors_targets_out, authors_targets_out_standby, ranking_mode, clustering_type, reduction_type, csd_in=False)
