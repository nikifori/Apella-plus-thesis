# Specter Embeddings Csd_in and Csd_out at "https://drive.google.com/drive/folders/1SKbwnNQR3aa94far3JcBtbQ1DKZcVE2v?usp=sharing"
import os.path

import torch
import numpy as np
import pandas as pd
import json

from sentence_transformers import util
from sbert_utils import read_authors, get_embedding, get_specter_model
from emb_clustering import embeddings_clustering


def create_position_object(title, description, targets_in, targets_out):
    position_dict = {
        "title": title,
        "description": description,
        "targets_in": targets_in,
        "targets_out": targets_out
    }
    with open('./specter_rankings/{}/{}.json'.format(title, title), 'w') as fp:
        json.dump(position_dict, fp, indent=2)


def find_author_relevance(authors_target, result):
    print("Selected Authors ranking:")
    result_names = list(result[['Name_roman']].values.flatten())
    total_authors = result[['Name_roman']].size
    target_result = []

    for i in range(total_authors):
        if result_names[i] in authors_target:
            print('{}/{}:{}'.format(i+1,total_authors,result_names[i]))
            target_result.append('{}/{}:{}'.format(i+1,total_authors,result_names[i]))

    return pd.DataFrame({'target_result':target_result})


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
    # print(author['romanize name'],' ',author_embedding.shape)
    return author_embedding


def rank_candidates(auth_dict, title, description, mode='mean', clustering_type='agglomerative', reduction_type='PCA',csd_in=True,input_type='csv'):
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
            cluster_centroids = embeddings_clustering(author_embeddings, type=clustering_type, reduction_type=reduction_type, n_clusters=n_clusters)
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


def main_ranking_authors(authors_dict, titles, descriptions, authors_targets, ranking_mode, clustering_type, reduction_type, input_type, csd_in):

    for i, title in enumerate(titles):
        res = rank_candidates(auth_dict=authors_dict,
                              title=title,
                              description=descriptions[i],
                              mode=ranking_mode,
                              clustering_type=clustering_type,
                              reduction_type=reduction_type,
                              csd_in=csd_in,
                              input_type=input_type)

        # Format Csv Name
        in_or_out = 'in' if csd_in else 'out'
        if not os.path.exists('./specter_rankings'):
            os.mkdir('./specter_rankings')

        if not os.path.exists('./specter_rankings/{}'.format(title)):
            os.mkdir('./specter_rankings/{}'.format(title))

        if not os.path.exists('./specter_rankings/{}/{}'.format(title, in_or_out)):
            os.mkdir('./specter_rankings/{}/{}'.format(title, in_or_out))

        fname = './specter_rankings/{}/{}/{}'.format(title, in_or_out, ranking_mode)
        if ranking_mode == 'clustering': fname += '_{}_{}'.format(clustering_type,reduction_type)

        res_target = find_author_relevance(authors_targets[i], res)

        res.to_csv(fname + '.csv', encoding='utf-8', index=False)
        res_target.to_csv('{}_target.csv'.format(fname), encoding='utf-8', index=False)


if __name__ == '__main__':

    ##### SET PARAMETERS ######
    # ranking_mode = 'max_articles'  # Average of N most relevant papers (N=10 by default)
    # ranking_mode = 'mean'          # Average of all paper embeddings (title + abstract, for each paper)
    ranking_mode = 'clustering'    # Creates paper cluster (after dimensionality reduction) for each author
                                     # and computes the cosine score for the most similar cluster to the title

    clustering_type = 'agglomerative'  # clustering_options = ['agglomerative', 'kmeans', 'dbscan']
    reduction_type = 'PCA'             # reduction_options = ['PCA', 'SVD', 'isomap', 'LLE']
    input_type = 'json'                # csv or json

    ##### SET TITLES ######
    titles = []
    descriptions = []
    authors_targets_in = []
    authors_targets_out = []

    titles.append('Intelligent Systems - Symbolic Artificial Intelligence')
    descriptions.append('Development of intelligent systems using a combination of methodologies of symbolic Artificial Intelligence, such as Representation of Knowledge and Reasoning, Multiagent Systems, Machine Learning, Intelligent Autonomous Systems, Planning and Scheduling of Actions, Satisfaction')
    authors_targets_in.append(['Nikolaos Vasileiadis', 'Ioannis Vlachavas', 'Dimitrios Vrakas', 'Grigorios Tsoumakas', 'Athina Vakali'])
    authors_targets_out.append([])

    titles.append('Optical Communications')
    descriptions.append('')
    authors_targets_in.append(['Georgios Papadimitriou', 'Amalia Miliou'])
    authors_targets_out.append([])

    titles.append('Theoretical Cryptography')
    descriptions.append('Theoretical Cryptography is the development of cryptographical schemes based on difficult to solve algorithmic problems of number theory, as well as' +
                        'cryptanalytic methods for algorithmic solutions on such problems. Computational number theory is the algorithmic number theory problems solution such as' +
                        'factorisation, discrete logarithm problem, etc, which are used on modern cryptography schemes (RSA, Diffie-Hellman), while also quantity estimations in general,' +
                        'such as bounds for family solutions of diophantine equations, special form primes calculation, recursive sequences period calculations')
    authors_targets_in.append(['Panagiotis Katsaros', 'Anastasios Gounaris ', 'Nikolaos Konofaos', 'Georgios Papadimitriou'])
    authors_targets_out.append([])

    for i,title in enumerate(titles):
        create_position_object(title, descriptions[i], targets_in=authors_targets_in[i], targets_out=authors_targets_out[i])


    ##### CALCULATE RANKINGS ######
    csd_in = True
    authors_dict = read_authors(r'..\json_files\csd_in_with_abstract\csd_in_specter.json')
    main_ranking_authors(authors_dict, titles, descriptions, authors_targets_in, ranking_mode, clustering_type, reduction_type, input_type, csd_in)



    # csd_in = False
    # authors_dict = read_authors(r'..\json_files\csd_out_with_abstract\csd_out_specter_with_410_authors_missing_2.json')
    # main_ranking_authors(authors_dict, titles, descriptions, authors_targets_out, ranking_mode, clustering_type, input_type, csd_in)
