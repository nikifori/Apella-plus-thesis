import torch
import numpy as np
import kneed
import os.path
import re
import ijson
from itertools import islice

from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from nltk.cluster import cosine_distance, KMeansClusterer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import Isomap
from k_means_constrained import KMeansConstrained

from utils import open_json, save2json, mkdirs, get_underscored_name


def constrained_clustering(embeddings, n_samples: int, n_clusters: int):
    n_clusters_original = n_clusters
    while n_samples/n_clusters < 10:
        n_clusters-=1
        if n_clusters == 0: break

    if n_clusters <= 0:
        n_clusters = 1
        y_predict = []
        for i in range(n_samples):
            y_predict.append(0)
    else:
        clf = KMeansConstrained(n_clusters=n_clusters,
                                size_min=10,
                                random_state=0)
        y_predict = clf.fit_predict(embeddings)

    print(f"Constrained K-means Clustering, X:{n_samples} samples, n_clusters_targ:{n_clusters_original}, n_clusters:{n_clusters}")
    return np.array(y_predict)


def get_clusters(X, y_predict):
    clusters = []
    n_clusters = len(np.unique(y_predict))

    for cl in range(0, n_clusters):
        filt = [True if i == cl else False for i in y_predict]
        clusters.append(X[filt])

    return clusters


def reduce_dimensions(X, n_dims=5, type='PCA'):
    types = ['PCA', 'SVD', 'isomap']

    if type == 'PCA': reducer = PCA(n_components=n_dims)
    elif type == 'SVD': reducer = TruncatedSVD(n_dims)
    elif type == 'isomap': reducer = Isomap(n_components=n_dims)
    else:
        print(f"------ERROR-----, incorrect reduce dimension type, accepted values are {types}")
        return [], []

    X_hat = reducer.fit_transform(X)
    return X_hat, reducer


def embeddings_clustering(embeddings_input: np.array,
                          type='agglomerative',
                          reduction_type='PCA',
                          n_clusters=5,
                          dimensions_reduced=5):
    clustering_types = ['agglomerative', 'kmeans', 'spectral', 'agglomerative_cosine',
                        'kmeans_cosine', 'kmeans_euclidean_constr']
    dim_reduction_types = ['PCA', 'SVD', 'isomap']
    author_centroids = []

    # dimensions_reduced = 13
    min_samples = 5  # It must be dimensions_reduced < min_samples
    n_samples = embeddings_input.size(0)

    if dimensions_reduced == 768: embeddings = embeddings_input # No dim reduction
    elif n_samples > min_samples and n_samples > dimensions_reduced:
        embeddings,_ = reduce_dimensions(embeddings_input,
                                       n_dims=dimensions_reduced,
                                       type=reduction_type) \
            if embeddings_input.shape[1] > dimensions_reduced \
            else embeddings_input
    else:  # Too few samples
        embeddings = embeddings_input
        if n_samples <= n_clusters:
            for i in range(n_samples):
                author_centroids.append(embeddings[i, :].detach().cpu().numpy())
            return np.mean(author_centroids, axis=0)

    # Perform Clustering
    if type == 'agglomerative':             y_predict = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
    elif type == 'agglomerative_cosine':    y_predict = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='average').fit_predict(embeddings)
    elif type == 'kmeans':                  y_predict = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(embeddings)
    elif type == 'kmeans_cosine':
        vectors = [embeddings[i,:] for i in range(embeddings.shape[0])]
        clusterer = KMeansClusterer(n_clusters, cosine_distance, avoid_empty_clusters=True)
        y_predict = clusterer.cluster(vectors, True, trace=False)
    elif type == 'spectral':                y_predict = SpectralClustering(n_clusters=n_clusters).fit_predict(embeddings)
    elif type == 'kmeans_euclidean_constr': y_predict = constrained_clustering(embeddings, n_samples, n_clusters)
    else:
        print("------ERROR-----")
        print("Give correct type argument on embeddings_clustering()!!")
        print("Accepted values are: {}".format(clustering_types))
        return []

    clusters = get_clusters(embeddings_input, y_predict)

    author_centroids = []
    for cluster in clusters:
        centroid = cluster.mean(axis=0)
        author_centroids.append(centroid.detach().cpu().numpy())

    return author_centroids


def find_knee(y,x="", curve="", direction=""):
    x = range(len(y)) if x == "" else x
    kneedle = kneed.KneeLocator(x, y, S=1.0, curve=curve, direction=direction)
    return kneedle.knee, kneedle.knee_y


def embeddings_clustering_finder(author_name ,embeddings_input: np.array,
                                 in_or_out="in",
                                 type='kmeans',
                                 reduction_type='PCA'):

    max_dim = 10 if embeddings_input.shape[0] > 10 else embeddings_input.shape[0]

    if reduction_type == 'PCA' or reduction_type == 'SVD':
        embeddings, reducer = reduce_dimensions(torch.tensor(embeddings_input), n_dims=max_dim, type=reduction_type) \
                              if embeddings_input.shape[1] > max_dim else embeddings_input
        vals = reducer.explained_variance_ratio_.cumsum()
        knee, _ = find_knee(vals,range(max_dim), curve="concave", direction="increasing")
    elif reduction_type == 'isomap':
        dims_reduced = range(2, max_dim)
        vals, dims = ([], [])
        for d in dims_reduced:
            embeddings, reducer = reduce_dimensions(torch.tensor(embeddings_input), n_dims=d, type='isomap') \
                                  if embeddings_input.shape[1] > d else embeddings_input
            if not reducer: continue
            vals.append(reducer.reconstruction_error())
            dims.append(d)
        knee, _ = find_knee(vals, dims, curve="convex", direction="decreasing")

    embeddings,_ = reduce_dimensions(torch.tensor(embeddings_input), n_dims=knee, type=reduction_type) \
        if embeddings_input.shape[1] > knee else torch.tensor(embeddings_input)

    if not type == "mean": print("Method:{}, knee dimension:{}".format(reduction_type, knee))
    else: reduction_type = "avg"

    # Perform Clustering
    n_clusters = range(2,7)
    sil_scores = []
    clusters_all = []

    for cl in n_clusters:
        if type == 'agglomerative': y_predict = AgglomerativeClustering(n_clusters=cl).fit_predict(embeddings)
        elif type == 'agglomerative_cosine': y_predict = AgglomerativeClustering(n_clusters=cl, affinity='cosine', linkage='average').fit_predict(embeddings)
        elif type == 'kmeans': y_predict = KMeans(n_clusters=cl, random_state=0).fit_predict(embeddings)
        elif type == 'kmeans_cosine':
            vectors = [embeddings[i, :] for i in range(embeddings.shape[0])]
            clusterer = KMeansClusterer(cl, cosine_distance, avoid_empty_clusters=True)
            y_predict = clusterer.cluster(vectors, True, trace=False)
        else:
            sil_scores = [0]
            n_clusters = [1]
            clusters_all = [np.mean(embeddings_input, axis=0).tolist()]

            break
        sil_score = silhouette_score(embeddings, y_predict)
        sil_scores.append(sil_score)

        clusters = get_clusters(embeddings_input, y_predict)
        author_centroids = []
        for cluster in clusters:
            centroid = cluster.mean(axis=0)
            centroid = centroid.tolist()[0]
            author_centroids.append([float(i) for i in centroid])
        clusters_all.append(author_centroids)

    max_ind = max(range(len(sil_scores)), key=sil_scores.__getitem__)
    print("Method:{}, max silhouette score:{} for {} clusters\n".format(type, max(sil_scores), n_clusters[max_ind]))
    auth_underscore_name = get_underscored_name(author_name)
    path_name = fr"./author_embeddings/specter_embeddings/aggregations/{in_or_out}/{auth_underscore_name}"
    mkdirs(path_name)

    clustering_obj = {
                        'clusters': clusters_all[max_ind],
                        'dims_reduced': int(knee),
                        'n_clusters': n_clusters[max_ind],
                        'silhuette_score': max(sil_scores)
                     }

    if os.path.exists(path_name + "/aggregations.json"):
        aggregations = open_json(path_name + "/aggregations.json")

        if type not in aggregations: aggregations[type] = {f'{reduction_type}': clustering_obj}
        elif reduction_type not in aggregations[type]: aggregations[type][reduction_type] = clustering_obj
        save2json(aggregations, path_name + "/aggregations.json")
    else:
        aggregations = {f'{type}': {f'{reduction_type}': clustering_obj}}
        aggregations[type][reduction_type] = clustering_obj
        save2json(aggregations, path_name + "/aggregations.json")


if __name__ == "__main__":
    in_or_out="out"
    clustering_types = ['agglomerative','kmeans','agglomerative_cosine','kmeans_cosine']
    reduction_types  = ['PCA', 'isomap']

    fname_in = r'..\json_files\csd_in_with_abstract\csd_in_specter.json'
    fname_out = r'..\json_files\csd_out_with_abstract\csd_out_specter_out.json'
    fname = fname_in if in_or_out == "in" else fname_out

    # create clustering aggregations
    with open(fname, encoding='utf-8') as f:
        authors = ijson.items(f, 'item')
        authors = islice(authors, 500)
        for author in authors:
            author_embedding = []
            try:
                print("Author:{}, Publications:{}".format(author['romanize name'], len(author['Publications'])))
                for pub in author['Publications']:
                    if "Specter embedding" in pub:
                        specter_emb = np.array([float(f) for f in pub['Specter embedding'][0]])
                        author_embedding.append(specter_emb)
                    else:
                        print("SPECTER EMBEDDING IS MISSING FROM AUTHOR{}".format(author['Name']))
                        author_embedding = np.empty()
                        break
                author_embedding = np.matrix(author_embedding)
                for clustering_type in clustering_types:
                    for reduction_type in reduction_types:
                        embeddings_clustering_finder(author['romanize name'], author_embedding, in_or_out=in_or_out,
                                                     type=clustering_type, reduction_type=reduction_type)
            except Exception as e:
                print(e)
