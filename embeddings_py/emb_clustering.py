from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import Isomap, LocallyLinearEmbedding
import numpy as np


def get_clusters(X, y_predict):
    clusters = []
    n_clusters = len(np.unique(y_predict))

    for cl in range(0, n_clusters):
        filt = [True if i == cl else False
                for i in y_predict]
        clusters.append(X[filt])

    return clusters


def reduce_dimensions(X, n_dims=5, type='PCA'):

    types = ['PCA', 'SVD', 'isomap', 'LLE']
    X_hat = []
    reducer = []

    if type == 'PCA':
        pca = PCA(n_components=n_dims)
        pca.fit(X)
        X_hat = pca.transform(X)
        reducer = pca
    elif type == 'SVD':
        truncatedSVD = TruncatedSVD(n_dims)
        X_hat = truncatedSVD.fit_transform(X)
        reducer = truncatedSVD
    elif type == 'isomap':
        isomap = Isomap(n_components=n_dims)
        X_hat= isomap.fit_transform(X)
        reducer = isomap
    elif type == 'LLE':  #Local Linear Embedding
        lle = LocallyLinearEmbedding(n_components=n_dims)
        X_hat = lle.fit_transform(X)
        reducer = lle
    else:
        print("------ERROR-----")
        print("Give correct type argument on reduce_dimensions()!!")
        print("Accepted values are: {}".format(types))

    return X_hat, reducer


def embeddings_clustering(embeddings_input: np.array,
                          type='agglomerative',
                          reduction_type='PCA',
                          n_clusters=5):

    clustering_types = ['agglomerative', 'kmeans','dbscan']
    dim_reduction_types = ['PCA', 'SVD', 'isomap', 'LLE']
    author_centroids = []

    dimensions_reduced = 25  #It must be dimensions_reduced < min_samples
    min_samples = 30
    n_samples = embeddings_input.size(0)

    if n_samples > min_samples:
        embeddings, reducer = reduce_dimensions(embeddings_input,
                                                n_dims=dimensions_reduced,
                                                type=reduction_type) \
            if embeddings_input.shape[1] > dimensions_reduced \
            else embeddings_input
    else:  # Too few samples
        embeddings = embeddings_input
        if n_samples <= n_clusters:
            for i in range(n_samples):
                author_centroids.append(embeddings[i, :].detach().cpu().numpy())
            return author_centroids

    # Perform Clustering
    if type == 'agglomerative':
        y_predict = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(embeddings)
    elif type == 'kmeans':
        y_predict = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(embeddings)
    elif type == 'dbscan':
        y_predict = DBSCAN(eps=0.2, min_samples=5).fit_predict(embeddings)
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


if __name__ == "__main__":
    print("emb_clustering.py")
