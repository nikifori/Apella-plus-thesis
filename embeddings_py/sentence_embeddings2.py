# Specter Embeddings Csd_in and Csd_out at "https://drive.google.com/drive/folders/1SKbwnNQR3aa94far3JcBtbQ1DKZcVE2v?usp=sharing"
import torch
import numpy as np
import pandas as pd
# from gensim.corpora import Dictionary
from sentence_transformers import SentenceTransformer, util
from pathlib import Path

from sbert_utils import chunks, batching, read_authors
from transformers import AutoTokenizer, AutoModel
# from nltk.corpus import stopwords
# from nltk import download
#
# from pyemd import emd
# from numpy import sum as np_sum
# import math

from emb_clustering import embeddings_clustering


def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    return model(**inputs).last_hidden_state[:, 0, :]


def get_embedding_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    return model, tokenizer


def get_sbert_model(model_name):
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
    title_abs = [paper['Title'] + tokenizer.sep_token + paper['Abstract'] if "Abstract" in paper else paper['Title'] for
                 paper in papers]
    return title_abs


def read_author_embedding(author, csd_in=True):
    subfolder = "/csd_in/" if csd_in else "/csd_out/"

    fname = "author_embeddings" + subfolder + author['romanize name'].replace(" ", "_") + "_embeddings.csv"
    author_embedding = np.genfromtxt(fname, delimiter=',')
    return author_embedding


def rank_candidates(auth_dict, title, description, mode='mean', clustering_type='agglomerative'):
    model, tokenizer = get_specter_model()

    title_embedding = get_embedding(title + tokenizer.sep_token + description, model, tokenizer)

    similarity_score = []
    roman_name = []

    for author in auth_dict:
        try:
            author_embeddings_np = read_author_embedding(author, csd_in=True)    # np.genfromtxt(fname, delimiter=',')
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

#
# ## TODO with word2vec, glove, other static word embeddings
# def rank_candidates_by_wmd_distance(auth_dict, title):
#     # # Generate SciBERT model to create word embeddings
#     bert_model, bert_tokenizer = get_embedding_model('allenai/scibert_scivocab_uncased')
#     download('stopwords')
#     stop_words = stopwords.words('english')
#
#     # Preprocess title sentence and get title word embeddings
#     title = title.lower().split()
#     title_sentence = [w for w in title if w not in stop_words]
#     inputs = bert_tokenizer(' '.join(title_sentence), padding=True, truncation=True, return_tensors="pt",
#                             max_length=512)
#     title_emb = bert_model(**inputs).last_hidden_state[0, :, :].detach().numpy()
#     title_emb = np.delete(title_emb, [0, -1], 0)  # remove first and last row
#
#     authors_names = []
#     authors_distances = []
#
#     # For each author
#     for author in auth_dict:
#         papers_dist = []
#         for paper in author['Publications']:
#             paper_text = paper['Title'] + paper['Abstract']
#
#             paper_sentence = paper_text.lower().split()
#             paper_sentence = [w for w in paper_sentence if w not in stop_words]
#
#             inputs = bert_tokenizer(paper_text, padding=True, truncation=True, return_tensors="pt", max_length=512,
#                                     add_special_tokens=True)
#             paper_emb = bert_model(**inputs).last_hidden_state[0, :, :].detach().numpy()
#             paper_emb = np.delete(paper_emb, [0, -1], 0)  # remove first and last row
#
#             # Calculate Word Mover's Distance between title and paper word embeddings
#             paper_dist = wmdistance(paper_sentence, title_sentence, title_emb, paper_emb)
#             # print(paper_dist)
#
#             papers_dist.append(paper_dist)
#
#         authors_names.append(author['romanize name'])
#         authors_distances.append(mean(papers_dist))
#
#     return pd.DataFrame({'Names_Roman': authors_names,
#                          'Similarity_score': authors_distances})
#
#
# ## TODO with word2vec, glove, etc
# def wmdistance(document1, document2, embeddings1, embeddings2):
#     """Compute the Word Mover's Distance between two documents.
#     When using this code, please consider citing the following papers:
#     * `Ofir Pele and Michael Werman "A linear time histogram metric for improved SIFT matching"
#       <http://www.cs.huji.ac.il/~werman/Papers/ECCV2008.pdf>`_
#     * `Ofir Pele and Michael Werman "Fast and robust earth mover's distances"
#       <https://ieeexplore.ieee.org/document/5459199/>`_
#     * `Matt Kusner et al. "From Word Embeddings To Document Distances"
#       <http://proceedings.mlr.press/v37/kusnerb15.pdf>`_.
#     Parameters
#     ----------
#     document1 : list of str
#         Input document.
#     document2 : list of str
#         Input document.
#     Returns
#     -------
#     float
#         Word Mover's distance between `document1` and `document2`.
#     Warnings
#     --------
#     This method only works if `pyemd <https://pypi.org/project/pyemd/>`_ is installed.
#     If one of the documents have no words that exist in the vocab, `float('inf')` (i.e. infinity)
#     will be returned.
#     Raises
#     """
#
#     print(document1)
#
#     if not document1 or not document2:
#         print(
#             "At least one of the documents had no words that were in the vocabulary. ",
#             "Aborting (returning inf)."
#         )
#         return float('inf')
#
#     dictionary = Dictionary(documents=[document1, document2])
#     vocab_len = len(dictionary)
#     print("vocab_len:", vocab_len)
#
#     if vocab_len == 1:
#         # Both documents are composed by a single unique token
#         return 0.0
#
#     # Sets for faster look-up.
#     docset1 = set(document1)
#     docset2 = set(document2)
#
#     # Compute distance matrix.
#     distance_matrix = np.zeros((vocab_len, vocab_len), dtype=float)
#     for i, t1 in dictionary.items():
#         if t1 not in docset1:
#             continue
#
#         for j, t2 in dictionary.items():
#             if t2 not in docset2 or distance_matrix[i, j] != 0.0:
#                 continue
#
#             print(i, j)
#             print(t1, t2)
#
#             # Compute Euclidean distance between word vectors.
#             # distance_matrix[i, j] = distance_matrix[j, i] = math.sqrt(np_sum((self[t1] - self[t2]) ** 2))
#             distance_matrix[i, j] = distance_matrix[j, i] = math.sqrt(np_sum((embeddings1[i] - embeddings2[j]) ** 2))
#
#     if np_sum(distance_matrix) == 0.0:
#         # `emd` gets stuck if the distance matrix contains only zeros.
#         print('The distance matrix is all zeros. Aborting (returning inf).')
#         return float('inf')
#
#     def nbow(document):
#         d = np.zeros(vocab_len, dtype=float)
#         nbow = dictionary.doc2bow(document)  # Word frequencies.
#         doc_len = len(document)
#         for idx, freq in nbow:
#             d[idx] = freq / float(doc_len)  # Normalized word frequencies.
#         return d
#
#     # Compute nBOW representation of documents.
#     d1 = nbow(document1)
#     d2 = nbow(document2)
#
#     # Compute WMD.
#     return emd(d1, d2, distance_matrix)
#

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
        res = np.linalg.norm(auth1_embedding - auth2_embedding)
        print("Authors Euclidean Distance:{}".format(res))

    # # Find the most relevant paper of Author 2 for each paper Author 1 has
    # cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    # for i in range(len(papers1)):
    #     i_max = max(range(len(cosine_scores[i, :])), key=cosine_scores[i, :].__getitem__)
    #     if cosine_scores[i, i_max] > 0.5:
    #         print("{} \t\t {} \t\t Score: {:.4f}".format(papers1[i], papers2[i_max], cosine_scores[i][i_max]))


def sbert_specter(papers, model=""):
    if model == "":
        model = get_sbert_model('allenai/specter')

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
        emb_total.append(result.last_hidden_state[:, 0, :].tolist()[0])

    pd.DataFrame(np.matrix(emb_total)).to_csv(
        "author_embeddings/" + author['romanize name'].replace(" ", "_") + "_embeddings.csv", header=False, index=False)


if __name__ == '__main__':

    # Get input data - preprocessing
    authors_dict = read_authors(r'..\json_files\csd_in_with_abstract\csd_in_with_abstracs_db.json')
    # authors_dict = read_authors(r'..\json_files\csd_out_with_abstract\csd_out_with_abstracts_db.json')

    # compare_authors_similarity(authors_dict[20], authors_dict[23], type='cosine')

    #######################################################
    # # Calculate Author Embeddings and store them to CSV
    # model, tokenizer = get_specter_model()
    # for author in authors_dict[36:]:
    #     create_author_embeddings(author, model, tokenizer)

    #######################################################
    # Compute Author Rankings with respect to similarity with a title (+description)

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
                              clustering_type=clustering_type)
        res.to_csv('./specter_rankings/{}_in_{}.csv'.format(title, ranking_mode), encoding='utf-8', index=False)
