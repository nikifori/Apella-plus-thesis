import math
import json
from os.path import splitext


# Returns paper list in batches of 20 papers to reduce RAM consumption
# FIXED: IT IS NOT NEEDED
def batching(papers):
    n = len(papers)
    batch = []
    batch_size = 3

    if n > batch_size:
        for i in range(0, n, batch_size):
            batch.append(papers[i:min(i + batch_size, n)])
    else:
        batch = [papers]

    return batch


def chunks(authors_dict, threads):
    chunked_list = []
    n = len(authors_dict)
    for i in range(threads):
        start = int(math.floor(i * n / threads))
        finish = int(math.floor((i + 1) * n / threads) - 1)
        chunked_list.append(authors_dict[start:(finish + 1)])

    return chunked_list


def read_authors(fname):
    _, file_extension = splitext(fname)
    if file_extension == '.json':
        with open(fname, encoding="utf8") as json_file:
            auth_dict = json.load(json_file)
    return auth_dict


def sbert_check_test():
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    sentences1 = ['The cat sits outside',
                  'A man is playing guitar',
                  'The new movie is awesome']

    sentences2 = ['The dog plays in the garden',
                  'A woman watches TV',
                  'The new movie is so great']

    # Compute embedding for both lists
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)

    # Compute cosine-similarits
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    # Output the pairs with their score
    for i in range(len(sentences1)):
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
