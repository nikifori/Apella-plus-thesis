import numpy as np
from itertools import islice
import ijson

from metrics import *
from embeddings.sentence_transformer_models import get_embedding, get_scibert_model, create_author_embeddings
from embeddings.metrics import find_author_relevance
from embeddings.utils import find_author_rank, get_underscored_name, mkdirs, get_positions
from embeddings.sentence_embeddings2 import calculate_similarities


def rank_candidates(fname, in_or_out, title, description, position_rank, mode, clustering_params:dict):
    model, tokenizer = get_scibert_model()
    model_type = 'scibert_average'
    title_embedding = get_embedding(title + tokenizer.sep_token + description, model, tokenizer)

    similarity_score = []
    roman_name = []

    with open(fname, encoding='utf-8') as f:
        objects = ijson.items(f, 'item')
        objects = islice(objects, 500)
        for i, author in enumerate(objects):
            print(f"{i}.",author['romanize name'])
            auth_underscore_name = get_underscored_name(author['romanize name'])
            author_rank = find_author_rank(author)
            sim_val = 0.0

            if author_rank <= position_rank:
                try:
                    fname_emb = fr"./author_embeddings/{model_type}_embeddings/{in_or_out}/{auth_underscore_name}.csv"
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
