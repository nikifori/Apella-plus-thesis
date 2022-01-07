import os.path
import ijson
from itertools import islice

from random import sample
from utils import find_author_rank, mkdirs
import pandas as pd


def average_precision(authors_target: list, authors_target_standby: list, pred_ranking: list):
    true_ranking = authors_target + authors_target_standby
    authors_len = len(true_ranking)

    precision_sum = 0
    authors_sum = 0

    for cc, author in enumerate(pred_ranking):
        if author in true_ranking:
            authors_sum += 1
            precision_sum += authors_sum / (cc + 1)

    averagePre = precision_sum / authors_len
    return averagePre


def top_k_precision(authors_target_all: list, pred_ranking: list, k: int):
    authors_target_len = len(authors_target_all)

    top_k_count = 0
    for i, author in enumerate(pred_ranking):
        if author in authors_target_all and i <= k:
            top_k_count+=1

    top_k_percent = top_k_count / authors_target_len * 100
    return top_k_percent


def sum_of_rankings_metric(authors_target_all: list, pred_ranking: list):
    target_sum = 0
    target_count = 0
    authors_target_len = len(authors_target_all)
    for i, author in enumerate(pred_ranking):
        if author in authors_target_all:
            target_count += 1
            target_sum += i+1

    for i in range(target_count, authors_target_len):
        target_sum += len(pred_ranking)

    return authors_target_len * (authors_target_len+1) / (2 * target_sum)


def prize_metric(authors_target_all: list, pred_ranking: list):

    target_len = len(authors_target_all)
    target_sum = 0
    n = len(pred_ranking)
    best_score = sum([i for i in range(n-target_len+1, n+1)])

    for i, author in enumerate(pred_ranking):
        if author in authors_target_all:
            target_sum += n - (i+1)

    return target_sum / best_score


def reciprocal_rank(authors_target_all: list, pred_ranking: list):
    authors_all_len = len(pred_ranking)

    for i, author in enumerate(pred_ranking):
        if author in authors_target_all:
            return i+1

    return authors_all_len


def coverage_rank(authors_target_all: list, pred_ranking: list):

    coverage = len(pred_ranking)
    target_count=0

    for i, author in enumerate(pred_ranking):
        if author in authors_target_all:
            coverage = i+1
            target_count += 1

    return coverage if target_count == len(authors_target_all) else len(pred_ranking)


def mean(l: list):
    mean_of_list = sum(l) / len(l)
    return mean_of_list


def random_rankings(fname, title, in_or_out, position_rank, authors_target, authors_target_standby):
    total_authors_len = 0
    total_authors_ranked_len = 0
    total_authors_target_len = len(authors_target) + len(authors_target_standby)
    total_trials = 4000

    av_pre = []
    prize_rank = []
    reciprocal = []
    coverage = []
    metric1 = []

    with open(fname, encoding='utf-8') as f:
        objects = ijson.items(f, 'item')
        objects = islice(objects, 500)
        for author in objects:
            author_rank = find_author_rank(author)
            total_authors_len += 1
            if author_rank <= position_rank:
                total_authors_ranked_len += 1

    total_authors_ranked = list(range(total_authors_ranked_len))
    total_authors = list(range(total_authors_len))
    for i in range(total_trials):
        results = sample(total_authors_ranked, total_authors_target_len)
        results.sort()
        av_pre.append(average_precision(authors_target=results, authors_target_standby=[], pred_ranking=total_authors))
        metric1.append(sum_of_rankings_metric(results, total_authors))
        prize_rank.append(prize_metric(results, total_authors))
        reciprocal.append(reciprocal_rank(results, total_authors))
        coverage.append(coverage_rank(results, total_authors))

    print("Random Rankings")
    print(f"Av_pre:{mean(av_pre)}")
    print(f"metric1:{mean(metric1)}")
    print(f"prize_rank:{mean(prize_rank)}")
    print(f"reciprocal:{mean(reciprocal)}")
    print(f"coverage:{mean(coverage)}")

    store_results(title, "random", in_or_out, mean(metric1), 0, 0, mean(av_pre), mean(prize_rank), mean(reciprocal),
                  mean(coverage))


def find_author_relevance(position_title, version, csd_in, authors_target, authors_target_standby, result):
    result_names = list(result[['Name_roman']].values.flatten())
    target_all = authors_target + authors_target_standby
    k = int(1.5 * len(authors_target) + len(authors_target_standby))
    target_result = []

    if len(authors_target) == 0:
        return pd.DataFrame({'target_result': []})

    for i, result_name in enumerate(result_names):
        if result_name in authors_target:
            rank_str = '{}/{}:{}'.format(i + 1, len(result_names), result_name)
            print(rank_str)
            target_result.append(rank_str)
        elif result_name in authors_target_standby:
            rank_str = '{}/{} ({}):{}'.format(i + 1, len(result_names), "standby", result_name)
            print(rank_str)
            target_result.append(rank_str)

    srm = sum_of_rankings_metric(target_all, result_names)
    top_k_perc = top_k_precision(target_all, result_names, k=k)
    average_precision_val = average_precision(authors_target, authors_target_standby, result_names)
    prize_val = prize_metric(target_all, result_names)
    reciprocal = reciprocal_rank(target_all, result_names)
    coverage = coverage_rank(target_all, result_names)
    results_str = f"Metric1:{srm}, top_{k}={top_k_perc}%, Average Precision:{average_precision_val}, Prize_metric:{prize_val}, Reciprocal Rank:{reciprocal}, Coverage:{coverage}"
    print(results_str)
    target_result.append(results_str)
    store_results(position_title, version, csd_in, srm, top_k_perc, k, average_precision_val, prize_val, reciprocal, coverage)
    return pd.DataFrame({'target_result': target_result})


def store_results(position_title, version, csd_in, srm, top_k, k, average_precision_val, prize_val, reciprocal, coverage):
    mkdirs(f"./results/specter/{position_title}")
    fname_in = f"./results/specter/{position_title}/{position_title}_{csd_in}.csv"

    row_names = ['Metric1', f'top_{k}', 'Average_Precision', 'prize_metric', 'Reciprocal_rank', 'Coverage']
    vals = [round(srm,3), round(top_k/100,3), round(average_precision_val,3), round(prize_val,3), reciprocal, coverage]

    if not os.path.exists(fname_in):
        df = pd.DataFrame(vals, index=row_names, columns=[version])
        df.to_csv(fname_in)
        return

    s = pd.Series(vals, index=row_names)
    df = pd.read_csv(fname_in, index_col=0, header=0)

    if version in df.columns:
        df[version] = s
    else:
        s.name = version
        df = pd.concat([df, s], axis=1)

    df.to_csv(fname_in)


def print_sorted_metrics(titles, metric='Average_Precision', ascending=False, in_or_out='out'):

    for i, title in enumerate(titles):
        fname = f'./results/specter/{title}/{title}_{in_or_out}.csv'
        res_temp = pd.read_csv(fname, header=0, index_col=0)
        if i == 0:
            df = res_temp.transpose()
        else:
            df += res_temp.transpose()

    res = df/len(titles)
    res = res.sort_values(by=[metric], ascending=[ascending])
    res = res.round(decimals=4)
    print(res[['Average_Precision','prize_metric','Metric1','Reciprocal_rank','Coverage']].to_string())

    res[['Average_Precision','prize_metric','Metric1','Reciprocal_rank','Coverage']].to_csv(f"./results/{in_or_out}_sort_by_{metric}.csv")


def print_final_results(titles):
    metrics = ['Average_Precision', 'prize_metric', 'Metric1', 'Reciprocal_rank', 'Coverage']
    ascendings = [False, False, False, True, True]

    for i, metric in enumerate(metrics):
        print_sorted_metrics(titles, metric=metric, ascending=ascendings[i], in_or_out='in')
        print_sorted_metrics(titles, metric=metric, ascending=ascendings[i], in_or_out='out')

