import pandas as pd
from google_search import get_scholar_name
from google_scholar_scraper import *


def get_names(chunk, res_list):
    sublist = []
    for item in chunk:
        new_author = get_scholar_name(item)
        sublist.append(new_author)

    res_list.append(sublist)


def scholar_names_parser_par(dict_dat, thread_num):
    if thread_num > len(dict_dat): thread_num = len(dict_dat)
    chunk_list = chunks(dict_dat, threads=thread_num)
    threads = []

    global listoflists
    return_list = []
    listoflists = []

    # find author name in google scholar
    for chunk in chunk_list:
        x = threading.Thread(target=get_names, args=(chunk, listoflists))
        threads.append(x)
        x.start()

    for thread in threads:
        thread.join()

    for sublist in listoflists:
        for author in sublist:
            return_list.append(author)

    return return_list


csd_in_dict = pd.read_csv(r"..\csv_files\csd_in_processed_ground_truth.csv").to_dict(orient="records")
csd_out_dict = pd.read_csv(r"..\csv_files\csd_out_processed_ground_truth.csv").to_dict(orient="records")

# Finds author names in parallel (useful for csd_out_dict)
csd_list = scholar_names_parser_par(csd_in_dict, thread_num=5)
