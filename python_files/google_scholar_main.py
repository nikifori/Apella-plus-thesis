'''
Filename: e:\GitHub_clones\Apella-plus-thesis\python_files\google_scholar_main.py
Path: e:\GitHub_clones\Apella-plus-thesis\python_files
Created Date: Saturday, November 13th 2021, 10:10:49 pm
Author: nikifori

Copyright (c) 2021 Your Company
'''
from csd_csv_parser import *
from google_search import *
from google_scholar_scraper import *
import pandas as pd


def get_names(chunk, res_list):
    for item in chunk:
        new_author = get_scholar_name(item)
        res_list.append(new_author)
        

def google_scholar_names_parser(dict_dat, thread_num):
    if thread_num > len(dict_dat): thread_num = len(dict_dat)
    chunk_list = chunks(dict_dat, threads=thread_num)
    threads = []
    result_list = []

    for chunk in chunk_list:
        x = threading.Thread(target=get_names, args=(chunk, result_list))
        threads.append(x)
        x.start()

    for thread in threads:
        thread.join()

    return result_list


csd_in = pd.read_excel(r"..\csv_files\csd_data_in.xlsx")
csd_out = pd.read_excel(r"..\csv_files\csd_data_out.xlsx")

# preprocessing
csd_in_dict = df_to_dict_parser(csd_in)
csd_out_dict = df_to_dict_parser(csd_out)

csd_in_dict_new = google_scholar_names_parser(csd_in_dict, thread_num=5)

# save in csv
df = pd.DataFrame.from_records(csd_in_dict_new)
df.to_csv(path_or_buf=r'..\csv_files\csd_data_in_processed.csv', index=False)
