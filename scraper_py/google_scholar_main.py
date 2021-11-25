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
import time
import pandas as pd
import my_time as mt
from scholarly import scholarly, ProxyGenerator

# csd_in = pd.read_excel(r"..\csv_files\csd_data_in.xlsx")
# csd_out = pd.read_excel(r"..\csv_files\csd_data_out.xlsx")

# preprocessing
# csd_in
# csd_in = df_to_dict_parser(csd_in)
# csd_out = df_to_dict_parser(csd_out)

# find author name in google scholar
# for i in range(len(csd_out)): # range(len(csd_out))
#     get_scholar_name(csd_out[i])
    
# save in csv
# df = pd.DataFrame.from_records(csd_out)
# df.to_csv(path_or_buf=r'..\csv_files\csd_data_out_processed.csv', index=False)

# compare results with ground truth
# df = pd.read_csv(r'..\csv_files\csd_data_out_processed.csv')
# df_ground_truth = pd.read_csv(r'..\csv_files\csd_data_out_processed_ground_truth.csv')
# matched = df[["Scholar id"]] == df_ground_truth[["Scholar id"]]

# matched
# print(matched.value_counts())

csd_in_list_dict = pd.read_csv(r"..\csv_files\csd_data_in_processed_ground_truth.csv").to_dict(orient="records")
csd_out_list_dict = pd.read_csv(r"..\csv_files\csd_data_out_processed_ground_truth.csv").to_dict(orient="records")


# for professor in csd_out_list_dict:
#     try:
#         pg = ProxyGenerator()
#         success = pg.SingleProxy(http = "http://kartzafos22:1gnsjksaDs6FkTGT@proxy.packetstream.io:31112")
#         scholarly.use_proxy(pg)
#         paper_scraper(professor, threads_num=25, host="localhost:27017", json_file_path=r"..\json_files")
#     except Exception as error:
#         print("There is a problem in paper_scraper")
#         print(error)

# open saved json file as dictionary
# with open(r'..\json_files\csd_in_with_abstract\csd_in_with_abstracs_db.json', encoding="utf8") as json_file:
#     csd_out_list_dict = json.load(json_file)

# save dictionary as json file
# json_file = json.dumps(author, indent=4)
# json_name = author["name"].replace(" ", "_").replace(r"/", "_")
# with open(fr'..\json_files\{json_name}.json', 'w', encoding='utf-8') as f:
#     f.write(f"[{json_file}]")









    