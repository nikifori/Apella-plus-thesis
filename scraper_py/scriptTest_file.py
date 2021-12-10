'''
Filename: e:\GitHub_clones\Apella-plus-thesis\scraper_py\scriptTest_file.py
Path: e:\GitHub_clones\Apella-plus-thesis\scraper_py
Created Date: Friday, December 10th 2021, 12:39:16 am
Author: nikifori

Copyright (c) 2021 Your Company
'''
from csd_csv_parser import *
from google_search import *
from google_scholar_scraper import *
from semantic_scholar_scraper import *
import time
import pandas as pd
import my_time as mt
from scholarly import scholarly, ProxyGenerator

with open(r'..\json_files\csd_in_with_abstract\csd_in_with_abstracts_db.json', encoding="utf8") as json_file:
    csd_in_list_dict = json.load(json_file)
with open(r'..\json_files\csd_out_with_abstract\csd_out_with_abstracts_db_include_unfetch.json', encoding="utf8") as json_file:
    csd_out_list_dict = json.load(json_file)
with open(r'..\json_files\can_not_fetch_complete.json', encoding="utf8") as json_file:
    can_not_fetch = json.load(json_file)

# fix unfetch authors
# cc=0
# for unlist_author in can_not_fetch:
#     for counter, author in enumerate(csd_out_list_dict):
#         if unlist_author["name"]==author["name"]:
#             cc+=1
#             print(cc)
#             csd_out_list_dict[counter] = unlist_author

# save dictionary as json file
# json_file = json.dumps(csd_out_list_dict, indent=4)
# with open(fr'..\json_files\csd_out_with_abstract\csd_out_with_abstracts_db_include_unfetch.json', 'w', encoding='utf-8') as f:
#     f.write(json_file)

# check authors that are not in Google Scholar: 39 authors
# cc = 0
# authors_out_of_GS = []   
# for author in csd_out_list_dict:
#     if "Publications" not in author:
#         authors_out_of_GS.append(author)
#         print(author["name"])
#         cc += 1
#         print(cc)
# print(cc)

# save dictionary as json file
# json_file = json.dumps(authors_out_of_GS, indent=4)
# with open(fr'..\json_files\csd_out_with_abstract\csd_out_authors_out_of_GS.json', 'w', encoding='utf-8') as f:
#     f.write(json_file)

# load auhors out of GS
with open(r'..\json_files\csd_out_with_abstract\csd_out_authors_out_of_GS.json', encoding="utf8") as json_file:
    csd_out_authors_out_of_GS = json.load(json_file)
















