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

csd_in = pd.read_excel(r"..\csv_files\csd_data_in.xlsx")
csd_out = pd.read_excel(r"..\csv_files\csd_data_out.xlsx")

# preprocessing
# csd_in
csd_in = df_to_dict_parser(csd_in)
csd_out = df_to_dict_parser(csd_out)

# find author name in google scholar
for i in range(5): # range(len(csd_out))
    time.sleep(1)
    get_scholar_name(csd_out[i])

# save in csv
df = pd.DataFrame.from_records(csd_out)
df.to_csv(path_or_buf=r'..\csv_files\csd_data_out_processed.csv', index=False)
    

