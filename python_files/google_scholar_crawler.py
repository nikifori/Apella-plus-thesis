'''
Filename: e:\GitHub_clones\Apella_plus_thesis\python_files\google_scholar_crawler.py
Path: e:\GitHub_clones\Apella_plus_thesis\python_files
Created Date: Saturday, November 6th 2021, 12:34:16 pm
Author: nikifori

Copyright (c) 2021 Your Company
'''


import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import json
from scholarly import scholarly


def papers_title_year(author_name=""):
    if author_name != "":
        search_query = scholarly.search_author("{}".format(author_name))
        author = next(search_query)     # object
        scholarly.fill(author, sections=["publications"])
        # author_name_without_space = author_name.replace(" ", "_")
        # df_name = author_name_without_space + "_papers"
        # print(df_name)
        df_name = pd.DataFrame(columns=["Title", "Publication Year"])
        
        for paper in author["publications"]:
            try:
                new_paper = {"Title": paper["bib"]["title"],
                             "Publication Year": paper["bib"]["pub_year"]}
                df_name = df_name.append(new_paper, ignore_index=True)
            except:
                new_paper = {"Title": paper["bib"]["title"],
                             "Publication Year": "unknown"}
                df_name = df_name.append(new_paper, ignore_index=True)
        
        return df_name
    else:
        print("no argument")
        
    
        
test = papers_title_year("Grigorios Tsoumakas")#.sort_values(by=['Publication Year'],
                                                            # ascending=False)

print(test.head())
















