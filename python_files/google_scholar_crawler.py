'''
Filename: e:\Projects\theses_tsoumakas\google_scholar_crawler.py
Path: e:\Projects\theses_tsoumakas
Created Date: Sunday, October 31st 2021, 3:42:53 pm
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
        print("Eimai Tasos")
        
    
        
test = papers_title_year("Grigorios Tsoumakas")#.sort_values(by=['Publication Year'],
                                                            # ascending=False)

















