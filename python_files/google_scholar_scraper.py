'''
Filename: e:\GitHub_clones\Apella_plus_thesis\python_files\google_scholar_crawler.py
Path: e:\GitHub_clones\Apella_plus_thesis\python_files
Created Date: Saturday, November 6th 2021, 12:34:16 pm
Author: nikifori, bill

Copyright (c) 2021 Your Company
'''


import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import json
from scholarly import scholarly, ProxyGenerator
import my_time as mt

t = mt.my_time()


# pg = ProxyGenerator()
# success = pg.SingleProxy(http = "http://kartzafos22:1gnsjksaDs6FkTGT@proxy.packetstream.io:31112")
# scholarly.use_proxy(pg)

def paper_scraper(author_name, abstract=False):
    
    if type(author_name) is not str:
        print(f"Author name must be string: {type(author_name)} given." )
        return 

    if type(abstract) is not bool:
         print(f"Abstract must be boolean: {type(abstract)} given.")
         return
    
    search_query = scholarly.search_author(author_name)
    author = next(search_query)     # object
    t.tic()
    scholarly.fill(author, sections=["publications"])
    t.toc()
    # author_name_without_space = author_name.replace(" ", "_")
    # df_papers = author_name_without_space + "_papers"
    # print(df_papers) 
    n=0
    if abstract:
        df_papers = pd.DataFrame(columns=["Title", "Publication Year", "Publication url", "Abstract"])
        for paper in author["publications"]:
            n += 1
            print(n)
            scholarly.fill(paper)
            new_paper = {}
            try:
                new_paper["Title"] = paper["bib"]["title"] if "title" in paper["bib"] else None
                new_paper["Publication Year"] = int(paper["bib"]["pub_year"]) if "pub_year" in paper["bib"] else None
                new_paper["Publication url"] = paper["pub_url"] if "pub_url" in paper else None
                new_paper["Abstract"] = paper["bib"]["abstract"] if "abstract" in paper["bib"] else None
                df_papers = df_papers.append(new_paper, ignore_index=True) 
            except:
                print("There is a problem")
                
        return df_papers
    
    else:
        df_papers = pd.DataFrame(columns=["Title", "Publication Year"])
        for paper in author["publications"]:
            n += 1
            print(n)
            new_paper = {}
            try:
                new_paper["Title"] = paper["bib"]["title"] if "title" in paper["bib"] else None
                new_paper["Publication Year"] = int(paper["bib"]["pub_year"]) if "pub_year" in paper["bib"] else None
                df_papers = df_papers.append(new_paper, ignore_index=True)
            except:
                print("There is a problem")
        
        return df_papers
                    
        
    
        
test = paper_scraper("Grigorios Tsoumakas", abstract=False)#.sort_values(by=['Publication Year'],
                                                        # ascending=False)

# abstract_entirety = pd.DataFrame(columns=["Abstract entirety"])
# temp_list = []
# for abstract in test["Abstract"]:
#     if abstract[-1]=='…':
#         temp_list.append(0)
#     else:
#         temp_list.append(1)

# test["Abstract entirety"] = temp_list

# test.to_csv(path_or_buf=r'E:\GitHub_clones\Apella-plus-thesis\tsoumakas_papers.csv', 
#             header=["Title", "Publication Year", "Publication url", "Abstract", "Abstract entirety"], 
#             index=False) 











