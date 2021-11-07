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
import unidecode


def paper_scraper(author_name="", abstract=0):
    if author_name != "":
        search_query = scholarly.search_author("{}".format(author_name))
        author = next(search_query)     # object
        scholarly.fill(author, sections=["publications"])
        # author_name_without_space = author_name.replace(" ", "_")
        # df_papers = author_name_without_space + "_papers"
        # print(df_papers)
        if abstract==1:
            df_papers = pd.DataFrame(columns=["Title", "Publication Year", "Publication url", "Abstract"])
        else:
            df_papers = pd.DataFrame(columns=["Title", "Publication Year"])
            
        n=0
        if abstract==1:
            for paper in author["publications"]:
                n += 1
                print(n)
                scholarly.fill(paper)
                new_paper = {}
                try:
                    if "title" in paper["bib"]:
                        new_paper["Title"] = paper["bib"]["title"]
                    else: 
                        new_paper["Title"] = "unknown"
                        
                    if "pub_year" in paper["bib"]:
                        new_paper["Publication Year"] = paper["bib"]["pub_year"]
                    else:
                        new_paper["Publication Year"] = "unknown"
                    
                    if "pub_url" in paper:
                        new_paper["Publication url"] = paper["pub_url"]
                    else:
                        new_paper["Publication url"] = "unknown"
                        
                    if "abstract" in paper["bib"]:
                        new_paper["Abstract"] = paper["bib"]["abstract"]
                    else:
                        new_paper["Abstract"] = "unknown"
                        
                    df_papers = df_papers.append(new_paper, ignore_index=True)
                    
                except:
                    print("There is a problem")
                    
            return df_papers
        
        elif abstract==0:
            for paper in author["publications"]:
                n += 1
                print(n)
                new_paper = {}
                try:
                    if "title" in paper["bib"]:
                        new_paper["Title"] = paper["bib"]["title"]
                    else: 
                        new_paper["Title"] = "unknown"
                        
                    if "pub_year" in paper["bib"]:
                        new_paper["Publication Year"] = paper["bib"]["pub_year"]
                    else:
                        new_paper["Publication Year"] = "unknown"
                        
                    df_papers = df_papers.append(new_paper, ignore_index=True)
                    
                except:
                    print("There is a problem")
            
            return df_papers
                    
        else:
            print("Invalid abstract argument")
                    
    else:
        print("no argument")
        
    
        
test = paper_scraper("Grigorios Tsoumakas", abstract=0)#.sort_values(by=['Publication Year'],
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











