'''
Filename: e:\GitHub_clones\Apella_plus_thesis\python_files\google_scholar_crawler.py
Path: e:\GitHub_clones\Apella_plus_thesis\python_files
Created Date: Saturday, November 6th 2021, 12:34:16 pm
Author: nikifori

Copyright (c) 2021 Your Company
'''


# import numpy as np
# from bs4 import BeautifulSoup
# import requests
# import json
import pandas as pd
from scholarly import scholarly, ProxyGenerator
import my_time as mt
import threading
import math


def paper_scraper(author_name, abstract=False, threads_num=1, make_csv=False):
    
    if type(author_name) is not str:
        print(f"Author name must be string: {type(author_name)} given." )
        return 

    if type(abstract) is not bool:
         print(f"Abstract must be boolean: {type(abstract)} given.")
         return
    
    if type(threads_num) is not int:
         print(f"Threads must be int: {type(threads_num)} given.")
         return
     
    threads = []
    search_query = scholarly.search_author(author_name)
    author = next(search_query)     # object
    scholarly.fill(author, sections=["publications"])
    paper_list = author["publications"]
    # threads_num = threads_num if len(paper_list)%threads_num==0 else threads_num+1
     
    if abstract:
        global result_list
        result_list = []
        chunked_list = chunks(paper_list, threads_num)
        for chunk in chunked_list:
            x = threading.Thread(target=paper_filler, args=(chunk, result_list))
            threads.append(x)
            x.start()
        
        for thread in threads:
            thread.join()
        
        final_df = pd.concat(result_list)
        
        # check abstract entirety
        final_df = df_abstract_entirety(final_df)
        if make_csv:
            csv_creator(author_name, final_df, abstract)
            
        return final_df
    
    else:
        n=0
        df_papers = pd.DataFrame(columns=["Title", "Publication Year"])
        for paper in author["publications"]:
            n += 1
            print(n)
            new_paper = {}
            try:
                new_paper["Title"] = paper["bib"]["title"] if "title" in paper["bib"] else "Unknown"
                new_paper["Publication Year"] = int(paper["bib"]["pub_year"]) if "pub_year" in paper["bib"] else 0
                df_papers = df_papers.append(new_paper, ignore_index=True)
            except:
                print("There is a problem")
        
        if make_csv:
            csv_creator(author_name, df_papers, abstract)
    
        return df_papers
    
def chunks(paper_list, threads):
    chunked_list = []
    step_size = math.ceil(len(paper_list)/threads)
    for i in range(0, len(paper_list), step_size):
        chunked_list.append(paper_list[i:step_size+i])
    
    return chunked_list
        
def paper_filler(chunk_of_papers, result_list):
    df_papers = pd.DataFrame(columns=["Title", "Publication Year", "Publication url", "Abstract"])
    for paper in chunk_of_papers:
        scholarly.fill(paper)
        new_paper = {}
        try:
            new_paper["Title"] = paper["bib"]["title"] if "title" in paper["bib"] else "Unknown"
            new_paper["Publication Year"] = int(paper["bib"]["pub_year"]) if "pub_year" in paper["bib"] else 0
            new_paper["Publication url"] = paper["pub_url"] if "pub_url" in paper else "Unknown"
            new_paper["Abstract"] = paper["bib"]["abstract"] if "abstract" in paper["bib"] else "Unknown"
            print(new_paper["Title"])
            df_papers = df_papers.append(new_paper, ignore_index=True)
        except:
            print("There is a problem")
           
    result_list.append(df_papers)        
    return df_papers

def df_abstract_entirety(final_df):
    temp_list = []
    for abstract in final_df["Abstract"]:
        if abstract[-1]=='…':
            temp_list.append(0)
        else:
            temp_list.append(1)
            
    final_df["Abstract entirety"] = temp_list        
    return final_df

def csv_creator(author_name, final_df, abstract=False):
    paper_name = author_name.replace(" ", "_")
    if abstract:
        final_df.to_csv(path_or_buf=fr'E:\GitHub_clones\Apella-plus-thesis\csv_files\{paper_name}_papers.csv',
                        header=["Title", "Publication Year", "Publication url", "Abstract", "Abstract entirety"],
                        index=False)
    else:
        final_df.to_csv(path_or_buf=fr'E:\GitHub_clones\Apella-plus-thesis\csv_files\{paper_name}_papers.csv',
                        header=["Title", "Publication Year"],
                        index=False)
#-----------------------------------------------------------------------------------------------------------------


pg = ProxyGenerator()
success = pg.SingleProxy(http = "http://kartzafos22:1gnsjksaDs6FkTGT@proxy.packetstream.io:31112")
scholarly.use_proxy(pg)

t = mt.my_time()

t.tic()
test = paper_scraper("Ioannis Partalas", abstract=True, threads_num=20, make_csv=True)
t.toc()












