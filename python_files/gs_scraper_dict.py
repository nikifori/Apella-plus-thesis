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
import pymongo
import json


def paper_scraper(author_name, threads_num=1):
    
    
    if type(author_name) is not str:
        print(f"Author name must be string: {type(author_name)} given." )
        return 
    
    if type(threads_num) is not int:
         print(f"Threads must be int: {type(threads_num)} given.")
         return
     
        
    threads = []
    search_query = scholarly.search_author(author_name)
    author = next(search_query)     # object
    author_dict = {"name": author["name"],
                   "affiliation": author["affiliation"],
                   "citedby": author["citedby"],
                   "interests": author["interests"],
                   "scholar_url": f'https://scholar.google.com/citations?user={author["scholar_id"]}&hl=en',
                   "publications": []
                   }
    scholarly.fill(author, sections=["publications"])
    paper_list = author["publications"]
    
     
    global result_list
    result_list = []
    # papers always >= threads
    if threads_num>len(paper_list):threads_num=len(paper_list) # papers always
    print(threads_num)
    chunked_list = chunks(paper_list, threads_num)
    for chunk in chunked_list:
        x = threading.Thread(target=paper_filler, args=(chunk, result_list))
        threads.append(x)
        x.start()
    
    for thread in threads:
        thread.join()
    
    author_dict["publications"] = result_list.copy()
    return author_dict
    
    
def chunks(paper_list, threads):
    chunked_list = []
    n = len(paper_list)
    for i in range(threads):
       start = int(math.floor(i * n / threads))
       finish = int(math.floor((i + 1) * n / threads) - 1)
       chunked_list.append(paper_list[start:(finish+1)])
       
    return chunked_list
        
def paper_filler(chunk_of_papers, result_list):
    for paper in chunk_of_papers:
        scholarly.fill(paper)
        new_paper = {}
        try:
            new_paper["Title"] = paper["bib"]["title"] if "title" in paper["bib"] else "Unknown"
            new_paper["Publication year"] = int(paper["bib"]["pub_year"]) if "pub_year" in paper["bib"] else 0
            new_paper["Publication url"] = paper["pub_url"] if "pub_url" in paper else "Unknown"
            new_paper["Abstract"] = paper["bib"]["abstract"] if "abstract" in paper["bib"] else "Unknown"
            new_paper["Author pub id"] = paper["author_pub_id"] if "author_pub_id" in paper else "Unknown"
            new_paper["Publisher"] = paper["bib"]["publisher"] if "publisher" in paper["bib"] else "Unknown"
            print(new_paper["Title"])
            result_list.append(new_paper)
        except:
            print("There is a problem")
           

#-----------------------------------------------------------------------------------------------------------------


pg = ProxyGenerator()
success = pg.SingleProxy(http = "http://kartzafos22:1gnsjksaDs6FkTGT@proxy.packetstream.io:31112")
scholarly.use_proxy(pg)

t = mt.my_time()

t.tic()
test = paper_scraper("Dimitris Floros", threads_num=20)
# connect to MongoDB local
myclient = pymongo.MongoClient("localhost:27017")
db = myclient.ApellaDB
collection = db.author
collection.insert_one(test)
# save in json file local
json_file = json.dumps(test, indent=4)
test.pop("_id")
json_name = test["name"].replace(" ", "_")
with open(fr'E:\GitHub_clones\Apella-plus-thesis\json_files\{json_name}.json', 'w', encoding='utf-8') as f:
    f.write(f"[{json.dumps(test, indent=4)}]")
t.toc()

