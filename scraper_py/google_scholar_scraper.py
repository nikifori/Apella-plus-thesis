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
from scholarly import scholarly, ProxyGenerator
import threading
import math
import pymongo
import json
import operator
from __utils__ import *



def paper_scraper(author_dict, threads_num=1, host=None, json_file_path=None):
    
    
    if type(author_dict) is not dict:
        print(f"Author name must be dict: {type(author_dict)} given." )
        return 
    
    if type(threads_num) is not int:
         print(f"Threads must be int: {type(threads_num)} given.")
         return
     
    if "Publications" not in author_dict:
        if author_dict["Scholar id"]!="Unknown":
            threads = []
            try:
                author = scholarly.search_author_id(author_dict["Scholar id"]) #object
            except Exception as error:
                print(error)
                print("id query does not work")
                try:
                    search_query = scholarly.search_author(author_dict["Scholar name"]) 
                    author = next(search_query) 
                except Exception as error:
                    print(error)
                    print("name query does not work")
                    if host:
                        myclient = pymongo.MongoClient(f"{host}")
                        db = myclient.ApellaDB
                        collection = db.author
                        collection.insert_one(author_dict)  # inserts _id in author dictionary
                    
                    # save in json file local
                    if json_file_path:
                        if host: author_dict.pop("_id")
                        json_file = json.dumps(author_dict, indent=4)
                        json_name = author_dict["name"].replace(" ", "_")
                        with open(fr'{json_file_path}\{json_name}.json', 'w', encoding='utf-8') as f:
                            f.write(f"[{json_file}]")
                            
                    return author_dict
                    
    
            if author["affiliation"]: author_dict["Affiliation"] = author["affiliation"] 
            if author["citedby"]: author_dict["Citedby"] = author["citedby"]
            if author["interests"]: author_dict["Interests"] = author["interests"]
            author_dict["Scholar url"] = f'https://scholar.google.com/citations?user={author_dict["Scholar id"]}&hl=en'
                                          
            scholarly.fill(author, sections=["publications"])
            # sort list of papers based on pub_year
            for paper in author["publications"]:
                paper["pub_year"] = int(paper["bib"]["pub_year"]) if "pub_year" in paper["bib"] else 0
                
            author["publications"].sort(key=operator.itemgetter("pub_year"), reverse=True) 
            paper_list = []
            for paper in author["publications"]:
                if paper["pub_year"]>=2000: paper_list.append(paper)
            
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
            
            author_dict["Publications"] = result_list.copy()
        
        # connect to MongoDB local
        if host:
            myclient = pymongo.MongoClient(f"{host}")
            db = myclient.ApellaDB
            collection = db.author
            collection.insert_one(author_dict)  # inserts _id in author dictionary
        
        # save in json file local
        if json_file_path:
            if host: author_dict.pop("_id")
            json_file = json.dumps(author_dict, indent=4)
            json_name = author_dict["name"].replace(" ", "_").replace(r"/", "_")
            with open(fr'{json_file_path}\{json_name}.json', 'w', encoding='utf-8') as f:
                f.write(f"[{json_file}]")
            
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
            new_paper["Abstract entirety"] = 0 if new_paper["Abstract"][-1]=='…' else 1
            new_paper["Author pub id"] = paper["author_pub_id"] if "author_pub_id" in paper else "Unknown"
            new_paper["Publisher"] = paper["bib"]["publisher"] if "publisher" in paper["bib"] else "Unknown"
            print(new_paper["Title"])
            result_list.append(new_paper)
            
        except Exception as error:
            print("There is a problem")
            print(error)
         
            

def paper_scraper_extra(author_dict: dict, threads_num=1):
    global result_list
    result_list = []
    threads = []
    
    if threads_num>len(author_dict['unscraped papers']):threads_num=len(author_dict['unscraped papers']) # papers always
    print(threads_num)
    print(author_dict.get('romanize name'))
    
    chunked_list = chunks(author_dict['unscraped papers'], threads_num)
    
    for chunk in chunked_list:
        x = threading.Thread(target=paper_filler_extra, args=(chunk, result_list))
        threads.append(x)
        x.start()
    
    for thread in threads:
        thread.join()
    
    if 'Publications' not in author_dict:
        author_dict["Publications"] = result_list.copy()
    else:
        author_dict['Publications'] + result_list
    
    return author_dict
            
    
def paper_filler_extra(chunk_of_papers, result_list):
    for paper in chunk_of_papers:
        if 'abstract' not in paper['bib']:
            scholarly.fill(paper)
            new_paper = {}
            try:
                new_paper["Title"] = paper["bib"]["title"] if "title" in paper["bib"] else "Unknown"
                new_paper["Publication year"] = int(paper["bib"]["pub_year"]) if "pub_year" in paper["bib"] else 0
                new_paper["Publication url"] = paper["pub_url"] if "pub_url" in paper else "Unknown"
                new_paper["Abstract"] = paper["bib"]["abstract"] if "abstract" in paper["bib"] else "Unknown"
                new_paper["Abstract entirety"] = 0 if new_paper["Abstract"][-1]=='…' else 1
                new_paper["Author pub id"] = paper["author_pub_id"] if "author_pub_id" in paper else "Unknown"
                new_paper["Publisher"] = paper["bib"]["publisher"] if "publisher" in paper["bib"] else "Unknown"
                print(new_paper["Title"])
                result_list.append(new_paper)
                
            except Exception as error:
                print("There is a problem")
                print(error)
    
    
def check_missings(authors_list: list):
    for cc, author in enumerate(authors_list):
        if 'Publications' in author and 'unscraped papers' in author:
            print(cc)
            print('Number of unscraped papers: {}    Number of Publications: {}   {}'.format(len(author['unscraped papers']),
                                                                                             len(author['Publications']),
                                                                                             author["romanize name"]))

def copy_unscraped_papers_fix(authors_list: list):
    for author in authors_list:
        if 'Publications' in author and 'unscraped papers' in author and author['Publications'] and author['unscraped papers']:
            publications_titles = [x.get('Title') for x in author.get('Publications')]
            for paper in author.get('unscraped papers'):
                if paper['bib']['title'] not in publications_titles:
                    try:
                        new_paper = {}
                        new_paper["Title"] = paper["bib"]["title"] if "title" in paper["bib"] else "Unknown"
                        new_paper["Publication year"] = int(paper["bib"]["pub_year"]) if "pub_year" in paper["bib"] else 0
                        new_paper["Publication url"] = paper["pub_url"] if "pub_url" in paper else "Unknown"
                        new_paper["Abstract"] = paper["bib"]["abstract"] if "abstract" in paper["bib"] else "Unknown"
                        new_paper["Abstract entirety"] = 0 if new_paper["Abstract"][-1]=='…' else 1
                        new_paper["Author pub id"] = paper["author_pub_id"] if "author_pub_id" in paper else "Unknown"
                        new_paper["Publisher"] = paper["bib"]["publisher"] if "publisher" in paper["bib"] else "Unknown"
                        print(new_paper["Title"])
                        author['Publications'].append(new_paper)
                        
                    except Exception as error:
                        print("There is a problem")
                        print(error)
                    
                    
    
    
#-----------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    pass
    # pg = ProxyGenerator()
    # success = pg.SingleProxy(http = "http://kartzafos22:1gnsjksaDs6FkTGT@proxy.packetstream.io:31112")
    # scholarly.use_proxy(pg)
    
    # t = mt.my_time()
    # t.tic()
    # test = paper_scraper("Thomas Karanikiotis", threads_num=20, host="localhost:27017", 
    #                      json_file_path = r"E:\GitHub_clones\Apella-plus-thesis\json_files")
    # t.toc()
    
    # counter=327
    # pg = ProxyGenerator()
    # success = pg.SingleProxy(http = "http://118.140.160.8480")
    # scholarly.use_proxy(pg)
    # for professor in test:
    #     try:
    #         counter += 1
    #         paper_scraper_extra(professor, threads_num=25)
    #         save2json(test, path2save=fr'..\json_files\csd_out_with_abstract\unscraped_papers{counter}.json')
    #     except Exception as error:
    #         print("There is a problem in paper_scraper")
    #         print(error)
    
    test = open_json(r'..\json_files\csd_out_with_abstract\unscraped_papers_with_abstracts.json')
    check_missings(test)
    copy_unscraped_papers_fix(test)
    
    
    