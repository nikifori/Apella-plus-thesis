#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   async_google_search.py
@Time    :   2022/01/09 22:21:39
@Author  :   nikifori 
@Version :   -
@Contact :   nikfkost@gmail.com
'''

import threading
import pandas as pd
import  os
from __utils__ import *
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import re
from difflib import SequenceMatcher
import math
from thefuzz import fuzz
import jellyfish

# similarity measurement
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# query constructor for Google Scholar search
def query_maker_GS(author_dict: dict):
    query = "{0} {1} google scholar".format(author_dict["romanize name"], author_dict["University email domain"])
    return query

# retrieve Google Scholar name/id
def get_scholar_name(author_dict: dict, proxy_dict: dict=None): 
    query = query_maker_GS(author_dict)
    links = search(query, num_results=5)
    for link in links:
        if "scholar" in link:
            try:
                author_page = requests.get(link) if not proxy_dict else requests.get(link, proxies=proxy_dict)
                soup = BeautifulSoup(author_page.content, "html.parser")
                name = soup.find("div", id="gsc_prf_in").text
                print(name)
                author_dict["Scholar name"] = name
                # name_similarity = similar(name.lower(), author_dict["romanize name"].lower())
                temp_id = link.split("user=")[1]
                author_dict["Scholar id"] = temp_id.split("&hl=")[0] if "&hl=" in temp_id else temp_id
                # author_dict["name_similarity"] = name_similarity
                author_dict["Semantic Scholar name"] = 'Unknown'
                author_dict["Semantic Scholar id"] = 'Unknown'
                author_dict["ResearchGate name"] = 'Unknown'
                author_dict["ResearchGate url name/id"] = 'Unknown'
                author_dict["ResearchGate url type"] = 'Unknown'

                return author_dict
            
            except Exception as error:
                print(error)
                print("Url does not start with http://scholar.google.com")
        
        else:
            continue 
        
    print("There is a problem in Google Scholar name/id retrieval")
    print("Unknown")
    author_dict["Scholar name"] = "Unknown"
    author_dict["Scholar id"] = "Unknown"
    return author_dict

# query constructor for Semantic Scholar search
def query_maker_SS(author_dict: dict):
    query = "{0} semantic scholar".format(author_dict["romanize name"])
    return query

# retrieve Semantic Scholar name/id
def get_semantic_name(author_dict: dict, proxy_dict: dict=None): 
    query = query_maker_SS(author_dict)
    links = search(query, num_results=10)
    for link in links:
        if bool(re.search("^https://www.semanticscholar.org/author/.",link)):
            try:
                author_page = requests.get(link) if not proxy_dict else requests.get(link, proxies=proxy_dict)
                soup = BeautifulSoup(author_page.content, "html.parser")
                name = soup.find(class_="author-detail-card__author-name").text
                # name_similarity = similar(name.lower(), author_dict["romanize name"].lower())
                print(name)
                author_dict["Semantic Scholar name"] = name
                author_dict["Semantic Scholar id"] = link.split("/")[-1]
                # author_dict["name_similarity"] = name_similarity
                author_dict["ResearchGate name"] = 'Unknown'
                author_dict["ResearchGate url name/id"] = 'Unknown'
                author_dict["ResearchGate url type"] = 'Unknown'
                
                return author_dict
            
            except Exception as error:
                print(error)
                print("Url does not start with https://www.semanticscholar.org/")
        
        else:
            continue 
        
    print("There is a problem in Semantic Scholar name/id retrieval")
    print("Unknown")
    author_dict["Semantic Scholar name"] = "Unknown"
    author_dict["Semantic Scholar id"] = "Unknown"
    return author_dict

# query constructor for ResearchGate search
def query_maker_RG(author_dict: dict):
    query = "{0} {1} researchgate".format(author_dict["romanize name"], author_dict["University email domain"])
    return query

# retrieve ResearchGate name/id and type of author's page (profile or scientific-contributions)
def get_researchgate_name(author_dict: dict, proxy_dict: dict=None): 
    # PROXY = {"http": "http://kartzafos22:1gnsjksaDs6FkTGT@proxy.packetstream.io:31112",
    #           "https": "http://kartzafos22:1gnsjksaDs6FkTGT@proxy.packetstream.io:31112"}
    query = query_maker_RG(author_dict)
    links = search(query, num_results=10)
    for link in links:
        if bool(re.search("^https://www.researchgate.net/profile/.",link)):
            try:
                author_page = requests.get(link) if not proxy_dict else requests.get(link, proxies=proxy_dict)
                soup = BeautifulSoup(author_page.content, "html.parser")
                name = soup.find("div", class_="nova-legacy-e-text nova-legacy-e-text--size-xxl nova-legacy-e-text--family-sans-serif nova-legacy-e-text--spacing-xxs nova-legacy-e-text--color-inherit fn").text
                # name_similarity = similar(name.lower(), author_dict["romanize name"].lower())
                print(name)
                author_dict["ResearchGate name"] = name
                author_dict["ResearchGate url name/id"] = link.split("/")[-1]
                # author_dict["name_similarity"] = name_similarity
                author_dict["ResearchGate url type"] = link.split("/")[-2]
                
                return author_dict
            
            except Exception as error:
                print(error)
                print("Url does not start with https://www.researchgate.net/profile/")
            
        elif bool(re.search("^https://www.researchgate.net/scientific-contributions/.",link)):
            try:
                author_page = requests.get(link) if not proxy_dict else requests.get(link, proxies=proxy_dict)
                soup = BeautifulSoup(author_page.content, "html.parser")
                name = link.split("/")[-1].split("-")
                name.pop()
                name = " ".join(x for x in name)
                print(name)
                # name_similarity = similar(name.lower(), author_dict["romanize name"].lower())
                author_dict["ResearchGate name"] = name
                author_dict["ResearchGate url name/id"] = link.split("/")[-1]
                # author_dict["name_similarity"] = name_similarity
                author_dict["ResearchGate url type"] = link.split("/")[-2]
                
                return author_dict
            
            except Exception as error:
                print(error)
                print("Url does not start with https://www.researchgate.net/scientific-contributions/")
                
        
        else:
            continue 
        
    print("Author is not in ResearchGate.")
    print("Unknown")
    author_dict["ResearchGate name"] = "Unknown"
    author_dict["ResearchGate url name/id"] = "Unknown"
    return author_dict


def pipeline(chuck_of_authors: list, global_result_list, proxy_dict: dict=None):
    for author_dict in chuck_of_authors:
        query_maker_GS(author_dict)
        author = get_scholar_name(author_dict, proxy_dict)

        if author['Scholar name']=='Unknown':
            query_maker_SS(author_dict)
            author = get_semantic_name(author_dict, proxy_dict)

            if author['Semantic Scholar name']=='Unknown':
                query_maker_RG(author_dict)
                author = get_researchgate_name(author_dict, proxy_dict)
    
        global_result_list.append(author_dict)


def chunks(authors_list: list, threads: int):
    chunked_list = []
    n = len(authors_list)
    for i in range(threads):
       start = int(math.floor(i * n / threads))
       finish = int(math.floor((i + 1) * n / threads) - 1)
       chunked_list.append(authors_list[start:(finish+1)])
    
    return chunked_list


def main(authors_list: list, threads_num: int=1, proxy_dict: dict=None):
    if threads_num>len(authors_list):threads_num=len(authors_list) # threads always equal or lower than number of authors
    print(threads_num)

    #list to save results
    global result_list
    result_list = []

    threads = []
    chunked_list = chunks(authors_list, threads_num)
    for chunk in chunked_list:
        x = threading.Thread(target=pipeline, args=(chunk, result_list, proxy_dict))
        threads.append(x)
        x.start()
    
    for thread in threads:
        thread.join()
    
    return result_list


def evaluate_results(exp_data: dict, ground_truth_data): 
    """
    Calculate the correct author names/ids

    Parameters
    ----------
    exp_data : dict
        DESCRIPTION.
    ground_truth_data : DataFrame
        DESCRIPTION.

    Returns
    -------
    Percentage of correct authors names/ids.

    """
    
    exp_data = pd.DataFrame(exp_data)
    all_data = pd.concat([exp_data, ground_truth_data], ignore_index=True)
    correct = all_data.duplicated(subset=['name', 'romanize name', 'School-Department', 'University',
                                          'University email domain', 'Rank', 'Apella_id', 'Scholar name',
                                          'Scholar id', 'Semantic Scholar name', 'Semantic Scholar id',
                                          'ResearchGate name', 'ResearchGate url name/id',
                                          'ResearchGate url type']).sum()
    
    incorrect_df = all_data[~all_data.duplicated(subset=['name', 'romanize name', 'School-Department', 'University',
                                          'University email domain', 'Rank', 'Apella_id', 'Scholar name',
                                          'Scholar id', 'Semantic Scholar name', 'Semantic Scholar id',
                                          'ResearchGate name', 'ResearchGate url name/id',
                                          'ResearchGate url type'], keep=False)]
    
    percentage = correct/len(ground_truth_data)    
    return percentage, incorrect_df




if __name__ == "__main__":
    
    # cd scraper_py folder
    # os.chdir(r'.\scraper_py')

    # unlabeled data
    csd_in_test = pd.read_csv(r'..\csv_files\csd_data_in_unlabeled.csv').to_dict(orient='records')
    csd_out_test = pd.read_csv(r'..\csv_files\csd_data_out_unlabeled.csv').to_dict(orient='records')

    # ground truth data
    csd_in_ground_truth = pd.read_csv(r'..\csv_files\csd_data_in_processed_ground_truth_completed.csv').to_dict(orient='records')
    csd_out_ground_truth = pd.read_csv(r'..\csv_files\csd_data_out_processed_ground_truth_completed.csv').to_dict(orient='records')

    csd_test = [csd_in_test, csd_out_test]
    csd_ground_truth = [csd_in_ground_truth, csd_out_ground_truth]
    
    
    # proxy = {'http': '110.77.200.135:8080'}
    proxy=None
    results = []
    percentage = []
    for cc, i in enumerate(csd_test):
        results.append(main(i, 20, proxy))
        percentage.append(evaluate_results(results[cc], csd_ground_truth[cc]))
        
    print(percentage)
        
    
    







    