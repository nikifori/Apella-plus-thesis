'''
Filename: e:\GitHub_clones\Apella-plus-thesis\scraper_py\researchgate_scraper.py
Path: e:\GitHub_clones\Apella-plus-thesis\scraper_py
Created Date: Tuesday, December 14th 2021, 12:26:59 am
Author: nikifori

Copyright (c) 2021 Your Company
'''
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import json
import pandas as pd
import re
from semanticscholar import SemanticScholar
import operator
from semantic_scholar_scraper import *

# similarity between stirngs
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# return the query  
def query_maker(author_dict: dict):
    query = "{0} {1} researchgate".format(author_dict["romanize name"], author_dict["University"])
    return query

# return author dictionary with scholar name
def get_researchgate_name(author_dict: dict): 
    # PROXY = {"http": "http://kartzafos22:1gnsjksaDs6FkTGT@proxy.packetstream.io:31112",
    #           "https": "http://kartzafos22:1gnsjksaDs6FkTGT@proxy.packetstream.io:31112"}
    query = query_maker(author_dict)
    links = search(query, num_results=10)
    for link in links:
        if bool(re.search("^https://www.researchgate.net/profile/.",link)):
            try:
                author_page = requests.get(link)
                soup = BeautifulSoup(author_page.content, "html.parser")
                name = soup.find("div", class_="nova-legacy-e-text nova-legacy-e-text--size-xxl nova-legacy-e-text--family-sans-serif nova-legacy-e-text--spacing-xxs nova-legacy-e-text--color-inherit fn").text
                name_similarity = similar(name.lower(), author_dict["romanize name"].lower())
                print(name)
                author_dict["ResearchGate name"] = name
                author_dict["ResearchGate url name/id"] = link.split("/")[-1]
                author_dict["name_similarity"] = name_similarity
                author_dict["ResearchGate url type"] = link.split("/")[-2]
                
                return author_dict
            
            except Exception as error:
                print(error)
                print("Author is not in profile section.")
            
        elif bool(re.search("^https://www.researchgate.net/scientific-contributions/.",link)):
            try:
                author_page = requests.get(link)
                soup = BeautifulSoup(author_page.content, "html.parser")
                name = link.split("/")[-1].split("-")
                name.pop()
                name = " ".join(x for x in name)
                print(name)
                name_similarity = similar(name.lower(), author_dict["romanize name"].lower())
                author_dict["ResearchGate name"] = name
                author_dict["ResearchGate url name/id"] = link.split("/")[-1]
                author_dict["name_similarity"] = name_similarity
                author_dict["ResearchGate url type"] = link.split("/")[-2]
                
                return author_dict
            
            except Exception as error:
                print(error)
                print("Author is not in scientific-contributions section.")
                
        
        else:
            continue 
        
    print("Url does not start with https://www.researchgate.net/profile/")
    print("Unknown")
    author_dict["ResearchGate name"] = "Unknown"
    author_dict["ResearchGate url name/id"] = "Unknown"
    return author_dict



if __name__ == '__main__':
    
    for author in unscraped_authors:
        get_researchgate_name(author)
    
    
    # save in csv
    df = pd.DataFrame.from_records(unscraped_authors)
    df.to_csv(path_or_buf=r'..\csv_files\csd_out_researchgate_name_similarity.csv', index=False)
    
    # csd_out_authors_out_of_GS = pd.read_csv(r'..\csv_files\csd_out_authors_out_of_GS_processed_similarity.csv').to_dict(orient="records")
    
    
    
    
    