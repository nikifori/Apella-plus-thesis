'''
Filename: e:\GitHub_clones\Apella-plus-thesis\scraper_py\semantic_scholar_scraper.py
Path: e:\GitHub_clones\Apella-plus-thesis\scraper_py
Created Date: Thursday, December 9th 2021, 2:44:27 am
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


# similarity between stirngs
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# return the query  
def query_maker(author_dict: dict):
    query = "{0} semantic scholar".format(author_dict["romanize name"])
    return query

# return author dictionary with scholar name
def get_semantic_name(author_dict: dict): 
    query = query_maker(author_dict)
    links = search(query, num_results=10)
    for link in links:
        if re.search("https://www.semanticscholar.org/.*",link):
            try:
                author_page = requests.get(link)
                soup = BeautifulSoup(author_page.content, "html.parser")
                name = soup.find(class_="author-detail-card__author-name").text
                name_similarity = similar(name.lower(), author_dict["romanize name"].lower())
                print(name)
                author_dict["Semantic Scholar name"] = name
                author_dict["Semantic Scholar id"] = link.split("/")[-1]
                author_dict["name_similarity"] = name_similarity
                return author_dict
            
            except Exception as error:
                print(error)
                print("There is a problem in Semantic Scholar id retrieval")
        
        else:
            continue 
        
    print("Url does not start with https://www.semanticscholar.org/")
    print("Unknown")
    author_dict["Semantic Scholar name"] = "Unknown"
    author_dict["Semantic Scholar id"] = "Unknown"
    return author_dict



if __name__ == '__main__':
    
    with open(r'..\json_files\csd_out_with_abstract\csd_out_authors_out_of_GS.json', encoding="utf8") as json_file:
        csd_out_authors_out_of_GS = json.load(json_file)
    
    # find author name in semantic scholar
    for i in range(len(csd_out_authors_out_of_GS)): 
        get_semantic_name(csd_out_authors_out_of_GS[i])
        
    # save in csv
    df = pd.DataFrame.from_records(csd_out_authors_out_of_GS)
    df.to_csv(path_or_buf=r'..\csv_files\csd_out_authors_out_of_GS_processed_similarity.csv', index=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    