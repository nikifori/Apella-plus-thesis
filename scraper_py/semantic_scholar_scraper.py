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

# similarity between stirngs
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# return the query  
def query_maker(author_dict: dict):
    query = "{0} {1} semantic scholar".format(author_dict["romanize name"], author_dict["University"])
    return query

# return author dictionary with scholar name
def get_semantic_name(author_dict: dict): 
    query = query_maker(author_dict)
    links = search(query, num_results=10)
    for link in links:
        if "semantic" in link:
            try:
                author_page = requests.get(link)
                soup = BeautifulSoup(author_page.content, "html.parser")
                name = soup.find(class_="author-detail-card__author-name").text
                name_similarity = similar(name.lower(), author_dict["romanize name"].lower())
                print(name)
                author_dict["Semantic Scholar id"] = link.split("/")[-1]
                author_dict["name_similarity"] = name_similarity
                return author_dict
            
            except Exception as error:
                print(error)
                print("There is a problem in Semantic Scholar id retrieval")
        
        else:
            continue 
        
    print("Url does not start with http://scholar.google.com")
    print("Unknown")
    author_dict["Scholar name"] = "Unknown"
    author_dict["Scholar id"] = "Unknown"
    return author_dict













