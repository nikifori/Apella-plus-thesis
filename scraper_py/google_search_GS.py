'''
Filename: e:\GitHub_clones\Apella-plus-thesis\python_files\google_search.py
Path: e:\GitHub_clones\Apella-plus-thesis\python_files
Created Date: Saturday, November 13th 2021, 12:21:20 pm
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
    query = "{0} {1} google scholar".format(author_dict["romanize name"], author_dict["University"])
    return query

# return author dictionary with scholar name
def get_scholar_name(author_dict: dict): 
    query = query_maker(author_dict)
    links = search(query, num_results=5)
    for link in links:
        if "scholar" in link:
            try:
                author_page = requests.get(link)
                soup = BeautifulSoup(author_page.content, "html.parser")
                name = soup.find("div", id="gsc_prf_in").text
                print(name)
                author_dict["Scholar name"] = name
                name_similarity = similar(name.lower(), author_dict["romanize name"].lower())
                temp_id = link.split("user=")[1]
                author_dict["Scholar id"] = temp_id.split("&hl=")[0] if "&hl=" in temp_id else temp_id
                author_dict["name_similarity"] = name_similarity
                return author_dict
            
            except:
                print("error")
                continue
        
        else:
            continue 
        
    print("Url does not start with http://scholar.google.com")
    print("Unknown")
    author_dict["Scholar name"] = "Unknown"
    author_dict["Scholar id"] = "Unknown"
    return author_dict
    

    