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

# return the query  
def query_maker(author_dict: dict):
    query = "{0} {1} google scholar".format(author_dict["romanize name"], "auth")
    return query

# return author dictionary with scholar name
def get_scholar_name(author_dict: dict): 
    query = query_maker(author_dict)
    link = search(f"{query}", num_results=0)[0]
    if "scholar" in link:
        author_page = requests.get(link)
        soup = BeautifulSoup(author_page.content, "html.parser")
        name = soup.find("div", id="gsc_prf_in").text
        print(name)
        author_dict["Scholar name"] = name
        return author_dict
    else: 
        print("Url does not start with http://scholar.google.com")
        print("Unknown")
        author_dict["Scholar name"] = "Unknown"
        return author_dict
    





    