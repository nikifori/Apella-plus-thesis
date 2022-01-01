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
from semanticscholar import SemanticScholar
import operator


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
        if bool(re.search("^https://www.semanticscholar.org/author/.",link)):
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

def fill_publications_title(author_dict: dict):
    try:
        sch = SemanticScholar(timeout=5)
        temp_author = sch.author(author_dict["Semantic Scholar id"])
        author_dict["Publications"] = temp_author.get("papers")
        return author_dict
    except Exception as error:
        print("There is a problem")
        print(error)

def sort_papers_byYear(author_dict: dict):
    if "Publications" in author_dict:
        author_dict["Publications"] = [x for x in author_dict["Publications"] if type(x.get("year")) is int]
        for paper in author_dict.get("Publications"):
            paper["year"] = int(paper.get("year")) if "year" in paper else 0
        
        author_dict["Publications"].sort(key=operator.itemgetter("year"), reverse=True) 
        paper_list = []
        for paper in author_dict["Publications"]:
            if paper["year"]>=2000: paper_list.append(paper)
        
        author_dict["Publications"] = paper_list.copy()
        return author_dict
    
def fill_publications(author_dict: dict):
    for counter, paper in enumerate(author_dict.get("Publications")):
        if "Abstract" not in paper:
            try:
                print(counter)
                sch = SemanticScholar(timeout=2)
                semantic_resp = sch.paper(paper.get("paperId"))
                paper["Abstract"] = semantic_resp.get("abstract")
            except Exception as error:
                print("There is a problem")
                print(error)
    
    return author_dict

        
if __name__ == '__main__':
    
    # with open(r'..\json_files\csd_out_with_abstract\csd_out_authors_out_of_GS.json', encoding="utf8") as json_file:
    #     csd_out_authors_out_of_GS = json.load(json_file)
    
    # find author name in semantic scholar
    # for i in range(len(csd_out_authors_out_of_GS)): 
    #     get_semantic_name(csd_out_authors_out_of_GS[i])
        
    # save in csv
    # df = pd.DataFrame.from_records(csd_out_authors_out_of_GS)
    # df.to_csv(path_or_buf=r'..\csv_files\csd_out_authors_out_of_GS_processed_similarity.csv', index=False)
    
    # csd_out_authors_out_of_GS = pd.read_csv(r'..\csv_files\csd_out_authors_out_of_GS_processed_similarity.csv').to_dict(orient="records")
    # csd_out_unknown = [author for author in csd_out_authors_out_of_GS if author.get("Semantic Scholar id")=="Unknown"]
    # csd_out_semantic_scholar = [author for author in csd_out_authors_out_of_GS if author.get("Semantic Scholar id")!="Unknown"]
    
    PROXY = {"http": "http://kartzafos22:1gnsjksaDs6FkTGT@proxy.packetstream.io:31112",
              "https": "http://kartzafos22:1gnsjksaDs6FkTGT@proxy.packetstream.io:31112"}
    
    # check request ip
    # for i in range(50):
    #     resp = requests.get("https://httpbin.org/ip", proxies=PROXY)
    #     print(resp.json())
    
    # for counter, author in enumerate(csd_out_semantic_scholar):
    #     print(counter)
    #     fill_publications_title(author)
    #     sort_papers_byYear(author)
        
    # save json with only titles
    # save2json(json_file="csd_out_semantic_scholar", path2save="csd_out_with_abstract\csd_out_semantic_scholar_only_titles.json")
    
    csd_out_semantic = open_json("csd_out_with_abstract\csd_out_semantic_scholar_abstracts.json")
    
    for counter, author in enumerate(csd_out_semantic):
        print(counter)
        fill_publications(author)
    
    save2json(json_fi=csd_out_semantic, path2save="csd_out_with_abstract\csd_out_semantic_scholar_abstracts.json")
    
    # rename title to Title and year to Publication year
    for counter, author in enumerate(csd_out_semantic):
        for paper in author["Publications"]:
            paper["Publication year"] = paper.pop("year")
            paper["Title"] = paper.pop("title")
    
    save2json(json_fi=csd_out_semantic, path2save="csd_out_with_abstract\csd_out_semantic_scholar_abstracts.json")
    save2json(json_fi=csd_out_semantic, path2save="csd_out_with_abstract\csd_out_semantic_scholar_abstracts_specter_embedding.json")
    
    
    
    
    
    
    
    
    