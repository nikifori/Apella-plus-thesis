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
import math
from scriptTest_file import *

# similarity between stirngs
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# return the query  
def query_maker(author_dict: dict):
    query = "{0} {1} researchgate".format(author_dict["romanize name"], author_dict["University"])
    return query

# return author dictionary with researchgate name
# Needs extreme improvement
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

def title_scraper(author_dict: dict):
    
    if author_dict["ResearchGate url name/id"]!="Unknown" and "Publications" not in author_dict:
        
        link = "https://www.researchgate.net/{}/{}".format(author_dict["ResearchGate url type"], 
                                                             author_dict["ResearchGate url name/id"])
        try:
            author_page = requests.get(link)
            soup = BeautifulSoup(author_page.content, "html.parser")
            if author_dict["ResearchGate url type"]=="scientific-contributions":
                my_sign = soup.find_all("h2", class_="nova-legacy-e-text nova-legacy-e-text--size-l nova-legacy-e-text--family-sans-serif nova-legacy-e-text--spacing-none nova-legacy-e-text--color-grey-600")
                page_num = 1
                author_dict["Publications"] = []
                while [x.text for x in my_sign if "Publications" in x.text]:
                    stop_at = soup.find_all("h2", class_="nova-legacy-e-text nova-legacy-e-text--size-l nova-legacy-e-text--family-sans-serif nova-legacy-e-text--spacing-none nova-legacy-e-text--color-grey-600")[1]
                    paper_section = stop_at.find_all_previous("div", class_="nova-legacy-o-stack nova-legacy-o-stack--gutter-xxl nova-legacy-o-stack--spacing-xl nova-legacy-o-stack--show-divider")[1]
                    paper_blocks = paper_section.find_all("div", class_="nova-legacy-o-stack__item")
                    for cc, i in enumerate(paper_blocks):
                        title = i.find("a", class_="nova-legacy-e-link nova-legacy-e-link--color-inherit nova-legacy-e-link--theme-bare").text
                        new_paper = {"Title": title}
                        print(cc)
                        print(title)
                        try: 
                            pub_year = i.find("li", class_="nova-legacy-e-list__item nova-legacy-v-publication-item__meta-data-item")
                            pub_year = re.findall('\d+', pub_year.text)[0]
                            new_paper["Publication year"] = int(pub_year)
                            author_dict["Publications"].append(new_paper)
                        except Exception as error:
                            print(error)
                            print("Paper pub year doesn't exist")
                            try:
                                author_dict["Publications"].append(new_paper)
                            except:
                                print("Problem in new paper.")
                    
                    page_num += 1
                    new_link = "https://www.researchgate.net/{}/{}/publications/{}".format(author_dict["ResearchGate url type"], 
                                                            author_dict["ResearchGate url name/id"], page_num)
                    next_page = requests.get(new_link)
                    soup = BeautifulSoup(next_page.content, "html.parser")
                    my_sign = soup.find("h2", class_="nova-legacy-e-text nova-legacy-e-text--size-l nova-legacy-e-text--family-sans-serif nova-legacy-e-text--spacing-none nova-legacy-e-text--color-grey-600")
                    
            elif author_dict["ResearchGate url type"]=="profile":
                my_sign = soup.find_all("div", class_="nova-legacy-e-text nova-legacy-e-text--size-l nova-legacy-e-text--family-sans-serif nova-legacy-e-text--spacing-none nova-legacy-e-text--color-grey-600")
                page_num = 1
                author_dict["Publications"] = []
                while [x.text for x in my_sign if "Publications" in x.text]:
                    paper_section = soup.find_all("div", id="research-items")[0]
                    paper_blocks = paper_section.find_all("div", class_="nova-legacy-v-publication-item__body")
                    for cc, i in enumerate(paper_blocks):
                        title = i.find('a', class_="nova-legacy-e-link nova-legacy-e-link--color-inherit nova-legacy-e-link--theme-bare").text
                        new_paper = {"Title": title}
                        print(cc)
                        print(title)
                        try: 
                            pub_year = i.find("li", class_="nova-legacy-e-list__item nova-legacy-v-publication-item__meta-data-item")
                            pub_year = re.findall('\d+', pub_year.text)[0]
                            new_paper["Publication year"] = int(pub_year)
                            author_dict["Publications"].append(new_paper)
                        except Exception as error:
                            print(error)
                            print("Paper pub year doesn't exist")
                            try:
                                author_dict["Publications"].append(new_paper)
                            except:
                                print("Problem in new paper.")
                    
                    page_num += 1
                    new_link = "https://www.researchgate.net/{}/{}/{}".format(author_dict["ResearchGate url type"], 
                                                            author_dict["ResearchGate url name/id"], page_num)
                    next_page = requests.get(new_link)
                    soup = BeautifulSoup(next_page.content, "html.parser")
                    my_sign = soup.find_all("div", class_="nova-legacy-e-text nova-legacy-e-text--size-l nova-legacy-e-text--family-sans-serif nova-legacy-e-text--spacing-none nova-legacy-e-text--color-grey-600")
                    
                    
            else: return author_dict
            
        except Exception as error:
            print(error)
            print("There is a problem in ResearchGate Title retrieval")




if __name__ == '__main__':
    
    # for author in unscraped_authors:
    #     get_researchgate_name(author)
    
    # # save in csv
    # df = pd.DataFrame.from_records(unscraped_authors)
    # df.to_csv(path_or_buf=r'..\csv_files\csd_out_researchgate_name_similarity.csv', index=False)
    
    # csd_out_authors_out_of_GS = pd.read_csv(r'..\csv_files\csd_out_authors_out_of_GS_processed_similarity.csv').to_dict(orient="records")
    
    csd_out_researchgate_ground_truth = pd.read_csv(r"..\csv_files\csd_out_researchgate_ground_truth.csv").to_dict(orient="records")
    
    # scrape data
    # for author in csd_out_researchgate_ground_truth:
    #     title_scraper(author)
    
    save2json(csd_out_researchgate_ground_truth, path2save="csd_out_with_abstract\csd_out_researchgate_only_titles.json")
    
    # delete 2 authors that cant be found
    for cc, author in enumerate(csd_out_researchgate_ground_truth):
        if author["ResearchGate name"]=="Unknown":
            print(author["name"])
            csd_out_researchgate_ground_truth.pop(cc)
    
    csd_out_specter = open_json("csd_out_with_abstract\csd_out_specter_with_410_authors.json")
    correct_authors(csd_out_specter, csd_out_researchgate_ground_truth)
    # specter_embedding(csd_out_specter)
    save2json(csd_out_specter, path2save="csd_out_with_abstract\csd_out_specter_with_410_authors_missing_2.json")
    
    unscraped_authors = check_unscraped_authors(csd_out_specter)
    