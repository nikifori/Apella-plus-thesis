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
import pandas as pd
from csd_csv_parser import *
import math

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
    

def correct_authors_dataset(main_authors_list: list, authors2add: list):
    cc=0
    for unlist_author in authors2add:
        for counter, author in enumerate(main_authors_list):
            if unlist_author["name"]==author["name"]:
                cc+=1
                print(cc)
                if 'Semantic Scholar name' in unlist_author:
                    author['Semantic Scholar name'] = unlist_author['Semantic Scholar name']
                    author['Semantic Scholar id'] = unlist_author['Semantic Scholar id']
                    author['ResearchGate name'] = 'Unknown'
                    author['ResearchGate url name/id'] = 'Unknown'
                    author['ResearchGate url type'] = 'Unknown'
                elif 'ResearchGate name' in unlist_author:
                    author['ResearchGate name'] = unlist_author['ResearchGate name']
                    author['ResearchGate url name/id'] = unlist_author['ResearchGate url name/id']
                    author['ResearchGate url type'] = unlist_author['ResearchGate url type'] #if not math.isnan(unlist_author['ResearchGate url type']) else 'Unknown'
                    author['Semantic Scholar name'] = 'Unknown'
                    author['Semantic Scholar id'] = 'Unknown'
                else:
                    author['Semantic Scholar name'] = 'Unknown'
                    author['Semantic Scholar id'] = 'Unknown'
                    author['ResearchGate name'] = 'Unknown'
                    author['ResearchGate url name/id'] = 'Unknown'
                    author['ResearchGate url type'] = 'Unknown'



def search_engine_label(authors_list: list):
    for author in authors_list:
        if author['Scholar name']!='Unknown':
            author['Search Engine label']=1
        elif author['Semantic Scholar name']!='Unknown':
            author['Search Engine label']=2
        elif author['ResearchGate name']!='Unknown':
            author['Search Engine label']=3
        else:
            author['Search Engine label']=0




if __name__ == '__main__':
    csd_in_GS = pd.read_csv(r'..\csv_files\csd_data_in_processed_ground_truth.csv').to_dict(orient="records")
    csd_out_GS = pd.read_csv(r'..\csv_files\csd_data_out_processed_ground_truth.csv')#.to_dict(orient="records")
    # csd_out_SS = pd.read_csv(r'..\csv_files\csd_out_semantic_scholar_ground_truth.csv').to_dict(orient="records")
    # csd_out_RG = pd.read_csv(r'..\csv_files\csd_out_researchgate_ground_truth.csv').to_dict(orient="records")
    
    
    
    # preprocessing of initial data
    
    # List of Dictionaries
    csd_in = df_to_dict_parser(pd.read_excel(r"..\csv_files\csd_data_in.xlsx"))
    csd_out = df_to_dict_parser(pd.read_excel(r"..\csv_files\csd_data_out.xlsx"))
    
    # DataFrame
    csd_in = pd.DataFrame(csd_in)
    csd_out = pd.DataFrame(csd_out)
    
    csd_all = [csd_in, csd_out]
    
    # for csd in csd_all:
    #     print(csd['University email domain'].unique())
    #     print(csd['University email domain'].value_counts())
    
    # for csd in csd_all:
    #     for author in csd:
            
    # csd_out_GS['University email domain'] = csd_out['University email domain']
    # csd_out_GS['University'] = csd_out['University']
    # csd_out_GS = csd_out_GS[['name',
    #  'romanize name',
    #  'School-Department',
    #  'University',
    #  'University email domain',
    #  'Rank',
    #  'Apella_id',
    #  'Scholar name',
    #  'Scholar id']]
    x.to_csv(path_or_buf=r'..\csv_files\csd_data_in_processed_ground_truth_completed.csv', index=False)
    
    # csd_out_GS = csd_out_GS.to_dict(orient='records')
    
    # for author in csd_out_GS:
    #     if author["Scholar name"]!='Unknown':
    #         author['Semantic Scholar name'] ='Unknown'
    #         author['Semantic Scholar id'] ='Unknown'
    #         author['ResearchGate name'] ='Unknown'
    #         author['ResearchGate url name/id'] = 'Unknown'
    #         author['ResearchGate url type'] = 'Unknown'
            
    
    correct_authors_dataset(csd_out_GS, csd_out_SS)
    correct_authors_dataset(csd_out_GS, csd_out_RG)
    
    
    # csd_in_GS['University email domain'] = csd_in['University email domain']
    # csd_in_GS = csd_in_GS[['name',
    #   'romanize name',
    #   'School-Department',
    #   'University',
    #   'University email domain',
    #   'Rank',
    #   'Apella_id',
    #   'Scholar name',
    #   'Scholar id']]
    # csd_in_GS.to_csv(path_or_buf=r'..\csv_files\csd_data_in_processed_ground_truth2.csv', index=False)
    correct_authors_dataset(csd_in_GS, csd_in_GS)
    csd_in_GS[19]['ResearchGate name'] = "Dionysios Politis"
    csd_in_GS[19]['ResearchGate url name/id'] = "Dionysios-Politis"
    csd_in_GS[19]['ResearchGate url type'] = "profile"
    
    
    csd_in = pd.read_csv(r'..\csv_files\csd_data_in_processed_ground_truth_completed.csv')
    csd_out = pd.read_csv(r'..\csv_files\csd_data_out_processed_ground_truth_completed.csv')
    
    csd_in_unlabeled = csd_in.iloc[:,:7]
    csd_out_unlabeled = csd_out.iloc[:,:7]
    csd_in_unlabeled.to_csv(path_or_buf=r'..\csv_files\csd_data_in_unlabeled.csv', index=False)
    csd_out_unlabeled.to_csv(path_or_buf=r'..\csv_files\csd_data_out_unlabeled.csv', index=False)
    
    
    
    
    
    
    
    
    
    
    
