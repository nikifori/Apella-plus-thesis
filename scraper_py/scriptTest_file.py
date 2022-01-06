'''
Filename: e:\GitHub_clones\Apella-plus-thesis\scraper_py\scriptTest_file.py
Path: e:\GitHub_clones\Apella-plus-thesis\scraper_py
Created Date: Friday, December 10th 2021, 12:39:16 am
Author: nikifori

Copyright (c) 2021 Your Company
'''

import pandas as pd
import json
from __utils__ import *

# correct main authors list - fix unscraped authors
def correct_authors(main_authors_list: list, authors2add: list):
    cc=0
    for unlist_author in authors2add:
        for counter, author in enumerate(main_authors_list):
            if unlist_author["name"]==author["name"]:
                cc+=1
                print(cc)
                main_authors_list[counter] = unlist_author
    
    # return main_authors_list

# check unscraped authors
def check_unscraped_authors(authors_list: list):
    cc = 0
    unscraped_authors = []   
    for author in authors_list:
        if "Publications" not in author:
            unscraped_authors.append(author)
            print(author["name"])
            cc += 1
            print(cc)
    print(cc)
    return unscraped_authors

def del_abstracts(authors_list: list):
    for author in authors_list:
        if "Publications" in author:
            for pub in author["Publications"]:
                if "Abstract" in pub:
                    del pub["Abstract"]
                    
def del_specter_emb(authors_list: list):
    for author in authors_list:
        if "Publications" in author:
            for pub in author["Publications"]:
                if "Specter embedding" in pub:
                    del pub["Specter embedding"]

def correct_rank(authors_list: list):
    
    rank_1 = ["καθηγητής", "professor", "καθηγήτρια", "διευθυντής ερευνών", "ερευνητής α", "αθηγητής"]
    rank_2 = ["αναπληρωτής", "αναπληρώτρια", "associate", "κύριος ερευνητής", "ερευνητής β"]
    rank_3 = ["επίκουρος", "επίκουρη", "assistant", "εντεταλμένος αρευνητής", "ερευνητής γ"]
    
    for author in authors_list:
        if type(author.get("Rank")) is str:
            real_rank = author["Rank"].lower()
            
            if any(rank in real_rank for rank in rank_3): 
                author["Rank"]=3
            elif any(rank in real_rank for rank in rank_2): 
                author["Rank"]=2
            elif any(rank in real_rank for rank in rank_1): 
                author["Rank"]=1
    
def delete_NameSimilarity(authors_list: list):
    for author in authors_list:
        if "name_similarity" in author:
            del author["name_similarity"]
                

def paper_checker(authors_list: list):
    from scholarly import scholarly, ProxyGenerator
    import operator
    
    unscraped_papers_authors_list = []
    for cc, scraped_author in enumerate(authors_list):
        if 'Publications' in scraped_author and scraped_author['Scholar name']!='Unknown':
            print(cc)
            
            try:
                author = scholarly.search_author_id(scraped_author["Scholar id"]) #object
            except Exception as error:
                print(error)
                print("id query does not work")
                try:
                    search_query = scholarly.search_author(scraped_author["Scholar name"]) 
                    author = next(search_query) 
                except Exception as error:
                    print(error)
                    print("name query does not work")
                    return
            
            scholarly.fill(author, sections=["publications"])
            # sort list of papers based on pub_year
            for paper in author["publications"]:
                paper["pub_year"] = int(paper["bib"]["pub_year"]) if "pub_year" in paper["bib"] else 0
                
            author["publications"].sort(key=operator.itemgetter("pub_year"), reverse=True) 
            paper_list = []
            for paper in author["publications"]:
                if paper["pub_year"]>=2000: paper_list.append(paper)
            
            author2append = {"romanize name": scraped_author.get('romanize name'),
                             'Scholar name': scraped_author.get('Scholar name'),
                             'unscraped papers': []}
            
            scraped_papers_list_titles = [x['Title'] for x in scraped_author['Publications']]
            
            for paper in paper_list:
                if paper['bib']['title'] not in scraped_papers_list_titles:
                    author2append['unscraped papers'].append(paper)
            
            unscraped_papers_authors_list.append(author2append)
            
    return unscraped_papers_authors_list
                        


def append_unscraped_pubs(main_list: list, extra_list: list):
    cc=0
    for incomplete_author in extra_list:
        for counter, author in enumerate(main_list):
            if incomplete_author["Scholar name"]==author["Scholar name"]:
                cc+=1
                print(cc)
                print('{}     {}     {}'.format(author["Scholar name"], main_list[counter]['Scholar name'], incomplete_author['Scholar name']))
                main_list[counter]['Publications'] = main_list[counter]['Publications'] + incomplete_author["Publications"]


if __name__ == '__main__':
    
    pass
    

    # for file in files:
        # authors = open_json(fr"..\json_files\csd_out_with_abstract\{file}.json")
        # delete_NameSimilarity(authors)
        # save2json(authors, path2save=fr"..\json_files\csd_out_with_abstract\{file}2.json")
        
    # file = 'csd_out_completed_missing_2_no_greek_rank'
    # authors = open_json(fr"..\json_files\csd_out_with_abstract\{file}.json")
    
    # test = paper_checker(authors)
    test = open_json(r'..\json_files\csd_out_with_abstract\unscraped_papers\unscraped_papers_with_abstracts.json')
    
    # for author in test:
    #     print('{}     {}'.format((len(author['unscraped papers'])), author['romanize name']))
    
    # save2json(authors, path2save=r'..\json_files\csd_out_with_abstract\csd_out_completed_missing_2_no_greek_only_titles_rank2.json')
    
    # append_unscraped_pubs(authors, test)
    
