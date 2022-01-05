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
                
def publications_metrics(authors_list: list):
        pub_sum = 0
        average_pub_count = 0
        for author in authors_list:
            if "Publications" in author:
                pub_sum += len(author.get("Publications"))
        
        average_pub_count = pub_sum/len(authors_list)
        
        return pub_sum, average_pub_count

if __name__ == '__main__':
    
    # save dictionary as json file
    # json_file = json.dumps(authors_out_of_GS, indent=4)
    # with open(fr'..\json_files\csd_out_with_abstract\csd_out_authors_out_of_GS.json', 'w', encoding='utf-8') as f:
    #     f.write(json_file)
    
    # csd_out_specter = open_json("csd_out_with_abstract\csd_out_specter_with_410_authors.json")
    # csd_out_with_abstracts_30_missing_with_410_authors = open_json("csd_out_with_abstract\csd_out_with_abstracts_30_missing_with_410_authors.json")
    # csd_out_with_abstracts_30_missing = open_json("csd_out_with_abstract\csd_out_with_abstracts_30_missing.json")
    # save2json(json_fi=csd_out_specter, path2save="csd_out_with_abstract\csd_out_specter_with_410_authors.json")
    
    # unscraped_authors = check_unscraped_authors(csd_out_specter)
    # unscraped_authors = check_unscraped_authors(csd_out_with_abstracts_30_missing_with_410_authors)
    # correct_authors(csd_out_specter, can_not_fetch_complete)
    
    # df = pd.read_csv(r'..\csv_files\csd_data_out_processed_ground_truth.csv')
    
    # add 2 authors that didnt exist in main json file without specter embeddings
    # temp_list = []
    # for author in csd_out_with_abstracts_30_missing:
    #     temp_list.append(author.get("name"))
    
    # temp_df = pd.DataFrame(temp_list, columns=["name"])
    
    # matched = (temp_df["name"]==df["name"])
    
    # for i in df["name"]:
    #     counter = 0
    #     for k in temp_df["name"]:
    #         if k==i:
    #             counter=1
    #     if counter==0:
    #         print(i)      
    # save2json(csd_out_with_abstracts_30_missing, "csd_out_with_abstract\csd_out_with_abstracts_30_missing2.json")  
    #---------------------------------------------------------------------------------------------------------------
    # csd_out_specter.append(csd_out_with_abstracts_30_missing[-2])
    
    # csd_out_researchgate_only_titles = open_json("csd_out_with_abstract/csd_out_researchgate_only_titles.json")
    # correct_authors(csd_out_with_abstracts_30_missing_with_410_authors, csd_out_researchgate_only_titles)
    # save2json(csd_out_with_abstracts_30_missing_with_410_authors, path2save="csd_out_with_abstract\csd_out_completed_missing_2.json")
    
    # csd_in = open_json("..\json_files\csd_in_with_abstract\csd_in_completed_no_greek.json")
    # csd_out = open_json("..\json_files\csd_out_with_abstract\csd_out_completed_missing_2_no_greek.json")
    
    # csd_out_researchgate = open_json(r'..\json_files\csd_out_with_abstract\unused\csd_out_researchgate_with_abstract_title.json')
    
    
    # csd_out_specter_rank = open_json(r'..\json_files\csd_out_with_abstract\csd_out_specter_rank.json')
    # correct_authors(csd_out_specter_rank, csd_out_researchgate)
    
    # save2json(file, path2save=fr"..\json_files\csd_out_with_abstract\{name}.json")
    
    
    # csd_out_specter_rank = open_json("..\json_files\csd_out_with_abstract\csd_out_specter_rank.json")
    
    # delete name_similarity
    # files = ["csd_out_completed_missing_2_no_greek_only_titles_rank",
    #          "csd_out_completed_missing_2_no_greek_only_titles_rank_specter", 
    #          "csd_out_completed_missing_2_no_greek_rank", 
    #          "csd_out_specter_rank"]

    # for file in files:
    #     authors = open_json(fr"..\json_files\csd_out_with_abstract\{file}.json")
    #     delete_NameSimilarity(authors)
    #     save2json(authors, path2save=fr"..\json_files\csd_out_with_abstract\{file}2.json")

    csd_in = open_json("..\json_files\csd_in_with_abstract\csd_in_completed_no_greek_rank.json")
    csd_out = open_json("..\json_files\csd_out_with_abstract\csd_out_completed_missing_2_no_greek_rank.json")
    
    csd_in_pubs_sum, csd_in_average_pub_num = publications_metrics(csd_in)
    csd_out_pubs_sum, csd_out_average_pub_num = publications_metrics(csd_out)
    
    
    
    
    
    
    
    
    