'''
Filename: e:\GitHub_clones\Apella-plus-thesis\embeddings_py\test.py
Path: e:\GitHub_clones\Apella-plus-thesis\embeddings_py
Created Date: Thursday, November 25th 2021, 7:59:20 pm
Author: nikifori

Copyright (c) 2021 Your Company
'''

import numpy as np
import pandas as pd
import json
import math
import threading
#from transformers import *
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# papers = [{'title': 'BERT', 'abstract': 'We introduce a new language representation model called BERT'},
#           {'title': 'Attention is all you need', 'abstract': ' The dominant sequence transduction models are based on complex recurrent or convolutional neural networks'}]

# # concatenate title and abstract
# title_abs = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
# # preprocess the input
# inputs = tokenizer(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
# result = model(**inputs)
# # take the first token in the batch as the embedding
# embeddings = result.last_hidden_state[:, 0, :]

def specter_embedding(authors_list: list, global_result: list=None):
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter')
    
    for cc, author in enumerate(authors_list):
        print(cc)
        if "Publications" in author:
            try:
                for counter, pub in enumerate(author["Publications"]):
                    if "Specter embedding" not in pub:
                        temp_title_abs = pub['Title'] + tokenizer.sep_token + (pub.get('Abstract') or '')
                        input_pair = tokenizer(temp_title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
                        embedding = model(**input_pair).last_hidden_state[:, 0, :]
                        pub['Specter embedding'] = embedding.detach().numpy().tolist()
                        # print(counter)
            except Exception as error:
                print("Problem in specter_embedding function")
                print(error)
    
    # global_result = global_result + authors_list
    # return authors_list

def specter_embedding_parallel(authors_list: list, threads_num:int=1):
    if threads_num>len(authors_list):threads_num=len(authors_list) # papers always
    chunked_authors_list = chunks(authors_list, threads_num)
    threads = []
    global global_result
    global_result = []
    for chunk in chunked_authors_list:
        x = threading.Thread(target=specter_embedding, args=(chunk, global_result))
        threads.append(x)
        x.start()
    
    for thread in threads:
        thread.join()
        
    return global_result
          
def chunks(my_list, threads):
    chunked_list = []
    n = len(my_list)
    for i in range(threads):
       start = int(math.floor(i * n / threads))
       finish = int(math.floor((i + 1) * n / threads) - 1)
       chunked_list.append(my_list[start:(finish+1)])
       
    return chunked_list
#------------------------------------------------------------------------------      

if __name__ == '__main__':

    with open(r'..\json_files\csd_in_with_abstract\csd_in_with_abstracts_db.json', encoding="utf8") as json_file:
        csd_in_list_dict = json.load(json_file)
    with open(r'..\json_files\csd_out_with_abstract\csd_out_with_abstracts_db.json', encoding="utf8") as json_file:
        csd_out_list_dict = json.load(json_file)
        
    authors_in_with_embeddings = specter_embedding(csd_in_list_dict)
    # save dictionary as json file
    json_file = json.dumps(csd_in_list_dict, indent=4)
    json_name = "csd_in_specter"
    with open(fr'..\json_files\{json_name}.json', 'w', encoding='utf-8') as f:
        f.write(f"{json_file}")
    
    authors_out_with_embeddings = specter_embedding(csd_out_list_dict)
    # save dictionary as json file
    json_file = json.dumps(csd_out_list_dict, indent=4)
    json_name = "csd_out_specter"
    with open(fr'..\json_files\{json_name}.json', 'w', encoding='utf-8') as f:
        f.write(f"{json_file}")
    





























    