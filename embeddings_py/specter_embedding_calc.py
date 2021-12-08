'''
Filename: e:\GitHub_clones\Apella-plus-thesis\embeddings_py\specter_embedding_calc.py
Path: e:\GitHub_clones\Apella-plus-thesis\embeddings_py
Created Date: Sunday, November 28th 2021, 8:28:22 pm
Author: nikifori

Copyright (c) 2021 Your Company
'''
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


def author_specter_embedding(authors_list: list):
    for counter, author in enumerate(authors_list):
        print(counter)
        if "Publications" in author:
            list_of_embeddings = []
            try:
                for pub in author["Publications"]:
                    list_of_embeddings.append(pub['Specter embedding'])
                
                author['Mean specter embedding'] = np.mean(list_of_embeddings, axis=0)
                    
            except Exception as error:
                print("Problem in specter_embedding function")
                print(error)
    
def title_embedding(title: str):
    inp = tokenizer(title, padding=True, truncation=True, return_tensors="pt", max_length=512)
    title_embedding = model(**inp).last_hidden_state[:, 0, :]
    title_embedding = title_embedding.detach().numpy()
    
    return title_embedding

def compute_ranking(authors_list: list, title_embedding):
    ranking = {'Author name': [], 'Cosine similarity': []}
    for counter, author in enumerate(authors_list):
        print(counter)
        if 'Mean specter embedding' in author:
            sim = cosine_similarity(author['Mean specter embedding'], title_embedding)
            ranking['Author name'].append(author['name'])
            ranking['Cosine similarity'].append(sim[0][0])
            
    return ranking
# ---------------------------------------------------------------------------
with open(r'..\json_files\csd_in_with_abstract\csd_in_specter.json', encoding="utf8") as json_file:
    csd_in_specter = json.load(json_file)
with open(r'..\json_files\csd_out_with_abstract\csd_out_specter.json', encoding="utf8") as json_file:
    csd_out_specter = json.load(json_file)

# needed for title_embedding function
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
model = AutoModel.from_pretrained('allenai/specter')

# authors embeddings
author_specter_embedding(csd_in_specter)
author_specter_embedding(csd_out_specter)

# # title embedding
# title = title_job[1]
# title_emb = title_embedding(title)

# # compute ranking
# ranking_in = compute_ranking(csd_in_specter, title_emb)
# ranking_df_in = pd.DataFrame(data=ranking_in).sort_values(by=['Cosine similarity'], ascending=False, ignore_index=True)
# ranking_df_in.to_csv(path_or_buf=fr'..\csv_files\{title}_in.csv', index=False)
# ranking_out = compute_ranking(csd_out_specter, title_emb)
# ranking_df_out = pd.DataFrame(data=ranking_out).sort_values(by=['Cosine similarity'], ascending=False, ignore_index=True)
# ranking_df_out.to_csv(path_or_buf=fr'..\csv_files\{title}_out.csv', index=False)

title_job = ["Fuzzy Systems and Fuzzy Rules Based Systems", "Image and Video Processing", "Speech Recognition and Processing", "Software Engineering Techniques", "Unsupervised Learning and Pattern Recognition in Natural Language Processing"]
for title in title_job:
    print(title)
    title_emb = title_embedding(title)
    ranking_in = compute_ranking(csd_in_specter, title_emb)
    ranking_df_in = pd.DataFrame(data=ranking_in).sort_values(by=['Cosine similarity'], ascending=False, ignore_index=True)
    ranking_df_in.to_csv(path_or_buf=fr'..\csv_files\{title}_in.csv', index=False)
    ranking_out = compute_ranking(csd_out_specter, title_emb)
    ranking_df_out = pd.DataFrame(data=ranking_out).sort_values(by=['Cosine similarity'], ascending=False, ignore_index=True)
    ranking_df_out.to_csv(path_or_buf=fr'..\csv_files\{title}_out.csv', index=False)


# save csv
# df.to_csv(path_or_buf=r'..\csv_files\csd_data_out_processed.csv', index=False)

