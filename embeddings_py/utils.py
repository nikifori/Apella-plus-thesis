'''
Filename: e:\GitHub_clones\Apella-plus-thesis\scraper_py\scriptTest_file.py
Path: e:\GitHub_clones\Apella-plus-thesis\embeddings_py
Created Date: Saturday, December 18th 2021, 3:02:45 pm
Author: nikifori

Copyright (c) 2021 Your Company
'''
import json

def save2json(json_fi: list, path2save: str):
    json_file = json.dumps(json_fi, indent=4)
    with open(fr'{path2save}', 'w', encoding='utf-8') as f:
        f.write(json_file)
            
def open_json(path2read: str):
    with open(fr'{path2read}', encoding="utf8") as json_file:
        dictionary = json.load(json_file)
    return dictionary

if __name__ == '__main__':
    
    test = open_json("..\json_files\can_not_fetch_complete.json")