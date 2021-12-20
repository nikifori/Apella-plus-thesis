'''
Filename: e:\GitHub_clones\Apella-plus-thesis\scraper_py\scriptTest_file.py
Path: e:\GitHub_clones\Apella-plus-thesis\embeddings_py
Created Date: Saturday, December 18th 2021, 3:02:45 pm
Author: nikifori

Copyright (c) 2021 Your Company
'''
import json
import ijson
from itertools import islice
from os.path import splitext
import os.path


def save2json(json_fi: list, path2save: str):
    json_file = json.dumps(json_fi, indent=4)   #, ensure_ascii=False
    with open(fr'{path2save}', 'w', encoding='utf-8') as f:
        f.write(json_file)
            
def open_json(path2read: str):
    with open(fr'{path2read}', encoding="utf8") as json_file:
        dictionary = json.load(json_file)
    return dictionary


def my_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def read_authors(fname):
    _, file_extension = splitext(fname)
    if file_extension == '.json':
        with open(fname, encoding="utf8") as json_file:
            auth_dict = json.load(json_file)
    return auth_dict


def read_authors2(fname):
    _, file_extension = splitext(fname)
    auth_dict = []

    if file_extension == '.json':
        with open(fname) as f:
            objects = ijson.items(f, 'item')
            objects = islice(objects, 500)
            for author in objects:
                auth_dict.append(author)
    return auth_dict


if __name__ == '__main__':
    
    csd_out_completed_missing_2 = open_json(r"E:\GitHub_clones\Apella-plus-thesis\json_files\csd_out_with_abstract\csd_out_completed_missing_2.json")
    save2json(csd_out_completed_missing_2, path2save=r"E:\GitHub_clones\Apella-plus-thesis\json_files\csd_out_with_abstract\csd_out_completed_missing_22.json")
    test = open_json(r'.\specter_rankings\test_apella_data.json')
