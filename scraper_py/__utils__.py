#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   __utils__.py
@Time    :   2021/12/30 16:40:43
@Author  :   nikifori 
@Version :   -
@Contact :   nikfkost@gmail.com
'''
import json
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

if __name__ == '__main__':
    pass
    # csd_out_completed_missing_2 = open_json(r"E:\GitHub_clones\Apella-plus-thesis\json_files\csd_out_with_abstract\csd_out_completed_missing_2.json")
    # save2json(csd_out_completed_missing_2, path2save=r"E:\GitHub_clones\Apella-plus-thesis\json_files\csd_out_with_abstract\csd_out_completed_missing_22.json")
    # test = open_json(r'.\specter_rankings\test_apella_data.json')