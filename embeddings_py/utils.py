'''
Filename: e:\GitHub_clones\Apella-plus-thesis\scraper_py\scriptTest_file.py
Path: e:\GitHub_clones\Apella-plus-thesis\embeddings_py
Created Date: Saturday, December 18th 2021, 3:02:45 pm
Author: nikifori

Copyright (c) 2021 Your Company
'''
import json
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


def mkdirs(path_name):
    os.makedirs(path_name, exist_ok=True)

    if not os.path.exists(path_name):
        print(f"---Error creating directory {path_name}")
    else: pass


def read_authors(fname):
    _, file_extension = splitext(fname)
    if file_extension == '.json':
        with open(fname, encoding="utf8") as json_file:
            auth_dict = json.load(json_file)
    return auth_dict


def find_author_rank(author):
    # all_ranks = ['Αναπληρωτής καθηγητής', 'Αναπληρωτής Καθηγητής ', 'Καθηγητής', 'Αναπληρωτής Καθηγητής', 'Διευθυντής Ερευνών',
    #              'Κύριος Ερευνητής', 'Καθηγήτρια', 'Επίκουρος Καθηγητής (Μόνιμος)', 'Professor', ' Καθηγητής', 'Kαθηγητής']

    rank_3 = ['Επίκουρος Καθηγητής (Μόνιμος)']
    rank_2 = ['Αναπληρωτής καθηγητής', 'Αναπληρωτής Καθηγητής ', 'Αναπληρωτής Καθηγητής', 'Κύριος Ερευνητής']
    rank_1 = ['Καθηγητής', 'Διευθυντής Ερευνών', 'Καθηγήτρια', 'Professor', ' Καθηγητής', 'Kαθηγητής']

    if "Rank" in author:
        if author["Rank"] in [1, 2, 3]:
            return author["Rank"]
        else:
            true_rank = author["Rank"]
    else:
        return 1

    true_rank_int = 1

    if true_rank in rank_1: true_rank_int = 1
    if true_rank in rank_2: true_rank_int = 2
    if true_rank in rank_3: true_rank_int = 3

    return true_rank_int


def create_position_object(title, description, targets_in, targets_in_standby, targets_out, targets_out_standby,
                           position_rank):
    position_dict = {
        "title": title,
        "description": description,
        "rank": position_rank,
        "targets_in": targets_in,
        "targets_in_standby": targets_in_standby,
        "targets_out": targets_out,
        "targets_out_standby": targets_out_standby

    }

    mkdirs(r"./positions/")
    save2json(position_dict, path2save=r"./positions/{}.json".format(title))


def get_positions(path):
    titles = []
    descriptions = []
    authors_targets_in = []
    authors_targets_in_standby = []
    authors_targets_out = []
    authors_targets_out_standby = []
    position_ranks = []

    data = open_json(path)

    for i in data[:-1]:  # ignore last one without target_lists
        titles.append(i.get("title"))
        descriptions.append(i.get("description"))
        authors_targets_in.append(i.get("targets_in"))
        authors_targets_in_standby.append(i.get("targets_in_standby"))
        authors_targets_out.append(i.get("targets_out"))
        authors_targets_out_standby.append(i.get("targets_out_standby"))
        position_ranks.append(i.get("rank"))

    for i, title in enumerate(titles):
        create_position_object(title, descriptions[i],
                               targets_in=authors_targets_in[i],
                               targets_in_standby=authors_targets_in_standby[i],
                               targets_out=authors_targets_out[i],
                               targets_out_standby=authors_targets_out_standby[i],
                               position_rank=position_ranks[i])

    return titles, descriptions, authors_targets_in, authors_targets_in_standby, authors_targets_out, authors_targets_out_standby, position_ranks



if __name__ == '__main__':
    pass
    # csd_out_completed_missing_2 = open_json(r"E:\GitHub_clones\Apella-plus-thesis\json_files\csd_out_with_abstract\csd_out_completed_missing_2.json")
    # save2json(csd_out_completed_missing_2, path2save=r"E:\GitHub_clones\Apella-plus-thesis\json_files\csd_out_with_abstract\csd_out_completed_missing_22.json")
    # test = open_json(r'.\specter_rankings\test_apella_data.json')
