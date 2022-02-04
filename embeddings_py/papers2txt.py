#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   papers2txt.py
@Date    :   Fri Feb  4 23:22:38 2022
@Author  :   nikifori 
@Contact :   nikfkost@gmail.com
"""
import json
from utils import *
import os









if __name__ == "__main__":
    
    csd_in = open_json(r'..\json_files\csd_in_with_abstract\csd_in_completed_no_greek_rank.json')
    csd_out = open_json(r'..\json_files\csd_out_with_abstract\csd_out_completed_missing_2_no_greek_rank.json')
    
    papers = []
    csd_list = [csd_in, csd_out]
    for csd in csd_list:
        for author in csd:
            if 'Publications' in author:
                for paper in author.get('Publications'):
                    papers.append(paper)
    
    for paper in papers:
        paper['sentence'] = (paper.get('Title') or '') + '. ' + (paper.get('Abstract') or '')
        
    # delete every other paper feature, keep only sentence
    papers_dataset = []
    for paper in papers:
        papers_dataset.append(paper.get('sentence'))
    
    
    # write txt file line by line
    if os.path.exists('papers_dataset.txt'):   
        os.remove('papers_dataset.txt')
    for sentence in papers_dataset:
        with open('papers_dataset.txt', 'a', encoding="utf-8") as file:
            if sentence[-1]=='…': sentence = sentence[:-2] + '.'
            file.write('{}\n'.format(sentence))