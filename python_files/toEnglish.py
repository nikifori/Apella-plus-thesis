# -*- coding: utf-8 -*-
'''
Filename: e:\GitHub_clones\Apella-plus-thesis\python_files\toEnglish.py
Path: e:\GitHub_clones\Apella-plus-thesis\python_files
Created Date: Friday, November 12th 2021, 4:12:33 pm
Author: nikifori

Copyright (c) 2021 Your Company
'''


def toEnglish(name: str):
    grCaps = "ΑΆΒΓΔΕΈΖΗΉΘΙΊΪΚΛΜΝΞΟΌΠΡΣΤΥΎΫΦΧΨΩΏ"
    replacements = [
        {"greek": 'αι', "greeklish": 'ai'},
        {"greek": 'αί', "greeklish": 'ai'},
        {"greek": 'οι', "greeklish": 'oi'},
        {"greek": 'οί', "greeklish": 'oi'},
        {"greek": 'ου', "greeklish": 'ou'},
        {"greek": 'ού', "greeklish": 'ou'},
        {"greek": 'ει', "greeklish": 'ei'},
        {"greek": 'εί', "greeklish": 'ei'},
        {"greek": 'αυ', "fivi": 1},
        {"greek": 'αύ', "fivi": 1},
        {"greek": 'ευ', "fivi": 1},
        {"greek": 'εύ', "fivi": 1},
        {"greek": 'ηυ', "fivi": 1},
        {"greek": 'ηύ', "fivi": 1},
        {"greek": 'ντ', "greeklish": 'nt'},
        {"greek": 'μπ', "bi": 1},
        {"greek": 'τσ', "greeklish": 'ts'},
        {"greek": 'τς', "greeklish": 'ts'},
        {"greek": 'ΤΣ', "greeklish": 'ts'},
        {"greek": 'τζ', "greeklish": 'tz'},
        {"greek": 'γγ', "greeklish": 'ng'},
        {"greek": 'γκ', "greeklish": 'gk'},
        {"greek": 'θ', "greeklish": 'th'},
        {"greek": 'χ', "greeklish": 'ch'},
        {"greek": 'ψ', "greeklish": 'ps'},
    ]

