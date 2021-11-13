'''
Filename: e:\GitHub_clones\Apella-plus-thesis\python_files\csd_csv_parser.py
Path: e:\GitHub_clones\Apella-plus-thesis\python_files
Created Date: Friday, November 12th 2021, 10:03:16 pm
Author: nikifori

Copyright (c) 2021 Your Company
'''
import pandas as pd
from langdetect import detect
from romanize import romanize
from google_search import  get_scholar_name
import time


def df_to_dict_parser(df):
    try:
        df = df[["Επώνυμο", "Όνομα", "Τμήμα", "Ίδρυμα", "Βαθμίδα", "Κωδικός ΑΠΕΛΛΑ"]]
        df["name"] = df["Όνομα"] + " " + df["Επώνυμο"]
        df = df[["name", "Τμήμα", "Ίδρυμα", "Βαθμίδα", "Κωδικός ΑΠΕΛΛΑ"]]
        df = df.rename(columns={"Τμήμα": "School-Department", "Ίδρυμα": "University",
                                "Βαθμίδα": "Rank", "Κωδικός ΑΠΕΛΛΑ": "Apella_id"})
        roman_list = []
        for i in df["name"]:
            if detect(i)=="el":
                roman_list.append(romanize(i))
            else: roman_list.append(i)
        
        romanize_df = pd.DataFrame(roman_list, columns=["romanize name"])
        df.insert(1, "romanize name", romanize_df)
        
    except Exception as error:
        print("There is a problem")
        print(error)
    
    dictionary = df.to_dict(orient="records")
    return dictionary


csd_in = pd.read_excel(r"..\csv_files\csd_data_in.xlsx")
csd_out = pd.read_excel(r"..\csv_files\csd_data_out.xlsx")

# preprocessing
# csd_in
csd_in = df_to_dict_parser(csd_in)
csd_out = df_to_dict_parser(csd_out)

# find author name in google scholar
for i in range(len(csd_in)):
    time.sleep(1.5)
    get_scholar_name(csd_in[i])


