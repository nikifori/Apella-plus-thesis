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


def parser(df):
    try:
        df = df[["Επώνυμο", "Όνομα", "Τμήμα", "Ίδρυμα", "Βαθμίδα", "Κωδικός ΑΠΕΛΛΑ"]]
        df["name"] = df["Όνομα"] + " " + df["Επώνυμο"]
        df = df[["name", "Τμήμα", "Ίδρυμα", "Βαθμίδα", "Κωδικός ΑΠΕΛΛΑ"]]
        df = df.rename(columns={"Τμήμα": "School-Department", "Ίδρυμα": "University",
                                "Βαθμίδα": "Rank", "Κωδικός ΑΠΕΛΛΑ": "Apella_id"})
    except:
        print("There is a problem")
        
    return df


csd_in = pd.read_excel(r"..\csv_files\csd_data_in.xlsx")
csd_out = pd.read_excel(r"..\csv_files\csd_data_out.xlsx")

# preprocessing
# csd_in
csd_in = parser(csd_in)
csd_out = parser(csd_out)

