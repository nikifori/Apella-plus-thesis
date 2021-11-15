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

def df_to_dict_parser(df):
    uni_replace = {"ΑΠΘ": "auth","ΕΜΠ": "ntua","Πανεπιστήμιο Πατρών": "upatras","ΕΜΠ": "ntua",
               "Πανεπιστήμιο Κύπρου": "ucy","Πανεπιστήμιο Δυτικής Μακεδονίας": "uowm","ΕΚΠΑ": "uoa",
               "Πανεπιστήμιο Θεσσαλίας": "uth","Δημοκρίτειο Πανεπιστήμιο Θράκης": "duth","Τεχνολογικό Πανεπιστήμιο Κύπρου": "cut",
               "ΙΤΕ": "forth","ΟΠΑ": "aueb","Πανεπιστήμιο Κρήτης": "uoc","Χαροκόπειο Πανεπιστήμιο": "hua","Πανεπιστήμιο Πειραιώς": "unipi", 
               "Πανεπιστήμιο Ιωαννίνων": "uoi", "Ελληνικό Ανοικτό Πανεπιστήμιο": "eap", "Ιόνιο Πανεπιστήμιο": "ionio", "Πανεπιστήμιο Αιγαίου": "aegean", 
               "Πολυτεχνείο Κρήτης": "tuc", "ΔΠΘ": "duth", "Εθνικό Καποδιστριακο Πανεπιστήμιο Αθηνών": "uoa",
               "Πανεπιστήμιο Πελοπονήσσου": "uop", "Οικονομικό Πανεπιστήμιο Αθηνών": "aueb",
               "ΕΚΕΤΑ": "certh", 'Ερευνητικό Κέντρο Καινοτομίας στις Τεχνολογίες της Πληροφορίας, των Επικοινωνιών & τηςΣ Γνώσης - "ΑΘΗΝΑ" ': "athena-innovation", 
               "Πανεπιστήμιο Πειραιά": "unipi", "Πανεπιστήμιο Μακεδονίας": "uom", 
               "Παν/μιο Πατρών ": "upatras", "Πανεπιστήμιο Πελοποννήσου": "uop", "Πολυτεχνειο Κρήτης": "tuc", "Ε.Κ. Αθηνά": "Ε.Κ. Αθηνά", 
               "Πανεπιστήμιο Δυτικής Αττικής  (ΤΕΙ Αθήνας)": "uniwa", "Πανεπιστημιο Ιωαννίνων": "uoi",
               'Εθνικό Κέντρο Ερευνας Φυσικών Επιστημών "ΔΗΜΟΚΡΙΤΟΣ"': "demokritos", "Πολυτεχνέιο Κρήτης": "tuc", 
               "Πανεπιστήμιο Πειραιως": "unipi", "Ανοικτό Πανεπιστήμιο Κύπρου": "ouc"}
    
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
        df["University"] = df["University"].replace(uni_replace)
        
    except Exception as error:
        print("There is a problem")
        print(error)
    
    dictionary = df.to_dict(orient="records")
    return dictionary





