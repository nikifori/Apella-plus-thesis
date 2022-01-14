import editdistance
import fuzzy
import phonetics
from fuzzywuzzy import fuzz
import pandas as pd
import re

nicknames_all = [
        ['Spyridon', 'Spyros'],
        ['Miltiadis', 'Miltos'],
        ['Konstantinos', 'Kostas'],
        ['Konstantinos', 'Costas'],
        ['Konstantinos', 'Konnos'],
        ['Konstantinos', 'Kon/nos'],
        ['Panagiotis', 'Panos'],
        ['Nikolaos', 'Nikos'],
        ['Stylianos', 'Stelios'],
        ['Tasoula', 'Anastasia'],
        ['Theodoulos', 'Theo'],
        ['Timoleon', 'Timos'],
        ['Aristeidis', 'Aris'],
        ['Christoforos', 'Christos'],
        ['Iraklis', 'Hercules'],
        ['Alkiviadis', 'Alkis'],
        ['Themistoklis', 'Themis'],
        ['Emmanouil', 'Manolis'],
        ['Athanasios', 'Thanasis']
    ]


def get_possible_names(auth_name):
    possible_names = [auth_name]

    if "ch" in auth_name or "Ch" in auth_name:
        auth_name_temp = re.sub("Ch", "H", auth_name)
        auth_name_temp = re.sub("ch", "h", auth_name_temp)
        possible_names.append(auth_name_temp)

    if "Nt" in auth_name or "nt" in auth_name:
        auth_name_temp = re.sub("Nt", "D", auth_name)
        auth_name_temp = re.sub("nt", "d", auth_name_temp)
        possible_names.append(auth_name_temp)

    if "ou" in auth_name:
        auth_name_temp = re.sub("ou", "u", auth_name)
        possible_names.append(auth_name_temp)

    possible_names += get_other_nicknames(auth_name)

    if "Tz" in auth_name:
        auth_name_temp = re.sub("Tz", "J", auth_name)
        possible_names.append(auth_name_temp)

    if "Mp" in auth_name or "mp" in auth_name:
        auth_name_temp = re.sub("Mp", "B", auth_name)
        auth_name_temp = re.sub("mp", "b", auth_name_temp)
        possible_names.append(auth_name_temp)

    if "F" in auth_name or "f" in auth_name:
        auth_name_temp = re.sub("F", "Ph", auth_name)
        auth_name_temp = re.sub("f", "ph", auth_name_temp)
        possible_names.append(auth_name_temp)

    return possible_names


def remove_extra_spaces(name):
    name = name.split(" ")
    name_l = list(filter(None, name))

    return " ".join(name_l)


def nickname_check(name_in: str):
    name = name_in.capitalize()

    for nicknames in nicknames_all:
        if name in nicknames:
            return nicknames[0]

    return name


def get_other_nicknames(name_in: str):
    name = name_in.capitalize()

    for nicknames in nicknames_all:
        if name in nicknames:
            res = []
            for n in nicknames:
                if name != n:
                    res.append(n)
            return res

    return []


def preprocess_name(name):
    name_pr = remove_extra_spaces(name)
    name_pr = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", name_pr)
    name_pr = re.sub("/", "", name_pr)
    name_pr = re.sub("-", " ", name_pr)
    name_pr = re.sub("\.", "", name_pr)
    name_pr = [nickname_check(w) for w in name_pr.split(" ") if len(w) >= 3]
    return name_pr


def compare_names(true_name, test_name):
    if test_name == "Unknown" or "":
        print(f"{true_name} NOT equals Unknown")
        return False

    true_name_pr = preprocess_name(true_name)
    test_name_pr = preprocess_name(test_name)
    starts_with_ch = any([w.startswith("Ch") for w in true_name_pr])
    matched_in_true = 0

    found = [0 for i in range(len(true_name_pr))]
    for w in test_name_pr:
        for i,q in enumerate(true_name_pr):
            if name_similarity(w, q) < 3 and not found[i]:
                matched_in_true += 1
                found[i] = 1
                break

    if matched_in_true >= 2 or matched_in_true == len(true_name_pr) or matched_in_true == len(test_name_pr):
        # print(f"{true_name} equals {test_name}")
        result = True
        return result
    else:
        print(f"{true_name} NOT equals {test_name}")
        result = False

    if starts_with_ch:
        matched_in_true = 0
        true_name_pr = [w if not w.startswith("Ch") else w[1:].capitalize() for w in true_name_pr]
        found = [0 for i in range(len(true_name_pr))]

        for w in test_name_pr:
            for i,q in enumerate(true_name_pr):
                if name_similarity(w, q) < 3 and not found[i]:
                    matched_in_true += 1
                    found[i] = 1
                    break

        if matched_in_true >= 2 or matched_in_true == len(true_name_pr):
            # print(f"{true_name} equals {test_name}")
            result = True
            return result
        else:
            print(f"{true_name} NOT equals {test_name}")
            result = False

    return result


def name_similarity(name1, name2):
    return editdistance.eval(fuzzy.nysiis(name1), fuzzy.nysiis(name2))


def name_similarity2(name1, name2):
    code1 = phonetics.metaphone(name1)
    code2 = phonetics.metaphone(name2)
    return fuzz.ratio(code1, code2)


if __name__ == '__main__':

    name1 = "Evripidis Nikolaos Chelas"
    name2 = "Evripidis Chatzikraniotis"
    print(compare_names(name1, name2))
