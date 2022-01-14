import re
import math
import threading
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search

from names import compare_names, get_possible_names


# retrieve Google Scholar name/id
def get_scholar_name(author_dict: dict, proxy_dict: dict = None):

    author_name = author_dict['romanize name']
    auth_email = author_dict["University email domain"]
    possible_names = get_possible_names(author_name)
    unknown_fields = ["Semantic Scholar name", "Semantic Scholar id", "ResearchGate name", "ResearchGate url name/id", "ResearchGate url type"]

    for auth_name in possible_names:
        query = f"{auth_name} {auth_email} google scholar"
        links = search(query, num_results=5)
        if len(links) < 3: links = search(f"{auth_name} google scholar", num_results=5)
        for link in links:
            if "scholar" in link:
                try:
                    author_page = requests.get(link) if not proxy_dict else requests.get(link, proxies=proxy_dict)
                    soup = BeautifulSoup(author_page.content, "html.parser")
                    name = soup.find("div", id="gsc_prf_in").text
                    is_correct = compare_names(true_name=auth_name, test_name=name)
                    if not is_correct: continue
                    author_dict["Scholar name"] = name
                    temp_id = link.split("user=")[1]
                    author_dict["Scholar id"] = temp_id.split("&hl=")[0] if "&hl=" in temp_id else temp_id
                    for field in unknown_fields:
                        author_dict[field] = 'Unknown'
                    return author_dict
                except Exception as error: print(f"{error}\nError while parsing URL {link}")
            else:
                continue

    for auth_name in possible_names:
        query = f"site:scholar.google.com {auth_name}"
        links = search(query, num_results=5)
        for link in links:
            if "scholar" in link:
                try:
                    author_page = requests.get(link) if not proxy_dict else requests.get(link, proxies=proxy_dict)
                    soup = BeautifulSoup(author_page.content, "html.parser")
                    name = soup.find("div", id="gsc_prf_in").text
                    is_correct = compare_names(true_name=author_dict['romanize name'], test_name=name)
                    if not is_correct: continue
                    author_dict["Scholar name"] = name
                    temp_id = link.split("user=")[1]
                    author_dict["Scholar id"] = temp_id.split("&hl=")[0] if "&hl=" in temp_id else temp_id
                    for field in unknown_fields:
                        author_dict[field] = 'Unknown'
                    return author_dict
                except Exception as error: print(f"{error}\nError while parsing URL {link}")
            else:
                continue

    print("There is a problem in Google Scholar name/id retrieval")
    print("Unknown")
    author_dict["Scholar name"] = "Unknown"
    author_dict["Scholar id"] = "Unknown"
    return author_dict


# retrieve Semantic Scholar name/id
def get_semantic_name(author_dict: dict, proxy_dict: dict = None):
    auth_name = author_dict["romanize name"]
    query = f"{auth_name} semantic scholar"
    links = search(query, num_results=10)
    for link in links:
        if bool(re.search("^https://www.semanticscholar.org/author/.", link)):
            try:
                author_page = requests.get(link) if not proxy_dict else requests.get(link, proxies=proxy_dict)
                soup = BeautifulSoup(author_page.content, "html.parser")
                name = soup.find(class_="author-detail-card__author-name").text
                is_correct = compare_names(true_name=author_dict['romanize name'], test_name=name)
                if not is_correct: continue
                author_dict["Semantic Scholar name"] = name
                author_dict["Semantic Scholar id"] = link.split("/")[-1]
                author_dict["ResearchGate name"] = 'Unknown'
                author_dict["ResearchGate url name/id"] = 'Unknown'
                author_dict["ResearchGate url type"] = 'Unknown'

                return author_dict
            except Exception as error:
                print(error)
                print("Url does not start with https://www.semanticscholar.org/")
        else:
            continue

    print("There is a problem in Semantic Scholar name/id retrieval")
    print("Unknown")
    author_dict["Semantic Scholar name"] = "Unknown"
    author_dict["Semantic Scholar id"] = "Unknown"
    return author_dict


def scrape_page(soup_item, class_name):
    name = ""
    try: name = soup_item.find("div",class_=class_name).text
    except: pass
    return name


# retrieve ResearchGate name/id and type of author's page (profile or scientific-contributions)
def get_researchgate_name(author_dict: dict, proxy_dict: dict = None):
    # PROXY = {"http": "http://kartzafos22:1gnsjksaDs6FkTGT@proxy.packetstream.io:31112",
    #           "https": "http://kartzafos22:1gnsjksaDs6FkTGT@proxy.packetstream.io:31112"}
    auth_name = author_dict["romanize name"]
    auth_email = author_dict["University email domain"]
    query = f"{auth_name} {auth_email} researchgate"
    links = search(query, num_results=10)
    for link in links:
        time.sleep(1)
        if bool(re.search("^https://www.researchgate.net/profile/.", link)):
            author_page = requests.get(link) if not proxy_dict else requests.get(link, proxies=proxy_dict)
            soup = BeautifulSoup(author_page.content, "html.parser")
            name = scrape_page(soup, "nova-legacy-e-text nova-legacy-e-text--size-xxl nova-legacy-e-text--family-sans-serif nova-legacy-e-text--spacing-xxs nova-legacy-e-text--color-inherit fn")
            if name == "": name = scrape_page(soup, "nova-legacy-e-text nova-legacy-e-text--size-xxl nova-legacy-e-text--family-sans-serif nova-legacy-e-text--spacing-xxs nova-legacy-e-text--color-inherit")
            if name == "":
                print(f"Error parsing {link}")
                continue

            is_correct = compare_names(true_name=author_dict['romanize name'], test_name=name)
            if not is_correct: continue
            print(f"Research Gate {name}")
            author_dict["ResearchGate name"] = name
            author_dict["ResearchGate url name/id"] = link.split("/")[-1]
            author_dict["ResearchGate url type"] = link.split("/")[-2]
            return author_dict

        elif bool(re.search("^https://www.researchgate.net/scientific-contributions/.", link)):
            try:
                author_page = requests.get(link) if not proxy_dict else requests.get(link, proxies=proxy_dict)
                soup = BeautifulSoup(author_page.content, "html.parser")
                name = link.split("/")[-1].split("-")
                name.pop()
                name = " ".join(x for x in name)
                print(name)

                is_correct = compare_names(true_name=author_dict['romanize name'], test_name=name)
                if not is_correct: continue

                author_dict["ResearchGate name"] = name
                author_dict["ResearchGate url name/id"] = link.split("/")[-1]
                author_dict["ResearchGate url type"] = link.split("/")[-2]

                return author_dict
            except Exception as error:
                print(error)
                print("Url does not start with https://www.researchgate.net/scientific-contributions/")
        else:
            continue

    print("Author is not in ResearchGate.")
    print("Unknown")
    author_dict["ResearchGate name"] = "Unknown"
    author_dict["ResearchGate url name/id"] = "Unknown"
    return author_dict


def pipeline(chuck_of_authors: list, global_result_list, proxy_dict: dict = None):
    for author_dict in chuck_of_authors:
        print("Author:{}".format(author_dict['romanize name']))
        author = get_scholar_name(author_dict, proxy_dict)

        if author['Scholar name'] == 'Unknown':
            author = get_semantic_name(author_dict, proxy_dict)

            if author['Semantic Scholar name'] == 'Unknown':
                author = get_researchgate_name(author_dict, proxy_dict)

        global_result_list.append(author_dict)


def chunks(authors_list: list, threads: int):
    chunked_list = []
    n = len(authors_list)
    for i in range(threads):
        start = int(math.floor(i * n / threads))
        finish = int(math.floor((i + 1) * n / threads) - 1)
        chunked_list.append(authors_list[start:(finish + 1)])

    return chunked_list


def main(authors_list: list, threads_num: int = 1, proxy_dict: dict = None):
    if threads_num > len(authors_list): threads_num = len(authors_list)  # threads always equal or lower than number of authors
    print(threads_num)

    # list to save results
    global result_list
    result_list = []

    threads = []
    chunked_list = chunks(authors_list, threads_num)
    for chunk in chunked_list:
        x = threading.Thread(target=pipeline, args=(chunk, result_list, proxy_dict))
        threads.append(x)
        x.start()

    for thread in threads:
        thread.join()

    return result_list


def evaluate_results(exp_data: dict, ground_truth_data):
    """
    Calculate the correct author names/ids

    Parameters
    ----------
    exp_data : dict
        DESCRIPTION.
    ground_truth_data : DataFrame
        DESCRIPTION.

    Returns
    -------
    Percentage of correct authors names/ids.

    """

    exp_data = pd.DataFrame(exp_data)
    all_data = pd.concat([exp_data, ground_truth_data], ignore_index=True)
    correct = all_data.duplicated(subset=['name', 'romanize name', 'School-Department', 'University',
                                          'University email domain', 'Rank', 'Apella_id', 'Scholar name',
                                          'Scholar id', 'Semantic Scholar name', 'Semantic Scholar id',
                                          'ResearchGate name', 'ResearchGate url name/id',
                                          'ResearchGate url type']).sum()

    percentage = correct / len(ground_truth_data)
    return percentage


def find_differences(result, df_ground_truth):
    df_res = pd.DataFrame(result)
    df_res = df_res.sort_values(by=['romanize name'])
    df_ground_truth = df_ground_truth.sort_values(by=['romanize name'])

    test = pd.concat([df_res, df_ground_truth.drop("Search Engine label", axis=1)])
    test = test.reset_index(drop=True)
    test_gpby = test.groupby(list(test.columns))
    idx = [y[0] for y in test_gpby.groups.values() if len(y) == 1]
    test.reindex(idx)
    diffs = test.reindex(idx)

    return diffs


if __name__ == "__main__":

    # unlabeled data
    csd_in_test = pd.read_csv(r'..\csv_files\csd_data_in_unlabeled.csv').to_dict(orient='records')
    csd_out_test = pd.read_csv(r'..\csv_files\csd_data_out_unlabeled.csv').to_dict(orient='records')

    # ground truth data
    csd_in_ground_truth = pd.read_csv(r'..\csv_files\csd_data_in_processed_ground_truth_completed.csv')  # .to_dict(orient='records')
    csd_out_ground_truth = pd.read_csv(r'..\csv_files\csd_data_out_processed_ground_truth_completed.csv')  # .to_dict(orient='records')

    csd_test = [csd_in_test, csd_out_test]
    csd_ground_truth = [csd_in_ground_truth, csd_out_ground_truth]
    in_out = ['in', 'out']

    # proxy = {'http': "54.37.160.88:1080"}
    proxy = None
    n_threads = 20 if proxy else 1
    results = []
    percentage = []

    for cc, i in enumerate(csd_test):
        results.append(main(i, n_threads, proxy))
        percentage.append(evaluate_results(results[cc], csd_ground_truth[cc]))

        diffs = find_differences(results[cc], csd_ground_truth[cc])
        diffs.to_csv(f'diffs_{in_out[cc]}.csv')

    print(percentage)
