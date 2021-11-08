'''
Filename: e:\GitHub_clones\Apella_plus_thesis\python_files\google_scholar_crawler.py
Path: e:\GitHub_clones\Apella_plus_thesis\python_files
Created Date: Saturday, November 6th 2021, 12:34:16 pm
Author: nikifori, bill

Copyright (c) 2021 Your Company
'''

import pandas as pd
from scholarly import scholarly, ProxyGenerator
import my_time as mt
import numpy as np
import threading

t = mt.my_time()
df_papers = pd.DataFrame(columns=["Title", "Publication Year", "Publication url", "Abstract"])


def splitter(i, n, thread_total):
    start = np.floor(i*n/thread_total)
    finish = np.floor((i+1)*n/thread_total)-1
    return int(start), int(finish)


def get_papers(papers, start, finish):

    for j in np.arange(start, finish+1):
        print(j)
        paper = papers[j]
        scholarly.fill(paper)
        try:
            df_papers.at[j, 'Title'] = paper["bib"]["title"] if "title" in paper["bib"] else None
            df_papers.at[j, 'Publication Year'] = int(paper["bib"]["pub_year"]) if "pub_year" in paper["bib"] else None
            df_papers.at[j, 'Publication url'] = paper["pub_url"] if "pub_url" in paper else None
            df_papers.at[j, 'Abstract'] = paper["bib"]["abstract"] if "abstract" in paper["bib"] else None
            print(df_papers.at[j, 'Title'])
        except:
            print("There is a problem")


def paper_scraper_threading(author_name, abstract=False):
    if type(author_name) is not str:
        print(f"Author name must be string: {type(author_name)} given.")
        return

    if type(abstract) is not bool:
        print(f"Abstract must be boolean: {type(abstract)} given.")
        return

    pg = ProxyGenerator()
    success = pg.SingleProxy(http="http://kartzafos22:1gnsjksaDs6FkTGT@proxy.packetstream.io:31112")
    scholarly.use_proxy(pg)

    search_query = scholarly.search_author(author_name)
    author = next(search_query)  # object
    scholarly.fill(author, sections=["publications"])

    papers = author["publications"]
    thread_total = 10
    print("Total publications:", len(papers))

    threads = list()

    # For each thread
    for thr in np.arange(0, thread_total):
        start, finish = splitter(thr, len(papers), thread_total)
        # get_papers(papers, start, finish)
        x = threading.Thread(target=get_papers, args=(papers, start, finish))
        threads.append(x)
        x.start()

    # Wait each thread to finish
    for index, thread in enumerate(threads):
        thread.join()


t.tic()
paper_scraper_threading("Grigorios Tsoumakas", abstract=False)  # .sort_values(by=['Publication Year'],
t.toc()

df_papers.to_csv(path_or_buf='test.csv',
            header=["Title", "Publication Year", "Publication url", "Abstract"],
            index=False)