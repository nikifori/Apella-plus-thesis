#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   unused_code.py
@Time    :   2022/01/11 18:57:01
@Author  :   nikifori 
@Version :   -
@Contact :   nikfkost@gmail.com
'''

def check4wrong_GS_id(authors_list: list, sus_authors: list):
    for author_dict in authors_list:
        if author_dict['Scholar id']!='Unknown':
            try:
                author = scholarly.search_author_id(author_dict["Scholar id"])
                print(author['name'])
                
            except Exception as error:
                print(error)
                sus_authors.append(author_dict['name'])
    
    # return sus_authors

def check4wrong_GS_id_parallel(authors_list: list, threads_num: int=1, proxy: str=None): 
    
    global sus_authors
    sus_authors = []
    threads = []
    
    if proxy:
        pg = ProxyGenerator()
        success = pg.SingleProxy(http = proxy)
        scholarly.use_proxy(pg)

    if threads_num>len(authors_list):threads_num=len(authors_list) 
    print(threads_num)
    chunked_list = chunks(authors_list, threads_num)
    for chunk in chunked_list:
        x = threading.Thread(target=check4wrong_GS_id, args=(chunk, sus_authors))
        threads.append(x)
        x.start()
    
    for thread in threads:
        thread.join()
    
    return sus_authors



if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # proxy = 'http://123.120.190.41'
    # from scholarly import scholarly, ProxyGenerator
    # x = check4wrong_GS_id_parallel(csd_in_ground_truth, 20, proxy)
    # y = check4wrong_GS_id_parallel(csd_out_ground_truth, 20, proxy)