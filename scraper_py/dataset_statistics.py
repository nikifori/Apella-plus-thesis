#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   dataset_statistics.py
@Time    :   2022/01/05 19:53:48
@Author  :   nikifori 
@Version :   -
@Contact :   nikfkost@gmail.com
'''


from __utils__ import *
import pandas as pd
import matplotlib.pyplot as plt

def publications_metrics(authors_list_1: list, authors_list_2: list = None):
    authors_list = authors_list_1 + authors_list_2 if authors_list_2 else authors_list_1
    pub_sum = 0
    average_pub_count = 0
    for author in authors_list:
        if "Publications" in author:
            if author["Scholar name"]!="Unknown": print('{}   '.format(len(author.get("Publications"))) + '{}     '.format(author["Scholar id"])+ '{}'.format(author["Scholar name"]))
            pub_sum += len(author.get("Publications"))
    
    average_pub_count = pub_sum/len(authors_list)
    
    print('Publications sum is: {}\nAverage Publications per Author is: {}'.format(pub_sum, average_pub_count))
    return pub_sum, average_pub_count


def plot_pub_distribution(authors_list_1: list, authors_list_2: list = None, title: str = None):
    """
    Plot publications number distribution
    
    Parameters
    ----------
    authors_list_1 : list
        main_author_list
    authors_list_2 : list, optional
        possibility for second list concatenation. The default is None.

    Returns
    -------
    None.

    """
    authors_list = authors_list_1 + authors_list_2 if authors_list_2 else authors_list_1
    pub_values = [len(x.get('Publications')) for x in authors_list if 'Publications' in x]
    plt.figure(num='Publications per Author Distribution')
    plt.hist(pub_values, bins=30)
    if title: plt.title(title)
    plt.show()

def boxplot_pub(authors_list_1: list, authors_list_2: list = None, title: str = None):
    """
    Plot publications number distribution
    
    Parameters
    ----------
    authors_list_1 : list
        main_author_list
    authors_list_2 : list, optional
        possibility for second list concatenation. The default is None.

    Returns
    -------
    None.

    """
    authors_list = authors_list_1 + authors_list_2 if authors_list_2 else authors_list_1
    pub_values = [len(x.get('Publications')) for x in authors_list if 'Publications' in x]
    plt.figure(num='Publications per Author Distribution')
    green_diamond = dict(markerfacecolor='g', marker='D')
    plt.boxplot(pub_values, showmeans=True, meanline=True, autorange=True, vert=False, flierprops=green_diamond, whis=[5, 95])
    if title: plt.title(title)
    plt.show()

if __name__ == "__main__":
    
    
    csd_in = open_json("..\json_files\csd_in_with_abstract\csd_in_completed_no_greek_rank.json")
    csd_out = open_json("..\json_files\csd_out_with_abstract\csd_out_completed_missing_2_no_greek_rank.json")
    
    csd_in_pubs_sum, csd_in_average_pub_num = publications_metrics(csd_in)
    csd_out_pubs_sum, csd_out_average_pub_num = publications_metrics(csd_out)
    csd_all_pubs_sum, csd_all_average_pub_num = publications_metrics(csd_in, csd_out)
    
    plot_pub_distribution(csd_in, csd_out, title='Publications per Author Distribution')
    boxplot_pub(csd_in, csd_out, title='Publications per Author BoxPlot')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    