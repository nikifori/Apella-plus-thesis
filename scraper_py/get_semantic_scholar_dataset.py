import os
import wget
import gzip
import shutil
import yaml
import json


def get_number(x):
    if not ((x >= 0) and (x <= 6000)):
        print("-------ERROR in get Number----------")

    if x < 10:
        return '00' + str(x)
    if x < 100:
        return '0' + str(x)
    if x < 1000:
        return str(x)
    return str(x)


def download_datasets(n= list(range(0,6000+1)), out='../datasets'):
    base_url = 'https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2021-11-01/'

    iterations = [n] if isinstance(n, int) else n

    for paper_ind in iterations:
        url = base_url + 's2-corpus-' + get_number(paper_ind) + '.gz'
        print('Downloading: ', url)
        file = wget.download(url, out=out)


def extract_corpus_file(n, base_path='../datasets/'):
    input_file = base_path + 's2-corpus-' + get_number(n) + '.gz'
    output_file = '../datasets/s2-corpus-' + get_number(n) + '.txt'

    # source https://stackoverflow.com/questions/31028815/how-to-unzip-gz-file-using-python
    with gzip.open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    f = open(output_file, encoding="utf8")
    line = f.readline()
    papers = []
    # attrs = ['id', 'title', 'authors', 'paperAbstract', 'year', 'fieldsOfStudy']
    attrs = ['id', 'title', 'authors', 'year', 'fieldsOfStudy']

    paper_now = 0
    total_paper = 33010

    for paper_ind in range(0, 33009):
        if not line: break

        d = yaml.load(line, Loader=yaml.FullLoader)

        paper = {}
        for attr in attrs: paper[attr] = d[attr]
        paper['hasAbstract'] = 0 if d['paperAbstract'] == "" else 1

        papers.append(paper)
        line = f.readline()
        if paper_now % 1000 == 0:
            print('{}/{}'.format(paper_now, total_paper))
        paper_now += 1

    with open('../datasets/semantic_scholar_papers_{}.json'.format(n), 'a', encoding='utf-8') as f:
        json.dump(papers, f, ensure_ascii=False, indent=4)

    os.remove(output_file)
    os.remove(input_file)
    f.close()


if __name__ == "__main__":
    n = 2 # 6000 for all sub_datasets

    for i in range(0, n):
        # download each dataset
        download_datasets(i)

        # extract paper info to the appropriate json
        extract_corpus_file(i)
