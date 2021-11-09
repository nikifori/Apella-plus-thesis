import wget
import gzip
import shutil

def get_number(x):
    if not ((x >= 0) and (x <= 6000)):
        print("-------ERROR in get Number----------")

    if(x<10):
        return '00' + str(x)
    if(x<100):
        return '0' + str(x)
    if(x<1000):
        return str(x)

    return str(x)

def download_datasets(n=6000, out='../datasets'):
    base_url = 'https://s3-us-west-2.amazonaws.com/ai2-s2-research-public/open-corpus/2021-11-01/'

    for i in range(0,n+1):
        url = base_url + 's2-corpus-' + get_number(i) + '.gz'
        print('Downloading: ', url)
        file = wget.download(url, out=out)

def extract_corpus_file(n, base_path='../datasets/'):
    input_file = base_path + 's2-corpus-' + get_number(n) + '.gz'
    output_file = '../datasets/s2-corpus-' + get_number(n) + '.txt'

    # source https://stackoverflow.com/questions/31028815/how-to-unzip-gz-file-using-python
    with gzip.open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


n=6  #n = 6000 to download all corpus
download_datasets(n)
extract_corpus_file(0) # tester for extract corpus