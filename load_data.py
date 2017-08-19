import os
import tarfile
from six.moves import urllib

URL = 'http://ufldl.stanford.edu/housenumbers'
DATA_PATH = "data"


def download_data(filename, url=URL, data_path=DATA_PATH):
    tgz_path = os.path.join(data_path, filename)
    url_path = os.path.join(url, filename)
    urllib.request.urlretrieve(url_path, tgz_path)
    tgz = tarfile.open(tgz_path)
    tgz.extractall(path=data_path)
    tgz.close()

# Download data
print 'Downloading data...'

download_data("test.tar.gz")
download_data("train.tar.gz")

print 'Data was successfully downloaded and extracted!'
