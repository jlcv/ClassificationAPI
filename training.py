#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import os
from os import environ
from os.path import expanduser
from os.path import join
from os.path import exists
from os.path import splitext
from os import listdir
import tarfile
import pickle
import shutil
import re
import codecs
import numbers
import json
from string import punctuation

import nltk
from nltk import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from urllib2 import urlopen
from os.path import isdir

# Set encoding
reload(sys)
sys.setdefaultencoding('UTF8')

ARCHIVE_NAME = "values.tar.gz"
CACHE_NAME = "values.pkz"
TRAIN_FOLDER = "values_train"
TEST_FOLDER = "values_test"

def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

class Bunch(dict):
    def __init__(self, **kwargs):
        super(Bunch, self).__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setstate__(self, state):
        pass

def load_files(container_path, description=None, categories=None,
               load_content=True, shuffle=True, encoding=None,
               decode_error='strict', random_state=0):

    target = []
    target_names = []
    filenames = []

    folders = [f for f in sorted(listdir(container_path))
               if isdir(join(container_path, f))]

    if categories is not None:
        folders = [f for f in folders if f in categories]

    for label, folder in enumerate(folders):
        target_names.append(folder)
        folder_path = join(container_path, folder)
        documents = [join(folder_path, d)
                     for d in sorted(listdir(folder_path))]
        target.extend(len(documents) * [label])
        filenames.extend(documents)

    
    filenames = np.array(filenames)
    target = np.array(target)

    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(filenames.shape[0])
        random_state.shuffle(indices)
        filenames = filenames[indices]
        target = target[indices]

    if load_content:
        data = []
        for filename in filenames:
            with open(filename, 'rb') as f:
                data.append(f.read())
        if encoding is not None:
            data = [d.decode(encoding, decode_error) for d in data]
        return Bunch(data=data,
                     filenames=filenames,
                     target_names=target_names,
                     target=target,
                     DESCR=description)

    return Bunch(filenames=filenames,
                 target_names=target_names,
                 target=target,
                 DESCR=description)


def download_values(target_dir, cache_path):
    archive_path = os.path.join(target_dir, ARCHIVE_NAME)
    train_path = os.path.join(target_dir, TRAIN_FOLDER)
    test_path = os.path.join(target_dir, TEST_FOLDER)

    print(archive_path)
    print(train_path)
    print(test_path)

    cache = dict(train=load_files(train_path, encoding='latin1'),
                 test=load_files(test_path, encoding='latin1'))
    compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
    with open(cache_path, 'wb') as f:
        f.write(compressed_content)

    return cache

def get_data_home(data_home=None, project_name=""):
    if data_home is None:
        data_home = os.path.dirname(os.path.abspath(__file__))
    data_home = expanduser(data_home)
    data_home = os.path.join(data_home, project_name)
    if not exists(data_home):
        os.makedirs(data_home)
    return data_home


def _pkl_filepath(*args, **kwargs):
    py3_suffix = kwargs.get("py3_suffix", "_py3")
    basename, ext = splitext(args[-1])
    if sys.version_info[0] >= 3:
        basename += py3_suffix
    new_args = args[:-1] + (basename + ext,)
    return join(*new_args)


def fetch_values(data_home=None, subset='train', categories=None, project_name="",
                       shuffle=True, random_state=42,
                       download_if_missing=True):

    data_home = get_data_home(data_home=data_home, project_name=project_name)
    cache_path = _pkl_filepath(data_home, CACHE_NAME)
    print(data_home)
    print(cache_path)
    cache = None

    if cache is None:
        if download_if_missing:
            cache = download_values(target_dir=data_home,
                                          cache_path=cache_path)
        else:
            raise IOError('Values dataset not found')

    if subset in ('train', 'test'):
        data = cache[subset]
    elif subset == 'all':
        data_lst = list()
        target = list()
        filenames = list()
        for subset in ('train', 'test'):
            data = cache[subset]
            data_lst.extend(data.data)
            target.extend(data.target)
            filenames.extend(data.filenames)

        data.data = data_lst
        data.target = np.array(target)
        data.filenames = np.array(filenames)
    else:
        raise ValueError(
            "subset can only be 'train', 'test' or 'all', got '%s'" % subset)

    data.description = 'the values dataset'

    if categories is not None:
        labels = [(data.target_names.index(cat), cat) for cat in categories]
        labels.sort()
        labels, categories = zip(*labels)
        mask = np.in1d(data.target, labels)
        data.filenames = data.filenames[mask]
        data.target = data.target[mask]
        data.target = np.searchsorted(labels, data.target)
        data.target_names = list(categories)
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[mask]
        data.data = data_lst.tolist()

    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(data.target.shape[0])
        random_state.shuffle(indices)
        data.filenames = data.filenames[indices]
        data.target = data.target[indices]
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[indices]
        data.data = data_lst.tolist()

    return data

def stem_tokens(tokens, stemmer):  
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):  
    text = ''.join([c for c in text if c not in non_words])
    tokens =  word_tokenize(text)

    try:
        stems = stem_tokens(tokens, stemmer)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


def get_training_results(categories=None, project_name=""):
    data_train = fetch_values(subset='train', categories=categories, project_name=project_name,
                                    shuffle=True, random_state=42)

    data_test = fetch_values(subset='test', categories=categories, project_name=project_name,
                                   shuffle=True, random_state=42)

    target_names = data_train.target_names

    y_train, y_test = data_train.target, data_test.target

    t0 = time()
    spanish_stemmer = SnowballStemmer('spanish')
    non_words = list(punctuation)  
    non_words.extend(['¿', '¡'])  
    non_words.extend(map(str,range(10)))
    stopwords_list = stopwords.words('spanish')
    stopwords_list.extend([
            'hola',
            'respuesta',
            'dias',
            'saludos',
            'buenos',
            'favor',
            'dia',
            'quisiera',
            'hoy',
            'buen',
            'espero',
            'muchas',
            'debido',
            'petición',
            'solicitar',
            'gracias',
            'antemano',
            'necesito',
            'tardes',
            'tener',
            'levantar',
            'motivo',
            'tiempo',
            'posible',
            'pedir',
            'buenas',
            'agradezco',
            'apoyo',
            ''
            ])

    vectorizer = HashingVectorizer(stop_words=stopwords_list, non_negative=True,
                                        strip_accents=unicode)
    X_train = vectorizer.transform(data_train.data)

    duration = time() - t0

    t0 = time()
    X_test = vectorizer.transform(data_test.data)
    duration = time() - t0

    feature_names = None

    if feature_names:
        feature_names = np.asarray(feature_names)


    results_data = {}
    results_data['multinomial_nb'] = benchmark(MultinomialNB(alpha=.01), X_train, X_test, y_train, y_test)[1]
    results_data['bernoulli_nb'] = benchmark(BernoulliNB(alpha=.01), X_train, X_test, y_train, y_test)[1]
    json_data = json.dumps(results_data)

    return json_data


def benchmark(clf, X_train, X_test, y_train, y_test):
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    score = metrics.accuracy_score(y_test, pred)

    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time































