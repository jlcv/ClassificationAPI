#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
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

ARCHIVE_NAME = "email_tickets.tar.gz"
CACHE_NAME = "email_tickets.pkz"
TRAIN_FOLDER = "email_tickets_train"
TEST_FOLDER = "email_tickets_test"

cache = None
target_names = []

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None (or np.random), return the RandomState singleton used
    by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

class Bunch(dict):
    """Container object for datasets

    Dictionary-like object that exposes its keys as attributes.

    >>> b = Bunch(a=1, b=2)
    >>> b['b']
    2
    >>> b.b
    2
    >>> b.a = 3
    >>> b['a']
    3
    >>> b.c = 6
    >>> b['c']
    6

    """

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
        # Bunch pickles generated with scikit-learn 0.16.* have an non
        # empty __dict__. This causes a surprising behaviour when
        # loading these pickles scikit-learn 0.17: reading bunch.key
        # uses __dict__ but assigning to bunch.key use __setattr__ and
        # only changes bunch['key']. More details can be found at:
        # https://github.com/scikit-learn/scikit-learn/issues/6196.
        # Overriding __setstate__ to be a noop has the effect of
        # ignoring the pickled __dict__
        pass


def download_email_tickets(target_dir, cache_path):
    """Load email data and stored it as a zipped pickle."""
    archive_path = os.path.join(target_dir, ARCHIVE_NAME)
    train_path = os.path.join(target_dir, TRAIN_FOLDER)
    test_path = os.path.join(target_dir, TEST_FOLDER)

    print(archive_path)
    print(train_path)
    print(test_path)

    # Store a zipped pickle
    cache = dict(train=load_files(train_path, encoding='latin1'),
                 test=load_files(test_path, encoding='latin1'))
    compressed_content = codecs.encode(pickle.dumps(cache), 'zlib_codec')
    with open(cache_path, 'wb') as f:
        f.write(compressed_content)

    return cache

def get_data_home(data_home=None):
    """Return the path of the scikit-learn data dir.

    This folder is used by some large dataset loaders to avoid
    downloading the data several times.

    By default the data dir is set to a folder named 'scikit_learn_data'
    in the user home folder.

    Alternatively, it can be set by the 'SCIKIT_LEARN_DATA' environment
    variable or programmatically by giving an explicit folder path. The
    '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.
    """
    if data_home is None:
        data_home = os.path.dirname(os.path.abspath(__file__))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def _pkl_filepath(*args, **kwargs):
    """Ensure different filenames for Python 2 and Python 3 pickles

    An object pickled under Python 3 cannot be loaded under Python 2.
    An object pickled under Python 2 can sometimes not be loaded
    correctly under Python 3 because some Python 2 strings are decoded as
    Python 3 strings which can be problematic for objects that use Python 2
    strings as byte buffers for numerical data instead of "real" strings.

    Therefore, dataset loaders in scikit-learn use different files for pickles
    manages by Python 2 and Python 3 in the same SCIKIT_LEARN_DATA folder so
    as to avoid conflicts.

    args[-1] is expected to be the ".pkl" filename. Under Python 3, a
    suffix is inserted before the extension to s

    _pkl_filepath('/path/to/folder', 'filename.pkl') returns:
      - /path/to/folder/filename.pkl under Python 2
      - /path/to/folder/filename_py3.pkl under Python 3+

    """
    py3_suffix = kwargs.get("py3_suffix", "_py3")
    basename, ext = splitext(args[-1])
    if sys.version_info[0] >= 3:
        basename += py3_suffix
    new_args = args[:-1] + (basename + ext,)
    return join(*new_args)


def fetch_email_tickets(data_home=None, subset='train', categories=None,
                       shuffle=True, random_state=42,
                       download_if_missing=True):
    """Load the filenames and data from the email dataset.

    Parameters
    ----------
    subset : 'train' or 'test', 'all', optional
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

    data_home : optional, default: None
        Specify a download and cache folder for the datasets. If None,
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    categories : None or collection of string or unicode
        If None (default), load all the categories.
        If not None, list of category names to load (other categories
        ignored).

    shuffle : bool, optional
        Whether or not to shuffle the data: might be important for models that
        make the assumption that the samples are independent and identically
        distributed (i.i.d.), such as stochastic gradient descent.

    random_state : numpy random number generator or seed integer
        Used to shuffle the dataset.

    download_if_missing : optional, True by default
        If False, raise an IOError if the data is not locally available
        instead of trying to download the data from the source site.

    remove : tuple
        May contain any subset of ('headers', 'footers', 'quotes'). Each of
        these are kinds of text that will be detected and removed from the
        texts, preventing classifiers from overfitting on
        metadata.

        'headers' removes headers, 'footers' removes blocks at the
        ends of mails that look like signatures, and 'quotes' removes lines
        that appear to be quoting another mail.

        'headers' follows an exact standard; the other filters are not always
        correct.
    """

    data_home = get_data_home(data_home=data_home)
    cache_path = _pkl_filepath(data_home, CACHE_NAME)
    email_tickets_home = os.path.join(data_home, "email_tickets_home")
    print(data_home)
    print(cache_path)
    print(email_tickets_home)
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                compressed_content = f.read()
            uncompressed_content = codecs.decode(
                compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        except Exception as e:
            print(80 * '_')
            print('Cache loading failed')
            print(80 * '_')
            print(e)

    if cache is None:
        if download_if_missing:
            cache = download_email_tickets(target_dir=email_tickets_home,
                                          cache_path=cache_path)
        else:
            raise IOError('Email tickets dataset not found')

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

    data.description = 'the email tickets dataset'

    # if 'headers' in remove:
    #     data.data = [strip_header(text) for text in data.data]
    # if 'footers' in remove:
    #     data.data = [strip_footer(text) for text in data.data]
    # if 'quotes' in remove:
    #     data.data = [strip_quoting(text) for text in data.data]

    if categories is not None:
        # print(data.target_names)
        labels = [(data.target_names.index(cat), cat) for cat in categories]
        # Sort the categories to have the ordering of the labels
        labels.sort()
        labels, categories = zip(*labels)
        mask = np.in1d(data.target, labels)
        data.filenames = data.filenames[mask]
        data.target = data.target[mask]
        # searchsorted to have continuous labels
        data.target = np.searchsorted(labels, data.target)
        data.target_names = list(categories)
        # Use an object array to shuffle: avoids memory copy
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[mask]
        data.data = data_lst.tolist()

    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(data.target.shape[0])
        random_state.shuffle(indices)
        data.filenames = data.filenames[indices]
        data.target = data.target[indices]
        # Use an object array to shuffle: avoids memory copy
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
    # remove punctuation
    text = ''.join([c for c in text if c not in non_words])
    # tokenize
    tokens =  word_tokenize(text)

    # stem
    try:
        stems = stem_tokens(tokens, stemmer)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems

def fetch_email_tickets_vectorized(subset="train", data_home=None):
    """Load the dataset and transform it into tf-idf vectors.

    This is a convenience function; the tf-idf transformation is done using the
    default settings for `sklearn.feature_extraction.text.Vectorizer`. For more
    advanced usage (stopword filtering, n-gram extraction, etc.), combine
    fetch_emails with a custom `Vectorizer` or `CountVectorizer`.

    Parameters
    ----------

    subset : 'train' or 'test', 'all', optional
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

    data_home : optional, default: None
        Specify an download and cache folder for the datasets. If None,
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    remove : tuple
        May contain any subset of ('headers', 'footers', 'quotes'). Each of
        these are kinds of text that will be detected and removed from the
        mails, preventing classifiers from overfitting on
        metadata.

        'headers' removes headers, 'footers' removes blocks at the
        ends of mails that look like signatures, and 'quotes' removes lines
        that appear to be quoting another mail.

    Returns
    -------

    bunch : Bunch object
        bunch.data: sparse matrix, shape [n_samples, n_features]
        bunch.target: array, shape [n_samples]
        bunch.target_names: list, length [n_classes]
    """
    data_home = get_data_home(data_home=data_home)
    filebase = 'email_tickets_vectorized'
    # if remove:
    #     filebase += 'remove-' + ('-'.join(remove))
    target_file = _pkl_filepath(data_home, filebase + ".pkl")

    # we shuffle but use a fixed seed for the memoization
    data_train = fetch_email_tickets(data_home=data_home,
                                    subset='train',
                                    categories=None,
                                    shuffle=True,
                                    random_state=12)

    data_test = fetch_email_tickets(data_home=data_home,
                                   subset='test',
                                   categories=None,
                                   shuffle=True,
                                   random_state=12)

    if os.path.exists(target_file):
        X_train, X_test = joblib.load(target_file)
    else:
        spanish_stemmer = SnowballStemmer('spanish')
        non_words = list(punctuation)  
        non_words.extend(['¿', '¡'])  
        non_words.extend(map(str,range(10)))
        spanish_stopwords = stopwords.words('spanish')
        vectorizer = CountVectorizer(dtype=np.int16, lowercase=True, stop_words=spanish_stopwords, strip_accents=unicode)
        vectorizer._validate_vocabulary()
        X_train = vectorizer.fit_transform(data_train.data).tocsr()
        X_test = vectorizer.transform(data_test.data).tocsr()
        joblib.dump((X_train, X_test), target_file, compress=9)

    # the data is stored as int16 for compactness
    # but normalize needs floats
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    normalize(X_train, copy=False)
    normalize(X_test, copy=False)

    target_names.extend(data_train.target_names)

    if subset == "train":
        data = X_train
        target = data_train.target
    elif subset == "test":
        data = X_test
        target = data_test.target
    elif subset == "all":
        data = sp.vstack((X_train, X_test)).tocsr()
        target = np.concatenate((data_train.target, data_test.target))
    else:
        raise ValueError("%r is not a valid subset: should be one of "
                         "['train', 'test', 'all']" % subset)

    return Bunch(data=data, target=target, target_names=target_names)

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6


def post_prediction(email_body="", categories=[]):
    ###############################################################################
    # Load some categories from the training set
    # if opts.filtered:
    #     remove = ('headers', 'footers', 'quotes')
    # else:
    #     remove = ()

    data_train = fetch_email_tickets(subset='train', categories=categories,
                                    shuffle=True, random_state=42)

    data_test = fetch_email_tickets(subset='test', categories=categories,
                                   shuffle=True, random_state=42)

    # order of labels in `target_names` can be different from `categories`
    target_names.extend(data_train.target_names)

    # split a training set and a test set
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

    # vectorizer = CountVectorizer(stop_words=stopwords_list,
    #                                     strip_accents=unicode)
    # X_train = vectorizer.fit_transform(data_train.data)

    # duration = time() - t0

    # t0 = time()
    # X_test = vectorizer.transform(data_test.data)
    # duration = time() - t0

    # mapping from integer feature name to original token string
    feature_names = None

    if feature_names:
        feature_names = np.asarray(feature_names)

    prediction_data = {}
    prediction_data['text'] = email_body
    prediction_data['multinomial_nb'] = predict(MultinomialNB(), data_train.data, data_test.data, y_train, y_test, email_body)
    prediction_data['bernoulli_nb'] = predict(BernoulliNB(), data_train.data, data_test.data, y_train, y_test, email_body)
    json_data = json.dumps(prediction_data)

    return json_data


###############################################################################
# Predict classifiers
def predict(clf, X_train, X_test, y_train, y_test, email_body):
    t0 = time()
    text_clf = Pipeline([('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', clf)])
    text_clf.fit(X_train, y_train)
    train_time = time() - t0
    
    texts = [
        u'%s' % (email_body)
    ]

    predicted = text_clf.predict(texts)
    prediction_string = ''

    for t, p in zip(texts, predicted):
        prediction_string = '%s' % (target_names[p])

    return prediction_string































