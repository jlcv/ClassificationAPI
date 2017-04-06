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

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
              " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove information that is easily overfit: "
              "headers, signatures, quotes")

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

# work-around for Jupyter notebook and IPython console
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

logger = logging.getLogger(__name__)

ARCHIVE_NAME = "email_tickets.tar.gz"
CACHE_NAME = "email_tickets.pkz"
TRAIN_FOLDER = "email_tickets_train"
TEST_FOLDER = "email_tickets_test"

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

def load_files(container_path, description=None, categories=None,
               load_content=True, shuffle=True, encoding=None,
               decode_error='strict', random_state=0):
    """Load text files with categories as subfolder names.

    Individual samples are assumed to be files stored a two levels folder
    structure such as the following:

        container_folder/
            category_1_folder/
                file_1.txt
                file_2.txt
                ...
                file_42.txt
            category_2_folder/
                file_43.txt
                file_44.txt
                ...

    The folder names are used as supervised signal label names. The
    individual file names are not important.

    This function does not try to extract features into a numpy array or
    scipy sparse matrix. In addition, if load_content is false it
    does not try to load the files in memory.

    To use text files in a scikit-learn classification or clustering
    algorithm, you will need to use the `sklearn.feature_extraction.text`
    module to build a feature extraction transformer that suits your
    problem.

    If you set load_content=True, you should also specify the encoding of
    the text using the 'encoding' parameter. For many modern text files,
    'utf-8' will be the correct encoding. If you leave encoding equal to None,
    then the content will be made of bytes instead of Unicode, and you will
    not be able to use most functions in `sklearn.feature_extraction.text`.

    Similar feature extractors should be built for other kind of unstructured
    data input such as images, audio, video, ...

    Read more in the :ref:`User Guide <datasets>`.

    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category

    description : string or unicode, optional (default=None)
        A paragraph describing the characteristic of the dataset: its source,
        reference, etc.

    categories : A collection of strings or None, optional (default=None)
        If None (default), load all the categories.
        If not None, list of category names to load (other categories ignored).

    load_content : boolean, optional (default=True)
        Whether to load or not the content of the different files. If
        true a 'data' attribute containing the text information is present
        in the data structure returned. If not, a filenames attribute
        gives the path to the files.

    encoding : string or None (default is None)
        If None, do not try to decode the content of the files (e.g. for
        images or other non-text content).
        If not None, encoding to use to decode text files to Unicode if
        load_content is True.

    decode_error : {'strict', 'ignore', 'replace'}, optional
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. Passed as keyword
        argument 'errors' to bytes.decode.

    shuffle : bool, optional (default=True)
        Whether or not to shuffle the data: might be important for models that
        make the assumption that the samples are independent and identically
        distributed (i.i.d.), such as stochastic gradient descent.

    random_state : int, RandomState instance or None, optional (default=0)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are: either
        data, the raw text data to learn, or 'filenames', the files
        holding it, 'target', the classification labels (integer index),
        'target_names', the meaning of the labels, and 'DESCR', the full
        description of the dataset.
    """
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

    # convert to array for fancy indexing
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
        data_home = environ.get('SCIKIT_LEARN_DATA', join('~', 'scikit_learn_data'))
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
    cache = None
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
            logger.info("Downloading email tickets dataset. "
                        "This may take a few minutes.")
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
        vectorizer = CountVectorizer(dtype=np.int16, lowercase=True, stop_words=spanish_stopwords, strip_accents=unicode)#(dtype=np.int16, stop_words=spanish_stopwords, lowercase=True, strip_accents=unicode, tokenizer=tokenize,)
        X_train = vectorizer.fit_transform(data_train.data).tocsr()
        X_test = vectorizer.transform(data_test.data).tocsr()
        joblib.dump((X_train, X_test), target_file, compress=9)

    # the data is stored as int16 for compactness
    # but normalize needs floats
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    normalize(X_train, copy=False)
    normalize(X_test, copy=False)

    target_names = data_train.target_names

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


###############################################################################
# Load some categories from the training set
if opts.all_categories:
    categories = None
else:
    categories = [
                   'ZInstalacionHardware',
                   'ZInstalacionSoftware',
                   'ZRespaldo',
                   'ZConfiguracionOutlook',
                   'ZAsignacionComputadora' ]

# if opts.filtered:
#     remove = ('headers', 'footers', 'quotes')
# else:
#     remove = ()

print("Loading emails dataset for categories:")
print(categories if categories else "all")

data_train = fetch_email_tickets(subset='train', categories=categories,
                                shuffle=True, random_state=42)

data_test = fetch_email_tickets(subset='test', categories=categories,
                               shuffle=True, random_state=42)
print('data loaded')

# order of labels in `target_names` can be different from `categories`
target_names = data_train.target_names


def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

print("%d documents - %0.3fMB (training set)" % (
                                                 len(data_train.data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
                                             len(data_test.data), data_test_size_mb))
print("%d categories" % len(categories))
print()

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training data using a sparse vectorizer")
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

if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words=stopwords_list, non_negative=True,
                                   n_features=opts.n_features, strip_accents=unicode)
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words=stopwords_list, use_idf=False, strip_accents=unicode)
    X_train = vectorizer.fit_transform(data_train.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test data using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" % opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                        in ch2.get_support(indices=True)]
        print("done in %fs" % (time() - t0))
print()

if feature_names:
    feature_names = np.asarray(feature_names)


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


###############################################################################
# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    
    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)
    
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    
    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        
        if opts.print_top10 and feature_names is not None:
            print("top 10 keywords per class:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-20:]
                print("%s: %s" % (label, " ".join(feature_names[top10])))
        print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))
    
    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


results = []

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time",
         color='c')
plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()































