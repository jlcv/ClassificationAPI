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

from flask import Flask, jsonify

ARCHIVE_NAME = "email_tickets.tar.gz"
CACHE_NAME = "email_tickets.pkz"
TEMP_FOLDER = "email_tickets_temp"
TRAIN_FOLDER = "email_tickets_train"
TEST_FOLDER = "email_tickets_test"
HOME_FOLDER = "email_classification_home"

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol', 
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web', 
        'done': False
    }
]

# Set encoding
reload(sys)
sys.setdefaultencoding('UTF8')

def get_dummy():
    return tasks

def post_email(email_body=""):
    filepath = os.path.join(get_data_home(), HOME_FOLDER)
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    filepath = os.path.join(filepath, TEMP_FOLDER)
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    i = 0
    while os.path.exists(filepath + "/temporal%s" % i):
        i += 1
    with open(os.path.join(filepath, "temporal%s" % i), "a") as f:
        f.write(email_body)
    return "true"

def get_data_home(data_home=None):
    if data_home is None:
        data_home = os.path.dirname(os.path.abspath(__file__))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home
