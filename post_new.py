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
from random import randint

ARCHIVE_NAME = "values.tar.gz"
CACHE_NAME = "values.pkz"
TEMP_FOLDER = "values_temp"
TRAIN_FOLDER = "values_train"
TEST_FOLDER = "values_test"

reload(sys)
sys.setdefaultencoding('UTF8')

def post_new_value(values=[""], category="None", project_name=""):
    for value in values:
        randomValue = randint(0,3)
        if randomValue == 0:
            TEMP_FOLDER = TEST_FOLDER
        else:
            TEMP_FOLDER = TRAIN_FOLDER
        filepath = get_data_home(project_name=project_name)
        if not os.path.isdir(filepath):
            os.makedirs(filepath)
        filepath = os.path.join(filepath, TEMP_FOLDER)
        if not os.path.isdir(filepath):
            os.makedirs(filepath)
        filepath = os.path.join(filepath, category)
        if not os.path.isdir(filepath):
            os.makedirs(filepath)
        i = 0
        while os.path.exists(filepath + "/custom%s" % i):
            i += 1
        with open(os.path.join(filepath, "custom%s" % i), "a") as f:
            f.write(value)

    return True

def get_data_home(data_home=None, project_name=""):
    if data_home is None:
        data_home = os.path.dirname(os.path.abspath(__file__))
    data_home = expanduser(data_home)
    data_home = os.path.join(data_home, project_name)
    if not exists(data_home):
        os.makedirs(data_home)
    return data_home
