from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import SGDClassifier
from nltk.stem import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, PrecisionRecallDisplay
from matplotlib import pyplot as plt
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed, dump, load
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
