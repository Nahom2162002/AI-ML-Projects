import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sb 

from imblearn.over_sampling import RandomOverSampler 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, MinMaxScaler 
from sklearn.feature_selection import SelectKBest, chi2 
from tqdm.notebook import tqdm 
from sklearn import metrics 
from sklearn.svm import SVC 
from xgboost import XGBClassifier 
from sklearn.linear_model import LogisticRegression 

import warnings 
warnings.filterwarnings('ignore')