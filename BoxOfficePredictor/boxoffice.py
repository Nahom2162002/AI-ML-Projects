import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn import metrics 
from xgboost import XGBRegressor

import warnings 
warnings.filterwarnings('ignore')

df = pd.read_csv('boxoffice.csv', encoding='latin-1')
df.head()

df.shape 
df.info()

df.describe().T 

to_remove = ['world_revenue', 'opening_revenue']
df.drop(to_remove, axis=1, inplace=True)

df.isnull().sum() * 100 / df.shape[0]

df.drop('budget', axis=1, inplace=True)

for col in ['MPAA', 'genres']:
    df[col] = df[col].fillna(df[col].mode()[0])

df.dropna(inplace=True)
df.isnull().sum().sum()

df['domestic_revenue'] = df['domestic_revenue'].str[1:]

for col in ['domestic_revenue', 'opening_theaters', 'release_days']:
    df[col] = df[col].str.replace(',', '')

    temp = (~df[col].isnull())
    df[temp][col] = df[temp][col].convert_dtypes(float)

    df[col] = pd.to_numeric(df[col], errors='coerce')

plt.figure(figsize=(10, 5))
sb.countplot(df['MPAA'])
plt.show()

df.groupby('MPAA').mean()['domestic_revenue']

plt.subplots(figsize=(15, 5))

features = ['domestic_revenue', 'opening_theaters', 'release_days']
for i, col in enumerate(features):
    plt.subplot(1, 3, i+1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()

plt.subplots(figsize=(15, 5))
for i, col in enumerate(features):
    plt.subplot(1, 3, i+1)
    sb.boxplot(df[col])
plt.tight_layout()
plt.show()

for col in features:
    df[col] = df[col].apply(lambda x: np.log10(x))

plt.subplots(figsize=(15, 5))
for i, col in enumerate(features):
    plt.subplot(1, 3, i+1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()

vectorizer = CountVectorizer()
vectorizer.fit(df['genres'])
features = vectorizer.transform(df['genres']).toarray()

genres = vectorizer.get_feature_names()
for i, name in enumerate(genres):
    df[name] = features[:, i]

df.drop('genres', axis=1, inplace=True)

removed = 0
for col in df.loc[:, 'action':'western'].columns:
    if (df[col] == 0).mean() > 0.95:
        removed += 1
        df.drop(col, axis=1, inplace=True)

print(removed)
print(df.shape)

for col in ['distributor', 'MPAA']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

plt.figure(figsize=(8, 8))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()

features = df.drop(['title', 'domestic_revenue', 'fi'], axis=1)
target = df['domestic_revenue'].values 

X_train, X_val, \
    Y_train, Y_val = train_test_split(features, target, test_size=0.1, random_state=22)
X_train.shape, X_val.shape 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

from sklearn.metrics import mean_absolute_error as mae 
model = XGBRegressor()
model.fit(X_train, Y_train)

train_preds = model[i].predict(X_train)
print('Training Error : ', mae(Y_train, train_preds))

val_preds = model[i].predict(X_val)
print('Validation Error : ', mae(Y_val, val_preds))
print()