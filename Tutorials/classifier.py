from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score 

data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
print("\n", label_names)

print("\n", labels[0])

print("\n", feature_names[0])
print("\n", features[0])

train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.40, random_state = 42)
gnb = GaussianNB()
model = gnb.fit(train, train_labels)

preds = gnb.predict(test)
print(preds)

print(accuracy_score(test_labels, preds))