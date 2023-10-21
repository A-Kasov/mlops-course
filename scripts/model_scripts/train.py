from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(os.path.dirname(BASE_DIR), 'data/baselines/train.csv')
train = pd.read_csv(path)


features = ["Pclass", "Sex", "SibSp", "Parch"]

x_train_full = train[features]
y_train_full = train["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full, train_size = 0.8, random_state = 42)


model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

print(accuracy_score(y_test, predictions))
pickle.dump(model, open(os.path.join(os.path.dirname(BASE_DIR), 'data/models/model.pkl'), "wb"))