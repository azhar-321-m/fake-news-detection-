import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

fake_path = os.path.join(BASE_DIR, "Fake.csv")
true_path = os.path.join(BASE_DIR, "True.csv")

data_fake = pd.read_csv(fake_path)
data_true = pd.read_csv(true_path)

print(data_fake.head())
print(data_true.head())

print("Fake Dataset Shape:", data_fake.shape)
print("True Dataset Shape:", data_true.shape)

print("\nFake Dataset Columns:")
print(data_fake.columns)

print("\nTrue Dataset Columns:")
print(data_true.columns)

print("\nMissing Values in Fake:")
print(data_fake.isnull().sum())

print("\nMissing Values in True:")
print(data_true.isnull().sum())

data_fake["class"] = 0   # Fake News
data_true["class"] = 1   # True News

data_merge = pd.concat([data_fake, data_true], axis=0)

from sklearn.utils import shuffle

data = shuffle(data_merge, random_state=42)
data.reset_index(drop=True, inplace=True)

print(data.head())

import re
import string

def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

data["content"] = data["title"] + " " + data["text"]
data["content"] = data["content"].apply(wordopt)

x = data["content"]
y = data["class"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()

xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
print(xv_train.shape)
print(xv_test.shape)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)


from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, pred_lr))

from sklearn.metrics import classification_report
print(classification_report(y_test, pred_lr))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred_lr)
print(cm)

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

import joblib
joblib.dump(LR, "model.pkl")
joblib.dump(vectorization, "vectorizer.pkl")
print("Model Saved Successfully!") 

###...python -m streamlit run app.py.......####