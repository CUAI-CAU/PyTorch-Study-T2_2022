import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import time
from sklearn.svm import SVC

train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")

X = train_data.text
y = train_data.target

vectorizer = CountVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
vectorizer.inverse_transform(X[0])


model = SVC(kernel="rbf", gamma=0.1, C=0.3)
start_time = time.time()
model.fit(X,y)

end_time = time.time() - start_time

print("train time : ", end_time)

y_pred = model.predict(X[0])

print("예측 라벨 : ", y_pred)
print("실제 라벨 : ", train_data.target[0])

test_X = test_data.text
test_X_vect = vectorizer.transform(test_X)

pred = model.predict(test_X_vect)



