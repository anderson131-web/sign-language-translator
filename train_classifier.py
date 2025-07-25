import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./sign lanuage update/sign-language-detector-python/data.pickle', 'rb'))


data = []
labels = []
for i, d in enumerate(data_dict['data']):
    if d:  # Only include non-empty data points
        data.append(d)
        labels.append(data_dict['labels'][i])


max_len = max(len(d) for d in data)
data_padded = []
for d in data:
    if len(d) < max_len:
       
        d = d + [0] * (max_len - len(d))
    data_padded.append(d)


data = np.array(data_padded)
labels = np.array(labels)


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)


model = RandomForestClassifier()
model.fit(x_train, y_train)


y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))


f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()