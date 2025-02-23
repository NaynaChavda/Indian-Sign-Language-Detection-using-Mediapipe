import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.preprocessing.sequence import pad_sequences
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))

# print(type(data_dict['data']))

data_np = data_dict['data']

# Pad the sequences so that all have the same length
padded_data = pad_sequences(data_np, padding='post', dtype='float32')

# Now convert the padded list to a NumPy array
data = np.asarray(padded_data)


# data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')
f1 = f1_score(y_test, y_predict, average='weighted')

print('Accuracy Score : {}% '.format(score * 100))
print('Precision: {}%'.format(precision * 100)) 
print('Recall : {}% '.format(recall * 100))
print('F1-Score : {}% '.format(f1 * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()