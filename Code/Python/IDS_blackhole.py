import pandas as pd
import numpy as np

# import keras
# from keras.models import Sequential
# from keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# import tensorflow as tf
# from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# dd = pd.read_csv('data.csv', usecols=(1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18))
# print(dd.shape)
print("Reading Dataset...")
df = pd.read_csv('black-hole.csv')
data = df.values
data = data[:,4:]

train_data = data[:,:-1]
labels = data[:,-1]

unique_labels = np.array(labels)
unique_labels = np.unique(unique_labels)

encoder = LabelBinarizer()
transfomed_label = encoder.fit_transform(labels)
print("\nDifferent Attack Types: ", unique_labels)
print("\nRed-hot encoding assigned to Atacks: ", unique_labels, "\n", encoder.fit_transform(unique_labels))
# c=0
# new = []
# for i in labels:
#     try:
#         new.append(int(i))
#     except:
#         new.append(0)
#         c+=1

# Y = np.array (new)
Y = transfomed_label
print("\nShape of training data (Rows, Column-features): ", train_data.shape)
print("Shape of labels (Rows, Column-Attack): ", Y.shape)

print("\nSplitting Data into training and testing data (75-25%): ")
X_train,X_test,Y_train,Y_test = train_test_split(train_data,Y)

print("Traing Data size: ", X_train.shape)

print("Testing Data size: ", X_test.shape)

Y_train = to_categorical(Y_train,num_classes=2)
Y_test = to_categorical(Y_test,num_classes=2)

# print("Y_train Shape: ", Y_train.shape)
# print("Y_test Shape: ", Y_test.shape)

model = Sequential()

model.add(Dense(10,activation='relu',input_shape=(14,)))
model.add(Dense(10,activation='relu'))

model.add(Dense(2,activation='softmax'))

model.compile(loss='categorical_crossentropy',metrics=['acc'],optimizer='adam')

model.summary()

model.fit(X_train,Y_train,epochs=10)

model.evaluate(X_test,Y_test)

