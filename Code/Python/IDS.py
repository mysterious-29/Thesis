import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

print("Reading Dataset...")
df = pd.read_csv('WSN-DS_original.csv')
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

Y = transfomed_label
print("\nShape of training data (Rows, Column-features): ", train_data.shape)
print("Shape of labels (Rows, Column-Attack): ", Y.shape)
print("\nSplitting Data into training and testing data (75-25%): ")
X_train,X_test,Y_train,Y_test = train_test_split(train_data,Y)

print("Traing Data size: ", X_train.shape)
print("Testing Data size: ", X_test.shape)


#model creation
model = Sequential()
model.add(Dense(14,activation='relu',input_shape=(14,)))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))

model.add(Dense(5,activation='softmax'))

model.compile(loss='categorical_crossentropy',metrics=['acc'],optimizer='adam')

model.summary()

history = model.fit(X_train,Y_train,epochs=10, validation_split = 0.20)

model.evaluate(X_test,Y_test)


y_pred_keras = model.predict(X_test).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test, y_pred_keras)
auc_keras = auc(fpr_keras, tpr_keras)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
# plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
# plt.legend(loc='best')
plt.show()

# predictions = model.predict(X_test)

# predictions

# print("Predictions", np.argmax(predictions[0]))
# print("Actual", Y_test[0])

# plt.figure()
# plt.imshow()
# plt.colorbar()
# plt.grid(Fasle)
# plt.show()
