#Description: This program uses convolutional neural networks to classify handwritten digits 0-9

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

#Loading and importing dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

y_train[0:5]

#View image as picture
plt.imshow(X_train[0])

X_train.shape

#Reshape the data to fit the model
X_train = X_train.reshape( 60000, 28, 28, 1)
X_test = X_test.reshape( 10000, 28, 28, 1)

#One-Hot Encoding
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

print(y_train_one_hot[0])

#Creating the CNN
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size = 3, activation = 'relu', input_shape =(28,28, 1)))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size = 3, activation = 'relu'))
model.add(MaxPool2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(10, activation = 'softmax'))

#Compile the CNN
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ["accuracy"])

#Training the model
hist = model.fit(X_train, y_train_one_hot, validation_data = (X_test, y_test_one_hot), epochs = 15)

#Visualization
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show()

#Show predictions as probabilities for first 4 images in test set
pred = model.predict(X_test)

print(pred)

#Print our predictions as number labels for first 4 images
print(np.argmax(pred, axis = 1))
#print answer
print(y_test[0:4])

for i in range (0, 4):
  image = X_test[i]
  image = np.array(image, dtype = 'float')
  pixels = image.reshape((28, 28))
  plt.imshow(pixels, cmap= 'gray')
  plt.show()