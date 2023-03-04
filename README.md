# WP3.2-Physical-surrogate
Physical surrogate model

This python program builds a physical surrogate model using physical data to replicate a digital twin which is modelled on a Convolutional Neural Network (CNN). It also monitors and continuously updates the model against the surrogate.

The necessary libraries for this algorithm are numpy, tensorflow, and the tensorflow.keras layers library.


#Importing necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Loading the data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#Reshaping the data
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

#Normalizing the data
X_train = X_train / 255.0
X_test = X_test / 255.0

#Defining the model
model = tf.keras.models.Sequential([
  Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  MaxPool2D(2, 2),

  Flatten(),
  Dense(128, activation='relu'),
  Dense(10, activation='softmax')
])

#Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Fitting the model
model.fit(X_train, y_train, epochs=3)

#Evaluating the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test accuracy: {}'.format(test_accuracy))

#Building the physical surrogate model
#Using ImageDataGenerator to create augmented data
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

#Fitting the model
train_generator = train_datagen.flow(X_train, y_

please continue

train, batch_size=64)
model.fit_generator(train_generator, steps_per_epoch=X_train.shape[0] // batch_size, epochs=3)

#Evaluating the model on surrogate data
test_loss, test_accuracy = model.evaluate_generator(train_generator, steps=X_test.shape[0] // batch_size)
print('Test accuracy: {}'.format(test_accuracy))

#Monitoring and continuously updating the Digital Twin model against the surrogate
#Creating a callback to monitor the model
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

#Defining and fitting the model with the callback
callbacks = myCallback()
model.fit(X_train, y_train, epochs=3, callbacks=[callbacks])


#Evaluating the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test accuracy: {}'.format(test_accuracy))


This python program builds a physical surrogate model using physical data to replicate a digital twin which is modelled on a Convolutional Neural Network (CNN). It also monitors and continuously updates the model against the surrogate.


The necessary libraries for this algorithm are numpy, tensorflow, and the tensorflow.keras layers library.
