import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import cifar10

NUM_TRAIN_SAMPLES = 1000
NUM_TEST_SAMPLES = 200

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[0:NUM_TRAIN_SAMPLES, :, : :]
y_train = y_train[0:NUM_TRAIN_SAMPLES, :]
x_test = x_test[0:NUM_TEST_SAMPLES, :, : :]
y_test = y_test[0:NUM_TEST_SAMPLES, :]

print('x_train shape:', x_train.shape)
print('y_train shape', y_train.shape)
print('x_test shape', x_test.shape)
print('y_test shape', y_test.shape)

# Declare variables
batch_size = 32
# 32 examples in a mini-batch, smaller batch size means more updates in one epoch
epochs = 1
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])

# Train the model
model.fit(x_train / 255.0, tf.keras.utils.to_categorical(y_train),
          batch_size=batch_size,
          shuffle=True,
          epochs=epochs,
          validation_data=(x_test / 255.0, tf.keras.utils.to_categorical(y_test))
          )

# Evaluate the model
scores = model.evaluate(x_test / 255.0, tf.keras.utils.to_categorical(y_test))

print('Loss: %.3f' % scores[0])
print('Accuracy: %.3f' % scores[1])
