import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import cifar10
import okera

# pylint: disable=line-too-long
NIGHTLY_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzUxMiJ9.eyJzdWIiOiJjZXJlYnJvIiwiaXNzIjoiY2VyZWJyb2RhdGEuY29tIiwiZ3JvdXBzIjpbInJvb3QiLCJhZG1pbiJdLCJleHAiOjE1OTA1MTA4MDd9.diIN9SfRgaIiMEx9tEBM4vksuxCAE1l9fLH-J7qtJxUu0oLrAXFW6BiJrFt6-YdlDmAQhbB5Q7zGiBv8uKrq8tvaqpzV-16IRDgZ3SQVh4SklE5G5SX-cQ6b46kMIl4L9xommA7oHGtc-DIHMf77OXB2lAQp9XM2o3AXzJuZ_du2plm6Dzxz8_KXXgbhTyyQDHdsR4w0jH2u7ClMaPt6bSKlabweaGCC3Lz7y_56HQw0LF12C6m3vEW9vkV9iB7fxFmi9TjEVnnFVPkiCZa0OHUU-L2iKjipfRecz4O3X3IgF_tykVuFBtsWVz_0TyLPMSMGUMtB-yEXXhIQcTlftL4Q2fS7ToMPWZDZmF5OX9pwHdYvk-1A_BOglClad1RaD0HooNUf8Qr_kScxwxU4TcIIIjQffvAcX9jC3lB_x5tdosfPTQlkiRgqfgBWk73ryvcUImWpV00hdksMFBxW-o8-5leTseFYXDGK_aD_YMIminTUzt602evVSieYTRG1w5VwDJGv_iEcVjMB3zE7SVQYz9vcoMPjlNtmuxL_VCvvPPTmb8OjDha-NMiMGu6jQve4i-5aJNMdeZt-idvGPZjNb81yyZa9CKqt9s9R5YyfO8nlPyAa9c-eWhJp5UWVLlesN2IfWlMcnwCNN0dCfoMCmHfhGrWX5EU-yVFXmHA'
# pylint: enable=line-too-long

ctx = okera.context()
def connect(server, port=12050, token=None):
  if token:
    ctx.enable_token_auth(token_str=token)
  return ctx.connect(host=server, port=port)

def connect_nightly(token=NIGHTLY_TOKEN):
  return connect(server='ec2-34-215-143-132.us-west-2.compute.amazonaws.com', token=token)

NUM_TRAIN_SAMPLES = 5000
NUM_TEST_SAMPLES = 200

def output_images(x, filename, idx, delim="\n", prefix=""):
  with open(filename, "w") as f:
    f.write(prefix)
    f.write("P3")
    f.write(delim)
    f.write("32 32 256")
    f.write(delim)
    for row in range(32):
      for col in range(32):
        r = x[idx][row][col][0]
        g = x[idx][row][col][1]
        b = x[idx][row][col][2]
        f.write(str(r) + " " + str(g) + " " + str(b) + " ")
      f.write(delim)

def save_images():
  for i in range(NUM_TRAIN_SAMPLES):
    label = y_train[i][0]
    output_images(x_train, 'ppm/train/' + str(i) + ".ppm", i,
                  delim=" ", prefix=str(label) + "|")
  for i in range(NUM_TEST_SAMPLES):
    label = y_test[i][0]
    output_images(x_test, 'ppm/test/' + str(i) + ".ppm", i,
                  delim=" ", prefix=str(label) + "|")

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[0:NUM_TRAIN_SAMPLES, :, : :]
y_train = y_train[0:NUM_TRAIN_SAMPLES, :]
x_test = x_test[0:NUM_TEST_SAMPLES, :, : :]
y_test = y_test[0:NUM_TEST_SAMPLES, :]

dataset = 'demo_test.cifar_train'
with connect_nightly() as conn:
  train = conn.scan_as_python(
      'select label_idx, image_data(img) from ' + dataset, max_records=NUM_TRAIN_SAMPLES)
  #train = conn.scan_as_python(
  #    'select label_idx, image_data(blur(img, 25)) from ' + dataset, max_records=NUM_TRAIN_SAMPLES)
  labels = []
  image_data = []

  # Turn the string of pixels into int array
  for v in train:
    labels.append(v[0])
    for pixel in v[1].split(' '):
      if not pixel:
        continue
      image_data.append(int(pixel))
  x_train = np.array(image_data).reshape(len(train), 32, 32, 3)
  y_train = np.array(labels).reshape((len(labels), 1))

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
