from __future__ import print_function

import numpy as np
import tflearn
import okera

# pylint: disable=line-too-long
NIGHTLY_TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzUxMiJ9.eyJzdWIiOiJjZXJlYnJvIiwiaXNzIjoiY2VyZWJyb2RhdGEuY29tIiwiZ3JvdXBzIjpbInJvb3QiLCJhZG1pbiJdLCJleHAiOjE1OTA1MTA4MDd9.diIN9SfRgaIiMEx9tEBM4vksuxCAE1l9fLH-J7qtJxUu0oLrAXFW6BiJrFt6-YdlDmAQhbB5Q7zGiBv8uKrq8tvaqpzV-16IRDgZ3SQVh4SklE5G5SX-cQ6b46kMIl4L9xommA7oHGtc-DIHMf77OXB2lAQp9XM2o3AXzJuZ_du2plm6Dzxz8_KXXgbhTyyQDHdsR4w0jH2u7ClMaPt6bSKlabweaGCC3Lz7y_56HQw0LF12C6m3vEW9vkV9iB7fxFmi9TjEVnnFVPkiCZa0OHUU-L2iKjipfRecz4O3X3IgF_tykVuFBtsWVz_0TyLPMSMGUMtB-yEXXhIQcTlftL4Q2fS7ToMPWZDZmF5OX9pwHdYvk-1A_BOglClad1RaD0HooNUf8Qr_kScxwxU4TcIIIjQffvAcX9jC3lB_x5tdosfPTQlkiRgqfgBWk73ryvcUImWpV00hdksMFBxW-o8-5leTseFYXDGK_aD_YMIminTUzt602evVSieYTRG1w5VwDJGv_iEcVjMB3zE7SVQYz9vcoMPjlNtmuxL_VCvvPPTmb8OjDha-NMiMGu6jQve4i-5aJNMdeZt-idvGPZjNb81yyZa9CKqt9s9R5YyfO8nlPyAa9c-eWhJp5UWVLlesN2IfWlMcnwCNN0dCfoMCmHfhGrWX5EU-yVFXmHA'
# pylint: enable=line-too-long

def connect(server, port=12050, token=None):
  if token:
    ctx.enable_token_auth(token_str=token)
  return ctx.connect(host=server, port=port)

def connect_nightly(token=NIGHTLY_TOKEN):
  return connect(server='ec2-34-215-143-132.us-west-2.compute.amazonaws.com', token=token)

# Download the Titanic dataset
#from tflearn.datasets import titanic
#titanic.download_dataset('titanic_dataset.csv')

# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', target_column=0,
                        categorical_labels=True, n_classes=2)
#print(data)
print(labels)

# Preprocessing function
def preprocess(passengers, columns_to_delete):
    # Sort by descending id and delete columns
    for column_to_delete in sorted(columns_to_delete, reverse=True):
        [passenger.pop(column_to_delete) for passenger in passengers]
    for i in range(len(passengers)):
        # Converting 'sex' field to float (id is 1 after removing labels column)
        passengers[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(passengers, dtype=np.float32)

# Ignore 'name' and 'ticket' columns (id 1 & 6 of data array)
to_ignore=[1, 6]

# Preprocess data
data = preprocess(data, to_ignore)

# Build neural network
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# Define model
model = tflearn.DNN(net)
# Start training (apply gradient descent algorithm)
model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

# Let's create some data for DiCaprio and Winslet
dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]
winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]
# Preprocess data
dicaprio, winslet = preprocess([dicaprio, winslet], to_ignore)
# Predict surviving chances (class 1 results)
pred = model.predict([dicaprio, winslet])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet Surviving Rate:", pred[1][1])
