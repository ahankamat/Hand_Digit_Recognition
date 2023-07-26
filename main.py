
import seaborn as sn
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import keras
import requests
import matplotlib
import seaborn as sb
# Replace 'TkAgg' with an appropriate backend for your system if needed.
matplotlib.use('TkAgg')
# Function to download the MNIST dataset


def download_mnist_dataset():
    base_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    filename = 'mnist.npz'
    download_url = os.path.join(base_url, filename)

    with open(filename, "wb") as f:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return filename


# Check if the MNIST dataset exists locally; if not, download it
if not os.path.exists('mnist.npz'):
    download_mnist_dataset()

# Load the dataset
with np.load('mnist.npz', allow_pickle=True) as data:
    X_train, Y_train = data['x_train'], data['y_train']
    X_test, Y_test = data['x_test'], data['y_test']

# Continue with the rest of your code...
# print(len(X_train)) -- 60000
# print(len(X_test))  -- 10000
# print(X_train[0].shape) -- (28,28)
# print(X_train[0]) --- array containing pixel values between 0 and 255

# plt.matshow(X_train[0])
# plt.show() -- shows the digits from the dataset plotted 


# flatten the array into column vector

# print(X_train.shape) ---   (60000,28,28)


# Note: When you dont scale it , accuracy here is approx 90%
# On scaling , accuracy rises.
# Scaling means each value will be between 0 and 255 i.e. each pixel / 255
X_train = X_train / 255
X_test = X_test / 255


X_train_flatten = X_train.reshape(len(X_train),-1)
# -1 means that 28*28 = 784
print(X_train_flatten.shape)

X_test_flatten = X_test.reshape(len(X_test),-1)
print(X_test_flatten.shape)

# make the model
# sequential is type of network and it has layers.Dense which 
# takes parameters(output , input_shape=() , activation='')
# model = keras.Sequential([
#     keras.layers.Dense(10,input_shape=(784,), activation='softmax')
# ])

# # compile the model
# model.compile(
#     optimizer='Adam',
#     loss = 'sparse_categorical_crossentropy',
#     metrics=['accuracy']
#     )

# model.fit(
#     X_train_flatten , Y_train , epochs=5
# )

# # to check accuracy on test dataset
# model.evaluate(X_test_flatten , Y_test)

# plt.matshow(X_test[0])
# plt.savefig('output_digit.png')
# plt.close()
# Y_predicted = model.predict(X_test_flatten)
# print(Y_predicted[0])
# print(np.argmax(Y_predicted[0]))

# Y_predicted_labels = [np.argmax(i) for i in Y_predicted]
# print(Y_predicted_labels[:5])
# print(Y_test[:5])

# cm = tf.math.confusion_matrix(
#     labels=Y_test,
#     predictions = Y_predicted_labels
# )
# plt.figure(figsize=(10, 7))
# sn.heatmap(cm, annot=True, fmt='d')
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
# plt.savefig('output_graph.png')
# plt.close()


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

# compile the model
model.compile(
    optimizer='Adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train_flatten, Y_train, epochs=5
)
