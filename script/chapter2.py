#2.1 初めてのニューラルネットワーク
import sys
print(sys.version)
import keras
print(keras.__version__)
import tensorflow
print(tensorflow.__version__)
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images.shape
len(train_labels)

train_labels

test_images.shape
len(test_labels)
test_labels

from keras import models
from keras import layers

network = models.Sequential()

network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28, )))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape(10000, 28 * 28)
test_images = test_images.astype('float32') / 255

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

#2.2.1 スカラー : 0次元テンソル
import numpy as np
x = np.array(12)
x
x.ndim

#2.2.2 ベクトル : 1次元テンソル
x = np.array([12, 3, 6, 14, 7])
x
x.ndim

#2.2.3 行列 : 2次元テンソル
x = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]])

x.ndim

#2.2.4 3次元テンソルとより高次元のテンソル
x = np.array([[[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]],
             [[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]],
             [[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]]])

x.ndim

#2.2.5 テンソルの重要な属性
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = \
        mnist.load_data()

print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)

#2.2.6 Numpy でのテンソルの操作
my_slice = train_images[10:100]
my_slice.shape

my_slice = train_images[10:100, :, :]
my_slice.shape

my_slice = train_images[10:100, 0:28, 0:28]
my_slice.shape

my_slice = train_images[:, 14, 14]

my_slice = train_images[:, 7:-7, 7:-7]

batch = train_images[:128]
batch = train_images[128:256]

n = 2
batch = train_images[128 * n:128 * (n + 1)]

#2.3 ニューラルネットワークの歯車 : テンソル演算
keras.layers.Dense(512, activation='relu')


#2.3.1 要素ごとの演算
def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x

def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x

import numpy as np
z = x + y
x = np.maximum(z, 0.)

#2.3.2 ブロードキャスト
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x[0]):
        for j in range(x[1]):
            x[i, j] += y[j]
    return x

import numpy as np
x = np.random.random((64, 3, 32, 10))
y = np.random.random((32,  10))
z = np.maximum(x, y)
print(x.shape)

def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z

import numpy as np

def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert lent(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z

def naive_matrix_vector_dot(x, y):
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y)
    return z

def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2

    assert x.shape[1] == y.shape[0]

    z = np.zeros((x.shape[0], y.shape[1]))

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            row_x = x[i, :]
            column_y = y[:, j]
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z

#2.3.4 テンソルの変形

x = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])
print(x.shape)

x = x.reshape((6, 1))
x
x.reshape((2, 3))
x = np.zeros((300, 20))
x = np.transpose(x)
print(x.shape)
