from __future__ import print_function
import tensorflow as tf
import numpy as np
from cifar import load_cifar_file

images, labels = load_cifar_file('./input/data_batch_1')
images = 2*images.astype(np.float)/255. - 1
N = labels.shape[0]

labels_tmp = np.zeros((N, 10))
labels_tmp[np.arange(N), labels] = 1
labels = labels_tmp

img = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int32, [None, 10])

kernel1 = tf.Variable(tf.fill([7, 7, 3, 100], 2/(49.*3)))
bias1 = tf.Variable(tf.zeros([100]))
filtered1 = tf.nn.relu(tf.nn.conv2d(img, kernel1, strides=(1,2,2,1), padding='SAME') + bias1)

kernel2 = tf.Variable(tf.fill([100*16*16, 10], 1./(100*16*16)))
bias2 = tf.Variable(tf.zeros([10]))
filtered2 = tf.matmul(tf.reshape(filtered1, (64, -1)), kernel2)+bias2# tf.nn.conv2d(filtered1, kernel2, strides=(1,1,1,1), padding='SAME') + bias2
filtered_mean = filtered2#tf.reduce_mean(filtered2, [1, 2])

loss = tf.contrib.losses.softmax_cross_entropy(filtered_mean, y)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100000):
  batch_ind = np.random.choice(N, 64)
  img_batch, label_batch = images[batch_ind], labels[batch_ind]
  loss_val, pred, _ = sess.run([loss, filtered_mean, train_op], {img: img_batch, y: label_batch})
  print('Loss:', loss_val, (pred.argmax(1) == label_batch.argmax(1)).mean())