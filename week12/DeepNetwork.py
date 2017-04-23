from __future__ import print_function
import tensorflow as tf
import numpy as np
from cifar import load_cifar_file

def get_data(datafile='../input/data_batch_1'):
  images, labels = load_cifar_file(datafile)
  images = 2*images.astype(np.float)/255. - 1
  N = labels.shape[0]
  #One-hot
  labels_tmp = np.zeros((N, 10))
  labels_tmp[np.arange(N), labels] = 1
  labels = labels_tmp
  return images, labels

def conv_layer(x, filters=100, k_size=3, name='conv'):
  with tf.name_scope(name):
    filters_in = x.get_shape().as_list()[-1]
    kernel = tf.Variable(tf.random_normal([k_size, k_size, filters_in, filters],
                                          stddev=np.sqrt(2) / (filters_in*k_size**2)),
                         name='W')
    bias = tf.Variable(tf.zeros([filters]), name='bias')
    out = tf.nn.relu(tf.nn.conv2d(x, kernel, strides=(1, 2, 2, 1), padding='SAME') + bias)
    tf.summary.histogram('activations', out)
    return out

def fc(x, n_classes=10):
  with tf.name_scope('fully_connected_layer'):
    filters_in = np.product(x.get_shape().as_list()[1:])
    kernel = tf.Variable(tf.random_normal([filters_in, n_classes], stddev=1/filters_in), name='W')
    bias = tf.Variable(tf.zeros([10]), name='b')
    out = tf.matmul(tf.reshape(x, (-1, filters_in)), kernel) + bias
    tf.summary.histogram('activations', out)
    return out

images, labels = get_data()
images_test, labels_test = get_data('../input/data_batch_2')
N = labels.shape[0]

img = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int32, [None, 10])
tf.summary.image('img', img)

#Layer 1
h1 = conv_layer(tf.random_crop(img, [64, 28, 28, 3]), k_size=3)
h2 = conv_layer(h1, k_size=3)
h3 = conv_layer(h2, k_size=3)
h4 = conv_layer(h3, k_size=3)
h5 = conv_layer(h4, k_size=3)

#Layer 2 fully-connected
out = fc(h5)

#Loss and initialize
loss = tf.contrib.losses.softmax_cross_entropy(out, y)
tf.summary.scalar('loss', loss)
grads = tf.gradients(loss, tf.trainable_variables())
for g, v in zip(grads, tf.trainable_variables()):
  tf.summary.histogram('gradient/'+v.name, g)

train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

merged_summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('train/', sess.graph)
#RUN
for i in range(100000):
  batch_ind = np.random.choice(N, 64)
  img_batch, label_batch = images[batch_ind], labels[batch_ind]
  loss_val, pred, _ = sess.run([loss, out, train_op], {img: img_batch, y: label_batch})
  if i%100==0:
    train_summary = sess.run(merged_summaries, {img: img_batch, y: label_batch})
    train_writer.add_summary(train_summary, i)
    print('Loss:', loss_val, (pred.argmax(1) == label_batch.argmax(1)).mean())
    batch_ind = np.random.choice(labels_test.shape[0], 64)
    img_batch, label_batch = images_test[batch_ind], labels_test[batch_ind]
    loss_val, pred = sess.run([loss, out], {img: img_batch, y: label_batch})
    print('\n\tTEST Loss:', loss_val, (pred.argmax(1) == label_batch.argmax(1)).mean())