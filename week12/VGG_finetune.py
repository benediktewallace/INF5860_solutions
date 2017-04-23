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

def conv_layer(x, filters=100, k_size=3, stride=1, padding='SAME', relu=True, name='conv'):
  with tf.name_scope(name):
    filters_in = x.get_shape().as_list()[-1]
    kernel = tf.Variable(tf.random_normal([k_size, k_size, filters_in, filters],
                                          stddev=np.sqrt(2) / (filters_in*k_size**2)),
                         name='weights')
    bias = tf.Variable(tf.zeros([filters]), name='biases')
    out = tf.nn.conv2d(x, kernel, strides=(1, stride, stride, 1), padding=padding) + bias
    if relu:
      out = tf.nn.relu(out)
    tf.summary.histogram('activations', out)
    return out

def vgg_net(x, num_classes=10):
  with tf.name_scope('vgg_16'):
    name = 'conv1'
    with tf.name_scope(name):
      x = conv_layer(x, 64, name=name+'_1')
      x = conv_layer(x, 64, name=name+'_2')
      x = tf.nn.max_pool(x, (1, 2, 2, 1), (1, 2, 2, 1), padding='VALID')
    name = 'conv2'
    with tf.name_scope(name):
      x = conv_layer(x, 128, name=name+'_1')
      x = conv_layer(x, 128, name=name+'_2')
      x = tf.nn.max_pool(x, (1, 2, 2, 1), (1, 2, 2, 1), padding='VALID')
    name = 'conv3'
    with tf.name_scope(name):
      x = conv_layer(x, 256, name=name+'_1')
      x = conv_layer(x, 256, name=name+'_2')
      x = conv_layer(x, 256, name=name+'_3')
      x = tf.nn.max_pool(x, (1, 2, 2, 1), (1, 2, 2, 1), padding='VALID')
    name = 'conv4'
    with tf.name_scope(name):
      x = conv_layer(x, 512, name=name+'_1')
      x = conv_layer(x, 512, name=name+'_2')
      x = conv_layer(x, 512, name=name+'_3')
      x = tf.nn.max_pool(x, (1, 2, 2, 1), (1, 2, 2, 1), padding='VALID')
    name = 'conv5'
    with tf.name_scope(name):
      x = conv_layer(x, 512, name=name+'_1')
      x = conv_layer(x, 512, name=name+'_2')
      x = conv_layer(x, 512, name=name+'_3')
      x = tf.nn.max_pool(x, (1, 2, 2, 1), (1, 2, 2, 1), padding='VALID')

    with tf.name_scope('fc6') as name:
      x = conv_layer(x, 4096, k_size=7, padding='VALID', name=name)

    with tf.name_scope('fc7') as name:
      x = conv_layer(x, 4096, k_size=1, name=name)

    with tf.name_scope('fc8') as name:
      x = conv_layer(x, num_classes, k_size=1, relu=False, name=name)
    return tf.squeeze(x, squeeze_dims=(1, 2))


images, labels = get_data()
images_test, labels_test = get_data('../input/data_batch_2')
N = labels.shape[0]

img = tf.placeholder(tf.float32, [None, 32, 32, 3])
x = tf.image.resize_bilinear(img, [224, 224])
y = tf.placeholder(tf.int32, [None, 10])

logits = vgg_net(x)


#Loss and initialize
loss = tf.contrib.losses.softmax_cross_entropy(logits, y)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss, var_list=tf.trainable_variables()[-2:])
sess = tf.Session()
load_vars = tf.trainable_variables()[:-2]
sess.run(tf.variables_initializer(filter(lambda x: x not in load_vars, tf.global_variables())))
saver = tf.train.Saver(load_vars)
saver.restore(sess, 'vgg_16.ckpt')

#RUN
batch_size = 16
for i in range(100000):
  batch_ind = np.random.choice(N, batch_size)
  img_batch, label_batch = images[batch_ind], labels[batch_ind]
  loss_val, pred, _ = sess.run([loss, logits, train_op], {img: img_batch*128, y: label_batch})
  print('Loss:', loss_val, (pred.argmax(1) == label_batch.argmax(1)).mean())
  if i%100==0:
    batch_ind = np.random.choice(labels_test.shape[0], batch_size)
    img_batch, label_batch = images_test[batch_ind], labels_test[batch_ind]
    loss_val, pred = sess.run([loss, logits], {img: img_batch*128, y: label_batch})
    print('\n\tTEST Loss:', loss_val, (pred.argmax(1) == label_batch.argmax(1)).mean())
