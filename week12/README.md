# INF5860 week 12

## Train a deeper network
Start with the code we wrote last week in *CNN.py*.

#### Write the code for a convolutional layer into a function, so it is easy to to reuse.

```python
def conv_layer(x, filters=100, k_size=3)
    #Your code
```
you may want to do the same with the fully connected layer.

#### Test architectures and see what works best (OBS don't spend to much time)


## Tensorboard

To make a simple summary of the *loss*, you can you

```python
tf.summary.scalar('loss', loss)
merged_summaries = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('train/', sess.graph)
```

Then in your traning loop, you have to save the summary with:

    train_summary = sess.run(merged_summaries, {img: img_batch, y: label_batch})
    train_writer.add_summary(train_summary, i)

Perhaps only every 100th iteration like this:
```python
for i in range(100000):
  batch_ind = np.random.choice(N, 64)
  img_batch, label_batch = images[batch_ind], labels[batch_ind]
  loss_val, pred, _ = sess.run([loss, out, train_op], {img: img_batch, y: label_batch})
  if i%100==0:
    train_summary = sess.run(merged_summaries, {img: img_batch, y: label_batch})
    train_writer.add_summary(train_summary, i)
```

From the **week12** folder you can then run a **tensorboard server** with:

    $ tensorboard --logdir train/
    
If you open the url **localhost:6006** in you browser you can see the tensorboard, and it may look similar to this:

![alt tag](https://www.google.no/url?sa=i&rct=j&q=&esrc=s&source=images&cd=&cad=rja&uact=8&ved=0ahUKEwiw0NHhvLvTAhUhCpoKHZpzD9YQjRwIBw&url=https%3A%2F%2Fwookayin.github.io%2Ftensorflow-talk-debugging%2F&psig=AFQjCNGzrOUFbnPEITT83Ly9q-e8oNj5ww&ust=1493067701105496)

#### Make an activation summary in you *conv_layer* function
In you *conv_layer* function you can visualize your activations with **tf.summary.histogram**

    tf.summary.histogram('conv', output)

You can also try to make summaries of your **weights**, **biases**, and your **gradients**.
 To get hold of your gradients you can use:
 
 ```python
variables = tf.trainable_variables() #to get hold of all your variables
grads = tf.gradients(loss, variables) #grads is now a list of your gradients for the loss w.r.t. the variables
for g, v in zip(grads, variables): tf.summary.histogram('gradients/'+v.name, g)
```

#### Check out the graph in tensorboard
Looks ugly right? To make it look decent you have to wrap the right parts of the graph with **tf.name_scope**.
As a first step you can try to wrap you **conv_layer** function like this:

```python
def conv_layer(x, filters=100, k_size=3, name='conv'):
    with tf.name_scope(name):
        #Your code
```

## Finetune on VGG features
In this exercise we only have a small part of the cifar10 dataset. We could therefore expect that we can get better
results by using a network that is pretrained. You can download the pretrained weights [vgg_16.ckpt](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz).

A .ckpt file only contains the weights, so you have to define the network yourself. So your first step is to define the
VGG network. Here is an image of the vgg16 network to help you along. You can use **tf.nn.max_pool** as maxpool function.
The maxpool functions should have a kernel size of **2x2** and a stride of **2**.
![alt tag](https://blog.keras.io/img/imgclf/vgg16_original.png)

**OBS!** In this implementation they don't use **fully connected layers**, they do a convolution with a 7x7x512x4096 kernel with padding='VALID'.
Then convolution with 1x1x4096x4096 and 1x1x4096xNUMBER_OF_CLASSES.

You don't have to use dropout, since we are not going to train the weights.
If it is hard to find the right architechture you can of course look at the solution.

**Also** the vgg weights are trained on input in the range of -128 to 128, so it may be a good idea to multiply the 
cifar10 images with 128

#### First test you network without loading weights

## Load weights

To load the weights you first have to download and extract the weights.
 
The main problem with loading the weights is that you need to use the **exact** same names and sizes. Here you can see the
names and sizes in the **vgg_16.ckpt** file:

- vgg_16/conv1/conv1_1/weights **shape**: [3, 3, 3, 64]
- vgg_16/conv1/conv1_1/biases **shape**: [64]
- vgg_16/conv1/conv1_2/weights **shape**: [3, 3, 64, 64]
- vgg_16/conv1/conv1_2/biases **shape**: [64]
- vgg_16/conv2/conv2_1/weights **shape**: [3, 3, 64, 128]
- vgg_16/conv2/conv2_1/biases **shape**: [128]
- vgg_16/conv2/conv2_2/weights **shape**: [3, 3, 128, 128]
- vgg_16/conv2/conv2_2/biases **shape**: [128]
- vgg_16/conv3/conv3_1/weights **shape**: [3, 3, 128, 256]
- vgg_16/conv3/conv3_1/biases **shape**: [256]
- vgg_16/conv3/conv3_2/weights **shape**: [3, 3, 256, 256]
- vgg_16/conv3/conv3_2/biases **shape**: [256]
- vgg_16/conv3/conv3_3/weights **shape**: [3, 3, 256, 256]
- vgg_16/conv3/conv3_3/biases **shape**: [256]
- vgg_16/conv4/conv4_1/weights **shape**: [3, 3, 256, 512]
- vgg_16/conv4/conv4_1/biases **shape**: [512]
- vgg_16/conv4/conv4_2/weights **shape**: [3, 3, 512, 512]
- vgg_16/conv4/conv4_2/biases **shape**: [512]
- vgg_16/conv4/conv4_3/weights **shape**: [3, 3, 512, 512]
- vgg_16/conv4/conv4_3/biases **shape**: [512]
- vgg_16/conv5/conv5_1/weights **shape**: [3, 3, 512, 512]
- vgg_16/conv5/conv5_1/biases **shape**: [512]
- vgg_16/conv5/conv5_2/weights **shape**: [3, 3, 512, 512]
- vgg_16/conv5/conv5_2/biases **shape**: [512]
- vgg_16/conv5/conv5_3/weights **shape**: [3, 3, 512, 512]
- vgg_16/conv5/conv5_3/biases **shape**: [512]
- vgg_16/fc6/weights **shape**: [7, 7, 512, 4096]
- vgg_16/fc6/biases **shape**: [4096]
- vgg_16/fc7/weights **shape**: [1, 1, 4096, 4096]
- vgg_16/fc7/biases **shape**: [4096]
- vgg_16/fc8/weights **shape**: [1, 1, 4096, 10]
- vgg_16/fc8/biases **shape**: [10]

To get the names *exact* you can use the **with tf.name_scope(name):** nested.

Then you have to create a saver for the variables you are intrested in. Since you are going to retrain
the last layer you can use:

    saver = tf.train.Saver(tf.trainable_variables()[:-2]
    
so you skip the last weight and bias. You can then restore the weights with:

    saver.restore(sess, 'vgg_16.ckpt')

**OBS!** you still have to initialize the other variables, but don't initialize the restored values after *restore*,
because then they are reset again...

Then you can try and run your network, and hopefully see that you get a good result 80% acc ish...

#### Good luck!