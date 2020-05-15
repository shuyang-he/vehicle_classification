#coding=utf-8

from PIL import Image
import numpy as np
import tensorflow as tf
import os

data_dir = "train/work_van"

def read_data(data_dir):
    datas = []
    fpaths = []
    count = 0
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = Image.open(fpath)
        image = image.resize((256, 256))
        data = np.array(image)
        datas.append(data)
        if count >= 500:
            break
        count += 1
    datas = np.array(datas)

    print("shape of datas: {}".format(datas.shape))
    return fpaths, datas

fpaths, datas = read_data(data_dir)

datas_placeholder = tf.placeholder(tf.float32, [None, 256, 256, 3])
#labels_placeholder = tf.placeholder(tf.int32, [None])

dropout_placeholdr = tf.placeholder(tf.float32)

conv1 = tf.layers.conv2d(
      inputs=datas_placeholder,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#tf.layers.conv2d(datas_placeholder, 20, 3, padding="same", activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
#tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#tf.layers.conv2d(pool1, 40, 3, padding="same", activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
#tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

conv3=tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

conv4=tf.layers.conv2d(
      inputs=pool3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

# feature extraction done.

flatten = tf.layers.flatten(pool4)

#re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

dense1 = tf.layers.dense(inputs=flatten, 
                      units=1024, 
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.nn.l2_loss)
dense2= tf.layers.dense(inputs=dense1, 
                      units=512, 
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.nn.l2_loss)


#fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

sess=tf.Session()
sess.run(tf.global_variables_initializer()) 

 
fea=sess.run(dense2,feed_dict={datas_placeholder: datas[0:498]})
print(fea.shape)
np.savetxt('work_van.csv', fea, delimiter = ',')

sess.close()

