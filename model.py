# compatibility with python 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import librosa as dps
import tensorflow.contrib as tfc
import os as os
import matplotlib.pyplot as plt
import librosa.display as ldsp
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

class Record(object):
    pass


N = 200
K = 6
L = 10
M = 10
P = 14
R = 12

num_classes = 15
batch_size = 10

filenames = ('evaluation_setup/fold1_test.txt')

features_np,labels_np= np.genfromtxt(filenames,delimiter=' ', unpack=True, dtype=None)
print(features_np)
label_vect = []

'''

#do zapisu spektrogramów

for x in np.arange(len(features_np)):
    sig, fs = dps.core.load(features_np[x+941])
    print("krok:", x)
    S = dps.feature.melspectrogram(y=sig, sr=fs)
    S = ldsp.specshow(dps.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    print(type(S))
    zmienna = 'test/output' + str(x+941) + '.png'
    S.plot()
    fig = plt.gcf()
    fig.savefig(zmienna)

'''
#spectr_vect = tf.convert_to_tensor(spectr_vect)



def labels(input_label):
    input_label = np.char.decode(input_label)
    if input_label == "bus":
        true_label = 0
        return true_label
    elif input_label == 'cafe/restaurant':
        true_label = 1
        return true_label
    elif input_label == 'beach':
        true_label = 2
        return true_label
    elif input_label == 'car':
        true_label = 3
        return true_label
    elif input_label == 'city_center':
        true_label = 4
        return true_label
    elif input_label == 'forest_path':
        true_label = 5
        return true_label
    elif input_label == 'grocery_store':
        true_label = 6
        return true_label
    elif input_label == 'home':
        true_label = 7
        return true_label
    elif input_label == 'library':
        true_label = 8
        return true_label
    elif input_label == 'metro_station':
        true_label = 9
        return true_label
    elif input_label == 'office':
        true_label = 10
        return true_label
    elif input_label == 'park':
        true_label = 11
        return true_label
    elif input_label == 'residential_area':
        true_label = 12
        return true_label
    elif input_label == 'train':
        true_label = 13
        return true_label
    elif input_label == 'tram':
        true_label = 14
        return true_label


for x in np.arange(len(labels_np)):
    train_labels_for = labels_np[x]
    train_labels1_for = labels(train_labels_for)
    label_vect.append(train_labels1_for)

import os

spectr_vect = [os.path.join('spectrograms', 'output%d.png' % i) for i in range(14034)]


def parse(img_path, label):
    img = tf.read_file(img_path)
    img = tf.image.decode_png(img, 3)
    # img.set_shape([None, None, None, 3])
    # tf.image.resize_nearest_neighbor
    return img, label


# my_img = [/home/user/.../img.png, ...]
# label = [3, 4, 1, 5, ...]
train = tf.data.Dataset.from_tensor_slices((spectr_vect, label_vect))
train = train.shuffle(50000)
train = train.map(parse, 8)
train = train.batch(batch_size)
train = train.prefetch(batch_size)

train_iterator = train.make_initializable_iterator()
train_imgs, train_labels = train_iterator.get_next()
train_labels_one_hot = tf.one_hot(train_labels, num_classes, 1, 0)


def weights(output_size, nazwa):
    weight = tf.get_variable(nazwa, output_size, tf.float32, tfc.layers.xavier_initializer())
    return weight


def biases(output_size, nazwa):
    bias = tf.get_variable(nazwa, output_size, tf.float32, tfc.layers.xavier_initializer())
    return bias


def convolutional_layer(input, weigth, bias):
    layer = tf.nn.conv2d(input, weigth, strides=[1, 1, 1, 1], padding='SAME') + bias
    return layer


def convolutional_layer_2(input, weigth, bias):
    layer = tf.nn.conv2d(input, weigth, strides=[1, 2, 2, 1], padding='SAME') + bias
    return layer


def max_pool(input):
    layer = tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return layer


def weight_all():
    return


def biase_all():
    return


def model1(img):
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        W1 = weights([5, 5, 3, K], "1_w")
        B1 = biases(K, "first_bias")
        W2 = weights([5, 5, K, L], "2_w")
        B2 = biases(L, "sec_bias")
        W3 = weights([4, 4, L, M], "3_w")
        B3 = biases(M, "third_bias")
        W4 = weights([4, 4, M, P], "4_w")
        B4 = biases(P, "fourth_bias")
        W5 = weights([40 * 30 * P, N], "5_w")
        B5 = biases(N, "fifth_bias")
        W6 = weights([N, 15], "6_w")
        B6 = biases(15, "sixth_bias")

        print(tf.shape(img))
        img = tf.cast(img, tf.float32)
        # img = tf.reshape(img, shape=[-1, 320, 240, 1])

        Y1 = tf.nn.relu(convolutional_layer(img, W1, B1))

        # output 640x480  Y1 = 10x640x480x6 W2 = 5,5,6,10

        Y2 = convolutional_layer(Y1, W2, B2)
        Y2 = max_pool(Y2)
        Y2 = tf.nn.relu(Y2)

        # output 320x240  Y2 = 10x320x240x10 W3 = 5,5,10,14
        Y3 = convolutional_layer(Y2, W3, B3)
        Y3 = max_pool(Y3)
        Y3 = tf.nn.relu(Y3)

        # output 160x120  Y3 = 10x160x120x14 W4 = 5,5,14,18
        Y4 = convolutional_layer_2(Y3, W4, B4)
        Y4 = max_pool(Y4)
        Y4 = tf.nn.relu(Y4)

        # output 40x30  Y4 = 10x40x30x14
        YY = tf.reshape(Y4, shape=[-1, 40 * 30 * P])

        fc1 = tf.nn.relu(tf.matmul(YY, W5) + B5)

        output = tf.matmul(fc1, W6) + B6
        print(output)
        pred = tf.nn.softmax(output)
        print(pred)

        return output, pred


train_output, train_pred = model1(train_imgs)

# cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=train_labels, logits=train_output)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=train_output, labels=train_labels_one_hot)

loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

our_pred = (tf.argmax(train_pred, 1))

correct_prediction = tf.equal(tf.argmax(train_pred, 1), tf.cast(train_labels, tf.int64))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_accuracy = tf.summary.scalar('metrics/accuracy', accuracy)
train_loss = tf.summary.scalar('metrics/loss', loss)
stats = tf.summary.merge([train_accuracy, train_loss])

fwtrain = tf.summary.FileWriter(logdir='./training', graph=tf.get_default_graph())

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    i = 0
    for epoch in range(100):
        sess.run(train_iterator.initializer)
        while True:
            try:
                _, o_stats, acc, los = sess.run([optimizer, stats, accuracy, loss])
                fwtrain.add_summary(o_stats, i)
                i += 1

                print(epoch, acc, los)
            except tf.errors.OutOfRangeError:
                break
