# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    #if type(x) == list:
     #   x = np.array(x)
    #x = x.flatten()
    #o_h = np.zeros((len(x), n))
    o_h = np.zeros(n)
    o_h[x] = 1.
    #o_h[np.arange(len(x)), x] = 1
    return o_h


num_classes = 3
batch_size = 5

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image),  one_hot([i], num_classes)  # [one_hot(float(i), num_classes)]
        #image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch

# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, batch, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)
        #o5 = tf.layers.conv2d(inputs=o4, filters=128, kernel_size=3, activation=tf.nn.relu)
        #o6 = tf.layers.max_pooling2d(inputs=o5, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 33 * 64]), units=5, activation=tf.nn.relu)
        # 3 Capas Convolutivas
        #h = tf.layers.dense(inputs=tf.reshape(o6, [batch_size * 3, 8 * 15 * 128]), units=5, activation=tf.nn.relu)
        # 1 Capa Convolutiva
        #h = tf.layers.dense(inputs=tf.reshape(o2, [batch_size * 3, 39 * 69 * 32]), units=5, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y

example_batch_train, label_batch_train = dataSource(["data4/Training/0/*.jpg", "data4/Training/1/*.jpg", "data4/Training/2/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["data4/Validation/0/*.jpg", "data4/Validation/1/*.jpg", "data4/Validation/2/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test= dataSource(["data4/Testing/0/*.jpg", "data4/Testing/1/*.jpg", "data4/Testing/2/*.jpg"], batch_size=batch_size)

example_batch_train_predicted = myModel(example_batch_train, batch_size, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, batch_size, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, batch_size, reuse=True)

label_batch_train = tf.cast(label_batch_train, tf.float32)
label_batch_valid = tf.cast(label_batch_valid, tf.float32)
label_batch_test = tf.cast(label_batch_test, tf.float32)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - label_batch_train))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - label_batch_valid))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

list_error_training = []
list_error_validation = []
error_training = 0
error_validation = 0
error = -1

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for _ in range(300):
        sess.run(optimizer)
        if _ % 20 == 0:
            print("Iter:", _, "---------------------------------------------")
            print(sess.run(label_batch_train))
            print(sess.run(example_batch_train_predicted))
            #print("Error:", sess.run(cost_valid))
            error_training = sess.run(cost)
            error_validation = sess.run(cost_valid)
            print("Error Entrenamiento: ", error_training)
            print("Error Validación: ", error_validation)
            list_error_training.append(error_training)
            list_error_validation.append(error_validation)

        #errorP = error
        #error = error_validation
        #if (abs(error_validation - errorP) < 0.5): 
         #   break


    n_accert = 0
    dataset_testing = len(label_batch_test.eval())
    result_testing = sess.run(example_batch_test_predicted)

    for estimate, real in zip(label_batch_test.eval(), result_testing):
        if np.argmax(estimate) == np.argmax(real):
            n_accert = n_accert + 1

    capacidad_predictiva = (n_accert / dataset_testing) * 100
    print("La capacidad predictiva de la red neuronal es: " + str(capacidad_predictiva) + "%")


    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)

x_error_train = list(range(1, len(list_error_training) + 1))

plt.plot(list_error_training)
plt.title("Gráfica Conjunto de Entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Error")
plt.show()

plt.plot(list_error_validation)
plt.title("Gráfica Conjunto de Validación")
plt.xlabel("Épocas")
plt.ylabel("Error")
plt.show()
