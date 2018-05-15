# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import numpy as np

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
SUMMARY_DIR = "./log"



# 模型保存的路径和文件名
MODEL_SAVE_PATH = "./models/"
MODEL_NAME = "model.ckpt"





def train(mnist):
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, [
                            None,
                            mnist_inference.IMAGE_SIZE,
                            mnist_inference.IMAGE_SIZE,
                            mnist_inference.NUM_CHANNELS],
                           name='x-input')
        #
        tf.summary.image('input_image', x, 10)
        #
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference.inference(x, train=True, regularizer=regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数，学习率，滑动平均操作及训练过程
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        #
        # tf.summary.scalar('cross entropy', cross_entropy_mean)
        #
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        #
        tf.summary.scalar('whole losses', loss)
        #
    with tf.name_scope("train_step"):
        # learning_rate = tf.train.exponential_decay(
        #     LEARNING_RATE_BASE,
        #     global_step,
        #     mnist.train.num_examples / BATCH_SIZE,
        #     LEARNING_RATE_DECAY)
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        train_step = tf.train.AdamOptimizer(LEARNING_RATE_BASE).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

        # train accuracy#
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy_train = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_validate = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('training accuracy', accuracy_train)
        tf.summary.scalar('validation accuracy', accuracy_validate)


    # 将当前的计算图输出到TensorBoard日志文件
    merged = tf.summary.merge_all()

    xs_v = mnist.validation.images
    reshaped_xs_v = np.reshape(xs_v, (mnist.validation.num_examples,
                                  mnist_inference.IMAGE_SIZE,
                                  mnist_inference.IMAGE_SIZE,
                                  mnist_inference.NUM_CHANNELS))

    validated_feed = {x: reshaped_xs_v, y_: mnist.validation.labels}



    # 初始化Tensorflow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        tf.initialize_all_variables().run()

        for i in xrange(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,    (BATCH_SIZE,
                                             mnist_inference.IMAGE_SIZE,
                                             mnist_inference.IMAGE_SIZE,
                                             mnist_inference.NUM_CHANNELS))

            summary, _, loss_value, step, train_ac = sess.run(
                [merged, train_op, loss, global_step, accuracy_train], feed_dict={x: reshaped_xs, y_: ys})
            validate_ac = sess.run(accuracy_validate, feed_dict=validated_feed)


            if i % 1000 == 0:

                # 配置运行时需要记录的信息
                # run_options = tf.RunOptions(
                #     trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()

                # summary, _, loss_value, step , accuracy_train = sess.run([merged, train_op, loss, global_step, accuracy_train], feed_dict={x: reshaped_xs, y_: ys},
                #                                options=run_options, run_metadata=run_metadata)

                # summary_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                # summary_writer.add_summary(summary, i)

                print("After %d training step(s), loss on training batch is %g, training accuracy is %g, validation accuracy is %g" % (step, loss_value, train_ac, validate_ac))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
            # else:
                # summary, _, loss_value, step, accuracy_train = sess.run([merged, train_op, loss, global_step, accuracy_train], feed_dict={x: reshaped_xs, y_: ys})
                # summary_writer.add_summary(summary, i)
            summary_writer.add_summary(summary, i)

    summary_writer.close()



def main(argv=None):
    mnist = input_data.read_data_sets("/export/Data/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()