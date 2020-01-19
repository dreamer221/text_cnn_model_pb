# encoding:utf-8
from __future__ import print_function
from text_model import TextConfig, TextCNN
from loader import batch_iter, process_file, build_vocab, read_category, read_vocab, \
    export_word2vec_vectors, get_training_word2vec_vectors
from sklearn import metrics
import tensorflow as tf
import numpy as np
import sys
import os
import time
from datetime import timedelta


def evaluate(input_x, input_y, keep_prob, loss1, acc1, sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {input_x: x_batch, input_y: y_batch, keep_prob: 1.0}
        loss, acc = sess.run([loss1, acc1], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    return total_loss / data_len, total_acc / data_len


def tes():
    model_file = './checkpoints/cnn_model.pb'
    # x_test, y_test = process_file(config.test_filename, word_to_id, cat_to_id, config.seq_length)
    # np.savez("test.npz", x=x_test, y=y_test)
    data = np.load("test.npz")
    x_test, y_test = data['x'], data['y']

    output_graph_def = tf.compat.v1.GraphDef()
    with open(model_file, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as session:
        input_x = session.graph.get_tensor_by_name("input_x:0")
        input_y = session.graph.get_tensor_by_name("input_y:0")
        keep_prob = session.graph.get_tensor_by_name("dropout:0")
        loss = session.graph.get_tensor_by_name("loss/loss:0")
        acc = session.graph.get_tensor_by_name("accuracy/accuracy:0")
        prediction = session.graph.get_tensor_by_name("output/predict:0")
        test_loss, test_accuracy = evaluate(input_x, input_y, keep_prob, loss, acc, session, x_test, y_test)
        msg = 'Test Loss: {0:>.2}, Test Acc: {1:>.2%}'
        print(msg.format(test_loss, test_accuracy))

        batch_size = config.batch_size
        data_len = len(x_test)
        num_batch = int((data_len - 1) / batch_size) + 1
        y_test_cls = np.argmax(y_test, 1)
        y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            feed_dict = {input_x: x_test[start_id:end_id], keep_prob: 1.0}
            y_pred_cls[start_id:end_id] = session.run(prediction, feed_dict=feed_dict)

        # evaluate
        print("Precision, Recall and F1-Score...")
        print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

        print("Confusion Matrix...")
        cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
        print(cm)


if __name__ == '__main__':
    config = TextConfig()
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(config.vocab_filename)
    config.vocab_size = len(words)
    tes()
