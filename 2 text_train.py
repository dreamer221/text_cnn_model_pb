# encoding:utf-8
from __future__ import print_function
from text_model import TextConfig, TextCNN
from loader import batch_iter, process_file, build_vocab, read_category, read_vocab, \
    export_word2vec_vectors, get_training_word2vec_vectors
import tensorflow as tf
import numpy as np
import os
import time
from datetime import timedelta


def evaluate(sess, x_, y_):
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def train():
    print("Configuring TensorBoard and Saver...")
    tensorboard_dir = './tensorboard'  # 如果没有, 会自动创建
    model_file = './checkpoints/cnn_model.pb'
    if os.path.exists(model_file):
        print("模型已经存在")
        return

    # x_train, y_train = process_file(config.train_filename, word_to_id, cat_to_id, config.seq_length)
    # x_val, y_val = process_file(config.val_filename, word_to_id, cat_to_id, config.seq_length)
    # np.savez("train.npz", x=x_train, y=y_train)
    # np.savez("val.npz", x=x_val, y=y_val)

    x_train, y_train = np.load("train.npz")['x'], np.load("train.npz")['y']
    x_val, y_val = np.load("val.npz")['x'], np.load("val.npz")['y']

    print("训练集的形状:", x_train.shape, y_train.shape)  # ( , 300) ( , 10)
    print("验证集的形状:", x_val.shape, y_val.shape)  # (5000, 300) (5000, 10)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)
    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        writer.add_graph(session.graph)

        best_val_accuracy = 0
        last_improved = 0  # record global_step at best_val_accuracy
        require_improvement = 1000  # 超过1000次迭代没有提升就提前结束训练
        flag = False

        for epoch in range(config.num_epochs):
            batch_train = batch_iter(x_train, y_train, config.batch_size)
            # start = time.time()
            print('Epoch:', epoch + 1)
            for x_batch, y_batch in batch_train:
                feed_dict = feed_data(x_batch, y_batch, config.keep_prob)
                _, global_step, train_summaries, train_loss, train_accuracy = session.run([model.optim, model.global_step,
                                                                                           merged_summary, model.loss,
                                                                                           model.acc], feed_dict=feed_dict)
                if global_step % config.print_per_batch == 0:
                    val_loss, val_accuracy = evaluate(session, x_val, y_val)
                    writer.add_summary(train_summaries, global_step)

                    # If improved, save the model
                    if val_accuracy > best_val_accuracy:
                        # print("global_step:", global_step)
                        # # 第一种保存模型方式
                        # saver.save(session, save_path, global_step=global_step)  # 如果加上global-step参数,那么会保存最后的5次保存模型,
                        #                                                          # 如果不加这个参数,只保存最后一次的模型
                        # # 第二种保存模型方式
                        output_graph_def = tf.graph_util.convert_variables_to_constants(session, session.graph_def,
                                                                                        output_node_names=["input_x",
                                                                                                           "input_y",
                                                                                                           'dropout',
                                                                                                           "output/predict",
                                                                                                           "loss/loss",
                                                                                                           "accuracy/accuracy"])

                        with tf.gfile.FastGFile(model_file, mode='wb') as f:
                            f.write(output_graph_def.SerializeToString())

                        best_val_accuracy = val_accuracy
                        last_improved = global_step
                        improved_str = '*'
                    else:
                        improved_str = ''
                    print("step:{},train loss: {:.3f}, train acc: {:.3f}, val loss: {:.3f}, val acc: {:.3f},{}".format(
                            global_step, train_loss, train_accuracy, val_loss, val_accuracy, improved_str))

                if global_step - last_improved > require_improvement:
                    print("No optimization over 1000 steps, stop training")
                    flag = True  # 这种使用标志的方法就可以跳出内部的循环
                    break
            if flag:
                break
            if val_accuracy > 0.9 and epoch % 3 == 0:
                config.lr *= config.lr_decay  # 学习率衰减

        # # 第二种保存模型方式
        output_graph_def = tf.graph_util.convert_variables_to_constants(session, session.graph_def,
                                                                        output_node_names=["input_x",
                                                                                           "input_y",
                                                                                           'dropout',
                                                                                           "output/predict",
                                                                                           "loss/loss",
                                                                                           "accuracy/accuracy"])

        with tf.gfile.FastGFile(model_file, mode='wb') as f:
            f.write(output_graph_def.SerializeToString())


if __name__ == '__main__':
    print('Configuring CNN model...')
    config = TextConfig()
    filenames = [config.train_filename, config.test_filename, config.val_filename]

    if not os.path.exists(config.vocab_filename):
        # 根据文本, 创建词典
        build_vocab(filenames, config.vocab_filename, config.vocab_size)  # 创建词汇表文件

    # 类别和类别对应的id
    categories, cat_to_id = read_category()

    # 词汇和词汇在词汇表中的ID
    words, word_to_id = read_vocab(config.vocab_filename)
    config.vocab_size = len(words)  # 这样是因为有可能所有的词汇也不够指定的词汇数量

    # 将word2vec保存为二进制文件,每行是对应词汇 id 的词向量
    if not os.path.exists(config.vector_word_npz):
        export_word2vec_vectors(word_to_id, config.vector_word_filename, config.vector_word_npz)

    # 读取词向量的二进制文件的
    config.pre_trianing = get_training_word2vec_vectors(config.vector_word_npz)

    model = TextCNN(config)
    train()
