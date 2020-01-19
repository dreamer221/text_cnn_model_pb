# encoding:utf-8
from text_model import *
import tensorflow as tf
import tensorflow.contrib.keras as kr
from loader import read_vocab
import jieba
import re
import codecs


def predict(sentences):
    _, word_to_id = read_vocab(config.vocab_filename)
    input_x2 = process_file(sentences, word_to_id, max_length=config.seq_length)
    labels = {0: '体育',
              1: '财经',
              2: '房产',
              3: '家居',
              4: '教育',
              5: '科技',
              6: '时尚',
              7: '时政',
              8: '游戏',
              9: '娱乐'}
    output_graph_def = tf.compat.v1.GraphDef()
    with open(model_file, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")
    with tf.Session() as session:
        input_x = session.graph.get_tensor_by_name("input_x:0")
        keep_prob = session.graph.get_tensor_by_name("dropout:0")
        prediction = session.graph.get_tensor_by_name("output/predict:0")

        feed_dict = {input_x: input_x2, keep_prob: 1.0}
        y_prob = session.run(prediction, feed_dict=feed_dict)
        y_prob = y_prob.tolist()
        cat = []

        for prob in y_prob:
            cat.append(labels[prob])
        return cat


def sentence_cut(sentences):
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%]+)")  # the method of cutting text by punctuation
    seglist = []
    for sentence in sentences:
        words = []
        blocks = re_han.split(sentence)
        for blk in blocks:
            if re_han.match(blk):
                words.extend(jieba.lcut(blk))
        seglist.append(words)
    return seglist


def process_file(sentences, word_to_id, max_length=600):
    data_id = []
    seglist = sentence_cut(sentences)
    for i in range(len(seglist)):
        data_id.append([word_to_id[x] for x in seglist[i] if x in word_to_id])
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    return x_pad


if __name__ == '__main__':
    import random

    config = TextConfig()
    model_file = './checkpoints/cnn_model.pb'
    sentences = []
    labels = []
    with codecs.open('./data/cnews.test.txt', 'r', encoding='utf-8') as f:
        sample = random.sample(f.readlines(), 5)
        for line in sample:
            try:
                line = line.rstrip().split('\t')
                assert len(line) == 2
                sentences.append(line[1])
                labels.append(line[0])
            except:
                pass
    cat = predict(sentences)
    for i, sentence in enumerate(sentences, 0):
        print('----------------------the text-------------------------')
        print(sentence[:50] + '....')
        print('the  true   label:%s' % labels[i])
        print('the predict label:%s' % cat[i])
