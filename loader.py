# encoding:utf-8
from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
# import codecs
import re
import jieba


def read_file(filename):
    re_han = re.compile(u"([\u4E00-\u9FD5a-zA-Z]+)")  # the method of cutting text by punctuation
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.rstrip()
                assert len(line.split('\t')) == 2
                label, content = line.split('\t')
                labels.append(label)
                blocks = re_han.split(content)  # 使用非汉字和非字母进行分割
                word = []
                for blk in blocks:
                    if re_han.match(blk):  #
                        #
                        for w in jieba.cut(blk):
                            if len(w) >= 2:
                                word.append(w)
                contents.append(word)
            except:
                pass
    return labels, contents
    # contents = [['马晓旭', '意外',...], [], []] # 每个子元素是一篇新闻的分词


def build_vocab(filenames, vocab_dir, vocab_size=8000):
    all_data = []  # 所有词汇的列表
    for filename in filenames:
        _, data_train = read_file(filename)
        for content in data_train:
            all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1) # 出现频率最高的vocab_size - 1个词汇
    words, _ = list(zip(*count_pairs))
    words = ['<PAD>'] + list(words)  # 在 top k前加一个<pad> 表示之前没有出现的词汇,这样words就是出现的最多的vocab个词汇

    with open(vocab_dir, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    words = open(vocab_dir, 'r', encoding='utf-8').read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    # categories = ["产业", "体育", "健康", "军事", "政治", "教育", "文化", "法律", "生态", "社会", "科技", "经济", "金融", "领导人"]
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def process_file(filename, word_to_id, cat_to_id, max_length):
    labels, contents = read_file(filename)
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    with open("test_label.txt", 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(label + "\n")

    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')
    # 如果data_id 元素的长度小于max_length则在后面加0补充,如果大于长度则从后面剪切
    y_pad = kr.utils.to_categorical(label_id)  # 将一维变为多维
    return x_pad, y_pad


def batch_iter(x, y, batch_size=64):
    """ minibatch 数据生成器"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def export_word2vec_vectors(vocab, word2vec_dir, trimmed_filename):
    """ 将 word2vec 文件 根据 词汇对应的ID 转换为矩阵 """
    file_r = open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.split(' ')) # word2vec文件第一行是(词汇数,维度): 370695 100

    embeddings = np.zeros([len(vocab), vec_dim])  # 指定二维矩阵的形状 

    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(vec)
        except:
            pass
        line = file_r.readline()

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_training_word2vec_vectors(filename):
    """读取上个函数保存的二进制词向量"""
    with np.load(filename) as data:
        return data["embeddings"]
