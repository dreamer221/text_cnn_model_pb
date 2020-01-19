# encoding:utf-8
import tensorflow as tf


class TextConfig():
    embedding_size = 100  # dimension of word embedding
    vocab_size = 8000  # number of vocabulary
    pre_trianing = None  # use vector_char trained by word2vec

    seq_length = 200  # 600 max length of sentence
    num_classes = 10  # 4  # number of labels

    num_filters = 64  # 128  # number of convolution kernel
    filter_sizes = [2, 3, 4]  # size of convolution kernel

    keep_prob = 0.6  # droppout
    lr = 1e-3  # learning rate
    lr_decay = 0.8  # learning rate decay
    clip = 6.0  # gradient clipping threshold
    l2_reg_lambda = 0.02  # l2 regularization lambda

    num_epochs = 10  # epochs
    batch_size = 64  # batch_size
    print_per_batch = 100  # print result

    train_filename = './data/train.txt'  # train data
    test_filename = './data/test.txt'  # test data
    val_filename = './data/val.txt'  # validation data

    vocab_filename = './data/vocab.txt'  # vocabulary
    vector_word_filename = './data/vector_word.txt'  # vector_word trained by word2vec
    vector_word_npz = './data/vector_word.npz'  # save vector_word to numpy file


class TextCNN(object):
    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, shape=[None, self.config.seq_length], name='input_x')
        print("self.input_x :", self.input_x)
        # tf.placeholder() 参数:类型, 形状, 名称

        self.input_y = tf.placeholder(tf.float32, shape=[None, self.config.num_classes], name='input_y')
        print("self.input_y :", self.input_y)
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        print("self.keep_prob :", self.keep_prob)
        self.global_step = tf.Variable(0, trainable=False)  # , name='global_step')
        self.l2_loss = tf.constant(0.0)
        print("self.l2_loss :", self.l2_loss)
        self.cnn()

    def cnn(self):
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable("embeddings", # 初始化矩阵是使用词向量
                                             shape=[self.config.vocab_size, self.config.embedding_size],
                                             initializer=tf.constant_initializer(self.config.pre_trianing))
            # tf.constant_initializer(const)：常量初始化
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)
            print("self.embedding_inputs:", self.embedding_inputs)  # shape=(?, 300, 100)
            self.embedding_inputs_expanded = tf.expand_dims(self.embedding_inputs, -1) # 指定位置增加一个维度
            print("self.embedding_inputs_expanded:", self.embedding_inputs_expanded)  # shape=(?, 300, 100, 1)

        with tf.name_scope('cnn'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    filter_shape = [filter_size, self.config.embedding_size, 1, self.config.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    # tf.truncated_normal与tf.random_normal的作用都是从给定均值和方差的正态分布中输出变量。
                    # 两者的区别在于tf.truncated_normal 截取的是两个标准差以内的部分，换句话说就是截取随机变量更接近于均值

                    conv = tf.nn.conv2d(self.embedding_inputs_expanded, W, strides=[1, 1, 1, 1],
                                        padding="VALID", name="conv")
                    b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="b")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # tf.nn.bias_add 是 tf.add 的一个特例，tf.nn.bias_add 中 bias 一定是 1 维的张量；
                    # tf.nn.bias_add 中 value 最后一维长度和 bias 的长度一定得一样；
                    print("h :", h)  # shape=(?, 299, 1, 64)  shape=(?, 298, 1, 64)  shape=(?, 297, 1, 64)
                    pooled = tf.nn.max_pool(h, ksize=[1, self.config.seq_length - filter_size + 1, 1, 1],
                                            # ksize 第一个维度是样本个数,一般都是1; 第二个维度是高,
                                            # 第三个维度是宽, 第四个维度是通道
                                            strides=[1, 1, 1, 1], padding='VALID', name="pool")
                                            # strides 的第一项和第四项都是1, 第二项是高度的步伐,第三个是宽度的步伐
                    print("pooled :", pooled)  # shape=(?, 1, 1, 64);  shape=(?, 1, 1, 64) ; shape=(?, 1, 1, 64)
                    pooled_outputs.append(pooled)

            num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
            print("num_filters_total:", num_filters_total)  # 64*3=192

            self.h_pool = tf.concat(pooled_outputs, 3)  # 在指定的维度上叠加
            print("self.h_pool :", self.h_pool)  # shape=(?, 1, 1, 192),

            self.outputs = tf.reshape(self.h_pool, [-1, num_filters_total])
            print("self.outputs :", self.outputs)  # shape=(?, 192)

        with tf.name_scope("dropout"):
            self.final_output = tf.nn.dropout(self.outputs, self.keep_prob)
            print("self.final_output :", self.final_output)  # shape=(?, 192)

        with tf.name_scope('output'):
            fc_w = tf.get_variable('fc_w',  # 初始化
                                   shape=[self.final_output.shape[1].value, self.config.num_classes],
                                   initializer=tf.contrib.layers.xavier_initializer())
            # tf.contrib.layers.xavier_initializer() 随机权重矩阵初始化,每一层的输出与输入必须是同方差的，
            # 并且前向传播与反向传播时梯度也是同方差的。初始权重值应当为均值为0，的正态分布，这种策略称为Xavier初始化
            print("fc_w :", fc_w)  # (192, 10)
            fc_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_classes]), name='fc_b')
            # tf.constant() 返回常量值tensor
            # tf.Variable() 返回变量,根据指定初始化
            print("fc_b :", fc_b)  # shape=(10,)

            self.logits = tf.matmul(self.final_output, fc_w) + fc_b  # tf.matmul 矩阵相乘
            print("self.logits :", self.logits)   # shape=(?, 10)

            # self.prob = tf.nn.softmax(self.logits)
            # print("self.prob :", self.prob)  # shape=(?, 10)

            self.y_pred_cls = tf.argmax(self.logits, 1, name='predict')  # 每一行最大值所在的列的索引
            print("self.y_pred_cls :", self.y_pred_cls)  # shape=(?,)


        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            print("cross_entropy :", cross_entropy)  # shape=(?,)
            self.l2_loss += tf.nn.l2_loss(fc_w)
            self.l2_loss += tf.nn.l2_loss(fc_b)
            #
            self.loss = tf.add(x=tf.reduce_mean(cross_entropy), y=self.config.l2_reg_lambda * self.l2_loss, name='loss')
            # 为什么正则化只加最后一层的w和b
            print("self.loss :", self.loss)  # 一个数

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)  # 优化器
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # ??
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)  # 梯度剪切阈值
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)  # 最终的优化器

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)  # correct_pred 是bool类型
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')  # 转换成float后求平均值
            print("self.acc:", self.acc)
