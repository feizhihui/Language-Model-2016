# encoding=utf-8
"""
 Small config.
   init_scale = 0.1    # 相关参数的初始值为随机均匀分布，范围是[-init_scale,+init_scale]
   learning_rate = 1.0 # 学习速率，此值还会在模型学习过程中下降
   max_grad_norm = 5   # 用于控制梯度膨胀，如果梯度向量的L2模超过max_grad_norm，则等比例缩小
   num_layers = 2      # LSTM层数
   num_steps = 20      # 分隔句子的粒度大小，每次会把num_steps个单词划分为一句话(但是本模型与seq2seq模型不同，它仅仅是1对1模式，句子长度应该没有什么用处)。
   hidden_size = 200   # 隐层单元数目，每个词会表示成[hidden_size]大小的向量
   max_epoch = 4       # epoch<max_epoch时，lr_decay值=1,epoch>max_epoch时,lr_decay逐渐减小
   max_max_epoch = 13  # 完整的文本要循环的次数
   keep_prob = 1.0     # dropout率，1.0为不丢弃
   lr_decay = 0.5      # 学习速率衰减指数
   batch_size = 20     # 和num_steps共同作用，要把原始训练数据划分为batch_size组，每组划分为n个长度为num_steps的句子。
   vocab_size = 10000  # 单词数量(这份训练数据中单词刚好10000种)

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import reader


class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        lstm_net = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
        # 定义模型最初始的state tuple(num_layers*[batch_size,size])
        self._initial_state = lstm_net.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            # 定义embedding矩阵维度[vocab_size, size]
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=tf.float32)
            # 通过input_data中各元素的id找到embedding后的inputs, shape([batch_size, num_steps, size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        # ============================================
        # tf.squeeze remove all size 1 dimensions or by specifying `squeeze_dims`.
        # [batch_size, num_steps, size] => num_steps*[batch_size,1,size]=>num_steps*[batch_size,size]
        inputs = [tf.squeeze(input_, [1])
                  for input_ in tf.split(1, num_steps, inputs)]
        # 模型num_steps*[batch_size,hidden_size],和一个unit state [batch_size,size]
        outputs, state = tf.nn.rnn(lstm_net, inputs, initial_state=self._initial_state)

        # RNN variable_scope中 每一个LSTMCELL都有一个matrix与bias变量，还有一个state(c,h)能够被feed
        print([(v.name, tf.shape(v)) for v in tf.trainable_variables()])

        # =====================================
        # outputs = []
        # state = self._initial_state
        # with tf.variable_scope("RNN"):
        #     for time_step in range(num_steps):
        #         if time_step > 0: tf.get_variable_scope().reuse_variables()
        #         (cell_output, state) = lstm_net(inputs[:, time_step, :], state)
        #         outputs.append(cell_output)
        # =====================================
        # =>[batch_size,hidden_size*num_steps]=>[batch_size*num_steps,hidden_size]
        output = tf.reshape(tf.concat(1, outputs), [-1, size])

        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        # 定义rnn之后的全连接
        # [batch_size*num_steps,hidden_size]*[size,vocab_size]=[batch_size*num_steps,vocab_size]+[vocab_size]
        logits = tf.matmul(output, softmax_w) + softmax_b
        # 目标词语的平均负对数概率最小
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],  # 2D Tensors List
            [tf.reshape(self._targets, [-1])],  # 1D Tensors List
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])  # weights

        correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(self._targets, [-1]))
        self._accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32)) / batch_size

        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        # 这个函数会返回图中所有trainable=True的variable
        tvars = tf.trainable_variables()
        # 待修剪的张量和比例 当gradients的L2模大于max_gras_norm时,则会等比例缩放
        # grads，global_norm=grads * max_grad_norm / max(global_norm，max_grad_norm)
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(tf.float32, shape=[])
        # 定义一个op：更新学习率
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


class Config(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, model, data, eval_op, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    costs = 0.0
    accs = 0.0
    iters = 0
    # 每迭代一整遍数据集,执行op:zero_state一次
    # tuple(num_layors*[batch_size,size])
    lstm_state_value = session.run(model.initial_state)
    for step, (x, y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps)):
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        # foreach num = num_layors
        for i, (c, h) in enumerate(model.initial_state):
            # 可feed可不feed shape([batch_zie=20,size=200])
            feed_dict[c] = lstm_state_value[i].c
            feed_dict[h] = lstm_state_value[i].h
        # feed_dict{x,y,c1,h1,c2,h2}
        cost, acc, lstm_state_value, _ = session.run([model.cost, model.accuracy, model.final_state, eval_op],
                                                     feed_dict)
        accs += acc
        costs += cost  # batch中平均每一个样本的cost
        iters += model.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))
            print("Accuracy:", accs / iters)

    return np.exp(costs / iters), accs / iters


def main():
    # 获得 训练集,验证集,测试集
    raw_data = reader.ptb_raw_data('/home/feizhihui/MyData/dataset/PTB/')
    train_data, valid_data, test_data, _ = raw_data
    # 根据模型规模config{small,medium,large,or test}
    # 获得2套模型参数,参数封装在config类的属性当中
    config = Config()
    eval_config = Config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as session:
        # 创建一个初始化的容器
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        # 分别创建训练模型,验证模型,测试模型
        # reuse=None 不使用以往参数
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=True, config=config)
        # 使用以往的参数
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=False, config=config)
            mtest = PTBModel(is_training=False, config=eval_config)

        tf.global_variables_initializer().run()

        for i in range(config.max_max_epoch):
            # 0.5**(0,..,0 and 1 and 2,.. )
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            # 学习率在max_epoch之后开始随迭代次数下降
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity, train_accuracy = run_epoch(session, m, train_data, m.train_op,
                                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f, Train Accuracy: %.3f"
                  % (i + 1, train_perplexity, train_accuracy))
            valid_perplexity, valid_accuracy = run_epoch(session, mvalid, valid_data, tf.no_op(), verbose=True)
            print("Epoch: %d Valid Perplexity: %.3f, Valid Accuracy: %.3f"
                  % (i + 1, valid_perplexity, valid_accuracy))

        test_perplexity, test_accuracy = run_epoch(session, mtest, test_data, tf.no_op(), verbose=True)
        print("Test Perplexity: %.3f, Test Accuracy: %.3f" % (test_perplexity, test_accuracy))
        saver = tf.train.Saver()
        save_path = saver.save(session, "./PTB_Model/PTB_Variables.ckpt")
        print("Save to path: ", save_path)


if __name__ == "__main__":
    main()
