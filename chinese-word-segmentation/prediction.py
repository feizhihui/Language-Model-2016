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

import time

import numpy as np
import tensorflow as tf

import data_utils as reader


class CTBModel(object):
    """The CTB model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        # rnn神经元隐藏节点个数, embedding后的维度
        size = config.hidden_size
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        # sequence_lengths = tf.constant(np.ones(batch_size) * num_steps)
        self._sequence_lengths = tf.placeholder(tf.int32, shape=(batch_size))

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
        if is_training and config.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=config.keep_prob)
        lstm_net = tf.contrib.rnn.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)
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

        # ============================================
        # tf.squeeze remove all size 1 dimensions or by specifying `squeeze_dims`.
        # [batch_size, num_steps, size] => num_steps*[batch_size,1,size]=>num_steps*[batch_size,size]
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(inputs, num_steps, 1)]
        # 模型num_steps*[batch_size,hidden_size],和一个unit state [batch_size,size]
        outputs, state = tf.nn.dynamic_rnn(lstm_net, inputs, sequence_length=self._sequence_lengths,
                                           initial_state=self._initial_state)

        # =>[batch_size,hidden_size*num_steps]=>[batch_size*num_steps,hidden_size]
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])

        softmax_w = tf.get_variable(
            "softmax_w", [size, 4], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [4], dtype=tf.float32)
        # 定义rnn之后的全连接
        # [batch_size*num_steps,hidden_size]*[size,4]=[batch_size*num_steps,4]+[vocab_size]
        logits = tf.matmul(output, softmax_w) + softmax_b
        # 目标词语的平均负对数概率最小

        self._unary_scores = tf.reshape(logits, [-1, num_steps, 4])

        # Compute the log-likelihood of the gold sequences and keep the transition
        # params for inference at test time.
        #  unary_scores[batch_size,num_steps,4,4]
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            self._unary_scores, self._targets, sequence_lengths=self._sequence_lengths)
        self._transition_mat = transition_params
        # Add a training op to tune the parameters.
        loss = tf.reduce_mean(-log_likelihood)
        # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        # logits shape [batch_size*num_steps, 4]
        # print(logits)
        # print(self._targets)
        # correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.argmax(tf.reshape(self._targets, [-1, 4])))
        # self._accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32)) / (batch_size * num_steps)
        self._cost = loss
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        # 这个函数会返回图中所有trainable=True的variable
        tvars = tf.trainable_variables()
        # 待修剪的张量和比例 当gradients的L2模大于max_gras_norm时,则会等比例缩放
        # grads，global_norm=grads * max_grad_norm / max(global_norm，max_grad_norm)
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

        self._new_lr = tf.placeholder(tf.float32, shape=[])
        # 定义一个op：更新学习率
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def sequence_lengths(self):
        return self._sequence_lengths

    @property
    def transition_mat(self):
        return self._transition_mat

    @property
    def unary_scores(self):
        return self._unary_scores

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
    num_steps = 50
    hidden_size = 50
    max_epoch = 4
    max_max_epoch = 3
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 6000  # 5167


# crf的预测算法 tf_unary_scores(n,num_steps,4) tf_y(n,num_steps)
def cal_accuracy(tf_unary_scores, tf_y, tf_transition, sequence_len):
    correct_labels = 0
    total_labels = 0
    # 遍历每一个batch
    for tf_unary_scores_, y_, len_ in zip(tf_unary_scores, tf_y, sequence_len):
        tf_unary_scores_ = tf_unary_scores_[:len_]
        y_ = y_[:len_]
        viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
            tf_unary_scores_, tf_transition)
        correct_labels += np.sum(np.equal(viterbi_sequence, y_))
        total_labels += len_
    accuracy = 100.0 * correct_labels / float(total_labels)
    return accuracy


def run_epoch(session, model, tuple_data_, eval_op, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(tuple_data_[0]) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    accuracy = 0
    # 每迭代一整遍数据集,执行op:zero_state一次
    # tuple(num_layors*[batch_size,size])
    lstm_state_value = session.run(model.initial_state)
    sequence_len = np.ones(shape=(model.batch_size), dtype=np.int32) * model.num_steps
    # 一个batch训练一次
    for step, (x, y) in enumerate(reader.ctb_iterator(tuple_data_, model.batch_size, model.num_steps)):
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        feed_dict[model.sequence_lengths] = sequence_len
        # foreach num = num_layors
        for i, (c, h) in enumerate(model.initial_state):
            # feed shape([batch_zie=20,size=200])
            feed_dict[c] = lstm_state_value[i].c
            feed_dict[h] = lstm_state_value[i].h
        # feed_dict{x,y,c1,h1,c2,h2}
        cost, lstm_state_value, _, crf_feature, crf_y, crf_transition = session.run(
            [model.cost, model.final_state, eval_op, model.unary_scores, model.targets, model.transition_mat],
            feed_dict)

        acc = cal_accuracy(crf_feature, crf_y, crf_transition, sequence_len)
        accuracy += acc
        costs += cost  # batch中平均每一个样本的cost
        iters += model.num_steps
        # 每迭代1/10 epoch 打印一次模型损失度
        if verbose and step % (epoch_size // 10) == 10:
            print("%.1f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)), end=", ")
            print("Accuracy:%.3f%%" % (accuracy / (step + 1)))

    return np.exp(costs / iters), accuracy / (step + 1)


def main():
    target_dict = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
    reverse_target = {0: 'B', 1: 'M', 2: 'E', 3: 'S'}
    # 获得 训练集,验证集,测试集
    dictionary, reverse_dictionary = reader.get_dictionary('F:/Datas/msr_training/msr_training.utf8.ic')
    eval_config = Config()
    eval_config.batch_size = 1

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model", reuse=None):
            model = CTBModel(is_training=True, config=eval_config)
            saver = tf.train.Saver()
            saver.restore(session, "./CTB_Model/CTB_Variables.ckpt")
            while True:
                line = "。" + input('Please input a Chinese sentence:') + "。"
                words = []
                feed_dict = {}
                for word in line:
                    words.append(dictionary[word])

                words_length = len(words)
                if words_length <= model.num_steps:
                    words = words + [0] * (model.num_steps - words_length)
                else:
                    print("Please be sure sentence length is less than 49.")
                    continue
                feed_dict[model.input_data] = np.array([words])
                feed_dict[model.sequence_lengths] = np.array([words_length])

                crf_feature, crf_transition = session.run(
                    [model.unary_scores, model.transition_mat],
                    feed_dict)

                viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(crf_feature[0][:words_length], crf_transition)
                labels_seq = [reverse_target[x] for x in viterbi_sequence]
                # print(labels_seq)
                result = ""
                for i, h in enumerate(labels_seq[:-1]):
                    if h == 'S':
                        result = result + line[i] + '/'
                    elif h == 'B' or h == 'M':
                        result = result + line[i]
                    elif h == 'E':
                        result = result + line[i] + "/"
                    else:
                        raise Exception('Not in [S, B, M, E]')
                result += line[-1]
                result = result[2:-2]
                print(result)


if __name__ == "__main__":
    main()
