# encoding=utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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

        self._pred = pred = tf.argmax(logits, 1)
        correct_pred = tf.equal(tf.cast(pred, tf.int32), tf.reshape(self._targets, [-1]))
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

    @property
    def pred(self):
        return self._pred


class TestConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 1
    hidden_size = 200
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 1
    vocab_size = 10000


def predict(session, model, vocab, id_to_word, lstm_state_value):
    word = raw_input('Please input one word:')
    if word not in vocab:
        print('Non-existent!')
        return True
    print('id:', vocab[word])

    feed_dict = {}
    feed_dict[model.input_data] = [[vocab[word]]]

    for i, (c, h) in enumerate(model.initial_state):
        # 可feed可不feed shape([batch_zie=20,size=200])
        feed_dict[c] = lstm_state_value[i].c
        feed_dict[h] = lstm_state_value[i].h

    pred_value, lstm_state_value = session.run([model.pred, model.final_state],
                                               feed_dict)
    print(id_to_word[pred_value[0]])

    return True


def main():
    # 获得 训练集,验证集,测试集
    raw_data = reader.ptb_raw_data('/dataset/PTB/')
    train_data, valid_data, test_data, vocab = raw_data
    # 根据模型规模config{small,medium,large,or test}
    # 获得2套模型参数,参数封装在config类的属性当中
    config = TestConfig()
    with tf.Graph().as_default(), tf.Session() as session:
        with tf.variable_scope("model"):
            m = PTBModel(is_training=True, config=config)

        saver = tf.train.Saver()
        saver.restore(session, "PTB_Model/PTB_Variables.ckpt")

        print(vocab)
        rvocab = dict(zip(vocab.values(), vocab.keys()))
        lstm_state_value = session.run(m.initial_state)
        while (predict(session, m, vocab, rvocab, lstm_state_value)):
            pass


if __name__ == "__main__":
    main()
