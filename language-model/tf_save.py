# encoding=utf-8

import tensorflow as tf
import numpy as np

# # # restore 必须载入全部变量
# # ## Save to file
# # remember to define the same dtype and shape when restore
# W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
# b = tf.Variable([[3, 3, 3]], dtype=tf.float32, name='biases')
# W_ = tf.Variable([[7, 7, 7], [3, 4, 5]], dtype=tf.float32)
# print([(v.name, tf.shape(v)) for v in tf.trainable_variables()])
#
# # init= tf.initialize_all_variables() # tf 马上就要废弃这种写法
# # 替换成下面的写法:
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(init)
#
#     save_path = saver.save(sess, "./PTB_Model/save_net.ckpt")
#     print("Save to path: ", save_path)

# ====================Variable:0
# 先建立 W, b 的容器
W1 = tf.get_variable("weights", [3, 3], dtype=tf.float32)
b1 = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")
W_1 = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='Variable')

# 这里不需要初始化步骤 init= tf.initialize_all_variables()
print([(v.name, tf.shape(v)) for v in tf.trainable_variables()])

with tf.Session() as sess:
    # 提取变量
    # saver = tf.train.Saver()
    # saver.restore(sess, "PTB_Model/save_net.ckpt")
    tf.global_variables_initializer().run()
    print("weights:", sess.run(W1))
    print("biases:", sess.run(b1))
    print("W_", sess.run(W_1))
