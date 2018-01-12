#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import os
#import shutil
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
#import numpy as np
import matplotlib.pyplot as plt

# 创建一个神经网络层
def add_layer(input, in_size, out_size, layer_name, activation_function=None):
    """
    input:      神经网络层的输入
    in_zize:    输入数据的大小
    out_size:   输出数据的大小
    layer_name  神经网络层的名字
    activation_function:  神经网络激活函数，默认没有
    """
    with tf.name_scope('%s'%(layer_name)):
        # 定义神经网络的初始化权重
        with tf.name_scope('w'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	    tf.summary.histogram(layer_name+'/w', Weights)
        # 定义神经网络的偏置
        with tf.name_scope('b'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	    tf.summary.histogram(layer_name + '/b', biases)
        # 计算w*x+b
        with tf.name_scope('wx_b'):
            W_mul_x_plus_b = tf.matmul(input, Weights) + biases
	    tf.summary.histogram(layer_name + '/wx_b', W_mul_x_plus_b)
        # 进行dropout，可以注释和不注释来对比dropout的效果
	W_mul_x_plus_b = tf.nn.dropout(W_mul_x_plus_b, keep_prob)
	tf.summary.histogram(layer_name + '/wx_b_drop', W_mul_x_plus_b)
	    # 根据是否有激活函数进行处理
	if activation_function is None:
	    output = W_mul_x_plus_b
	else:
	    output = activation_function(W_mul_x_plus_b)
	# 查看权重变化
	tf.summary.histogram(layer_name + '/output', output)
    return output

def draw(train_list):
    """ 绘制训练曲线，便于观察 """
    #[step, loss_train, loss_test, accuracy_train, accuracy_test]
    plt.figure('training process')
    step_list = [i[0] for i in train_list] # step
    loss_train = [i[1] for i in train_list] # loss
    loss_test = [i[2] for i in train_list] # loss_dev
    accuracy_train = [i[3] for i in train_list] # accuracy
    accuracy_test = [i[4] for i in train_list] # accuracy_dev
    plt.subplot(211)
    plt.plot(step_list, loss_train, loss_test, '.')
    plt.legend(['train', 'test'])
    plt.grid()
    plt.ylabel('loss')
    plt.title('loss & accuracy')
    plt.subplot(212)
    plt.plot(step_list, accuracy_train, accuracy_test)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    #plt.title('accuracy')
    plt.savefig('out.png')
    plt.show()

def main():
    digits = load_digits() # 加载数据
    X = digits.data # 输入数据
    y = digits.target # 输出数据
    #显示图片样例
    """
    import numpy as np
    plt.imshow(np.reshape(X[0],[8,8]))
    plt.show()
    """
    # 标签变换 ont-hot编码
    y = LabelBinarizer().fit_transform(y)
    # 训练集和测试集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    #=========model==========
        # 定义dropout的placeholder
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    # 输入数据64个特征
    with tf.name_scope('input'):
        xs = tf.placeholder(tf.float32, [None, 64], name='x')  # 8x8
        # tensorboard上显示图片
        tf.summary.image('xs', tf.reshape(xs, [-1,8,8,1]), 10)
        ys = tf.placeholder(tf.float32, [None, 10], name='y')
    # 添加隐藏层和输出层
    layer1 = add_layer(xs, 64, 50, 'layer1', activation_function=tf.nn.tanh)
    layer2 = add_layer(layer1, 50, 80, 'layer2', activation_function=tf.nn.sigmoid)
    layer3 = add_layer(layer2, 80, 40, 'layer3', activation_function=tf.nn.tanh) # relu报错
    layer4 = add_layer(layer3, 40, 30, 'layer4', activation_function=tf.nn.tanh) # relu报错
    layer5 = add_layer(layer4, 30, 30, 'layer5', activation_function=tf.nn.tanh) # relu报错
    layer6 = add_layer(layer5, 30, 30, 'layer6', activation_function=tf.nn.tanh) # relu报错
    prediction = add_layer(layer6, 30, 10, 'prediction', activation_function=tf.nn.softmax)
    #pre = tf.nn.softmax(layer2, name="pre")
    #========================
    # 计算loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
        # 存储loss
        tf.summary.scalar('loss', loss)
    # 神经网络训练
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    # 准确率
    with tf.name_scope('accuracy'):
        correct_predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
        tf.summary.scalar('accuracy', accuracy) #可视化观看常量
    # 定义Session
    sess = tf.Session()
    # 收集所有的数据
    merged = tf.summary.merge_all()
    main_dir = '.'
    # 将数据写入到tensorboard中
    log_dir = '%s/log/dropout'%(main_dir)
    # tensorflow管理文件夹
    if tf.gfile.Exists(log_dir):
        print 'tf:日志目录已存在，删除%s'%(log_dir)
        tf.gfile.DeleteRecursively(log_dir)
    #__import__('shutil').rmtree(log_dir) # 简洁版
    # python方式管理文件夹
    '''
    if os.path.isdir(log_dir):
        print 'python:日志目录已存在，删除%s'%(log_dir)
        shutil.rmtree(log_dir)
    '''
    train_writer = tf.summary.FileWriter("%s/train"%(log_dir), sess.graph)
    test_writer = tf.summary.FileWriter("%s/test"%(log_dir), sess.graph)
    # step, init前定义
    step = tf.Variable(0, name="step", trainable=False)
    # 根据tensorflow版本选择初始化函数
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    # 模型保存
    #saver  = tf.train.saver()
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=5) # 保留最近5个模型
    model_dir = '%s/model/dropout'%(main_dir)
    if not tf.gfile.Exists(model_dir):
        tf.gfile.MakeDirs(model_dir)
    else:
        tf.gfile.DeleteRecursively(model_dir)
        tf.gfile.MakeDirs(model_dir)
    # 执行初始化
    sess.run(init)
    delta = 10
    train_info = []
    # 进行训练迭代
    for i in range(500):
        # 执行训练，dropout为1-0.5=0.5
        current_step = tf.train.global_step(sess, step)
        feed_dict_train = {xs: X_train, ys: y_train, keep_prob: 1.0}
        feed_dict_test = {xs: X_test, ys: y_test, keep_prob: 0.8}
        step_train, loss_train, accuracy_train = sess.run([step, loss, accuracy], feed_dict_train)
        loss_test, accuracy_test = sess.run([loss, accuracy], feed_dict_test)
        train_info.append([step_train, loss_train, loss_test, accuracy_train, accuracy_test])
        if i % delta == 0:
            # 记录损失
            train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1.0})
            test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1.0})
            train_writer.add_summary(train_result, i)
            test_writer.add_summary(test_result, i)
            print '第%s步, train_loss=%s, train accurcy=%s, test_loss=%s, test accuracy=%s'%(i, loss_train, accuracy_train, loss_test, accuracy_test) #pylint:disable=line-too-long
            path = saver.save(sess, model_dir+'/checkpoint', global_step=current_step)
            #saver.restore(sess, '../temp/nn_model')
            print "Saved model checkpoint to {}\n".format(path)
    draw(train_info)

if __name__ == '__main__':
    main()
    
