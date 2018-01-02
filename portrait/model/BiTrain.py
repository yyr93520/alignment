#! /usr/bin/python
# -*- coding: utf-8 -*-

import ConfigParser
import numpy as np
import sys
sys.path.append("../../")
import csv
import os
import shutil
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from preprocess import PreProcess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

class BiModel(object):

    def __init__(self, config_fp):
        self.feature_name = self.__class__.__name__
        self.feature_fp = None
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)
        self.proportion = 0
        self.feature_type = 0

    def frange(self, x, y, jump):
        while x < y:
            yield x
            x += jump

    def printaccuracy(self, predict, testTag):
        accuracy = accuracy_score(testTag, predict)
        print 'Model`s accuracy is %f' % (accuracy)
        print 'Model`s score report is \n'
        print classification_report(testTag, predict)
        print 'Model`s confusion is \n'
        print confusion_matrix(testTag, predict)
        print '\n'

    def load_features(self, file_name):
        train_features = None
        test_features = None
        feature_pt = self.config.get('FEATURE', 'bi_train_pt')
        for i in range(len(file_name)):
            feature_fp = '%s/%s_train.txt' % (feature_pt, file_name[i])
            feature = np.loadtxt(feature_fp, delimiter=' ')
            if train_features is None:
                train_features = feature
            else:
                train_features = np.concatenate((train_features, feature), axis=1)
            feature_fp = '%s/%s_test.txt' % (feature_pt, file_name[i])
            feature = np.loadtxt(feature_fp, delimiter=' ')
            if test_features is None:
                test_features = feature
            else:
                test_features = np.concatenate((test_features, feature), axis=1)
        self.x_train = train_features
        self.x_test = test_features
        y_fp = '%s/y_train.txt' % (feature_pt)
        self.y_train = np.loadtxt(y_fp, delimiter=' ')
        y_fp = '%s/y_test.txt' % (feature_pt)
        self.y_test = np.loadtxt(y_fp, delimiter=' ')
        align_file = '%s/align_test.txt' % (feature_pt)
        self.align_lst = []
        with open(align_file, 'r') as fin:
            for line in fin.readlines():
                self.align_lst.append(line.strip().split(' '))

    def get_test_result(self, prob, test_num):
        prob_dict = {}
        align_dict = {}
        num = 0
        for uids in self.align_lst:
            if not prob_dict.has_key(uids[0]):
                prob_dict[uids[0]] = {}
            prob_dict[uids[0]][uids[1]] = prob[num]
            if self.y_test[num] == 1:
                align_dict[uids[0]] = uids[1]
            num += 1
        num = [0 for i in range(test_num)]
        mrr_all = 0
        for key in prob_dict:
            # predict_uid = max(prob_dict[key], key = prob_dict[key].get)
            predict_sort = sorted(prob_dict[key].items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
            predict_uid = [v[0] for v in predict_sort]
            mrr_one = 0
            if not align_dict.has_key(key):
                continue
            for i in range(test_num):
                if i >= len(predict_uid):
                    continue
                if align_dict[key] in predict_uid[:(i + 1)]:
                    num[i] += 1
                if align_dict[key] == predict_uid[i]:
                    mrr_one += 1.0 / (i + 1)

            mrr_all += mrr_one

        for i in range(test_num):
            print "%d accuracy is %f" % (i + 1, num[i] * 1.0 / len(align_dict))
        return mrr_all / len(align_dict)

    def add_layer(self, inputs, in_size, out_size, n_layer, activation_function = None):
        layer_name = 'layer%s' % n_layer
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                Weights = tf.Variable(tf.truncated_normal([in_size,out_size]))
                tf.summary.histogram(layer_name + "/weights", Weights)
            with tf.name_scope('biases'):
                biases = tf.Variable(tf.zeros([1, out_size]))
                tf.summary.histogram(layer_name + "/biases", biases)
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.matmul(inputs, Weights) + biases  # inputs*Weight+biases
                tf.summary.histogram(layer_name + "/Wx_plus_b", Wx_plus_b)
            if activation_function is None:
                outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name + "/outputs", outputs)
            return outputs

    def tf_train_test(self, h1_units, epoch, test_num):
        in_units = self.x_train.shape[1]
        with tf.name_scope('inputs'):
            xs = tf.placeholder(tf.float32, [None, in_units], name='x_input')
            y_not_one_hot = tf.placeholder(tf.int32, shape=[None], name='y_input')

        y_ = tf.one_hot(y_not_one_hot, 2)

        layer1 = self.add_layer(xs, in_units, h1_units, n_layer=1, activation_function=tf.nn.relu)
        #layer2 = self.add_layer(layer1, h1_units, h2_units, n_layer=2, activation_function=tf.nn.relu)
        prediction = self.add_layer(layer1, h1_units, 2, n_layer=2, activation_function=tf.nn.softmax)
        with tf.name_scope('loss'):
            cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
            tf.summary.scalar('loss', cross_entropy)
        with tf.name_scope('train'):
            train_step = tf.train.AdagradOptimizer(0.1).minimize(cross_entropy)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        # 合并到Summary中
        merged = tf.summary.merge_all()
        if os.path.exists('log/'):
            shutil.rmtree('log/')
        os.makedirs('log/')
        writer = tf.summary.FileWriter("log/", sess.graph)
        sess.run(init)

        for i in range(epoch):
            sess.run(train_step, feed_dict={xs: self.x_train, y_not_one_hot: self.y_train})

            """
            if i % 50 == 0:
                result = sess.run(merged, feed_dict={xs: self.x_train, y_not_one_hot: self.y_train})  # merged也是需要run的
                writer.add_summary(result, i)  # result是summary类型的，需要放入writer中，i步数（x轴）
            """

        result = sess.run(prediction, feed_dict={xs: self.x_test, y_not_one_hot: self.y_test})
        result = result[:, 1]
        #correct_prediction = tf.argmax(prediction, 1)
        #test_tag = sess.run(correct_prediction, feed_dict={xs: self.x_test, y_not_one_hot: self.y_test})
        #self.printaccuracy(test_tag, self.y_test)
        mrr = self.get_test_result(result, test_num)
        self.save_roc_data(self.y_test, result)
        roc_auc = self.get_roc(result)
        return mrr, roc_auc

    def save_roc_data(self, true_tag, pred_tag):
        roc_fp = '%s/%s_%.2f.txt' % (self.config.get('FEATURE', 'bi_roc_pt'), self.feature_type, self.proportion)
        with open(roc_fp, 'wb') as fout:
            writer = csv.writer(fout)
            writer.writerow(true_tag)
            writer.writerow(pred_tag)

    def get_roc(self, predict):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(self.y_test, predict)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        #plt.title('Receiver Operating Characteristic')
        #plt.plot(false_positive_rate, true_positive_rate, 'b',
        #         label='AUC = %0.2f' % roc_auc)
        #plt.legend(loc='lower right')
        #plt.plot([0, 1], [0, 1], 'r--')
        #plt.xlim([-0.1, 1.2])
        #plt.ylim([-0.1, 1.2])
        #plt.ylabel('True Positive Rate')
        #plt.xlabel('False Positive Rate')
        #plt.savefig('%s/roc_%s_%.2f.png' % (self.config.get('FEATURE', 'roc_pt'), self.feature_type, self.proportion))
        #plt.close('all')
        return roc_auc

    def batch_test(self, config_fp):
        fout = open('%s/ours_result.csv' % self.config.get('FEATURE', 'result_pt'), 'wb')
        writer = csv.writer(fout)
        feature_1 = []
        feature_2 = []
        for prop in self.frange(0.005, 1, 0.005):
            self.proportion = prop
            PreProcess(config_fp).get_bi_feature(proportion=prop, test_sample = 9)
            print '========== result ============\n'
            print 'train set propertion is %f' % prop
            print 'features are [embedding]'
            self.load_features(['embedding'])
            self.feature_type = 1
            mrr, roc_auc = self.tf_train_test(80, 3000, 10)
            feature_1.append([1, self.proportion, mrr, roc_auc])
            print 'features are [embedding,feature]'
            #self.load_features(['embedding', 'feature'])
            #self.feature_type = 2
            #mrr, roc_auc = self.tf_train_test(60, 3000, 10)
            #feature_2.append([2, self.proportion, mrr, roc_auc])
        writer.writerows(feature_1)
        #writer.writerows(feature_2)
        fout.close()

if __name__ == '__main__':
    config_fp = '/home/yangyaru/project/alignment/portrait/extract/myconf.ini'
    BiModel(config_fp).batch_test(config_fp)
