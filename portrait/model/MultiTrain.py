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
from sklearn.metrics import roc_curve, auc
from preprocess import PreProcess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'

class MultiModel(object):

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

    def load_features(self, file_name):
        train_features = None
        test_features = None
        train_y = None
        test_y = None
        feature_pt = self.config.get('FEATURE', 'multi_train_pt')
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
            y_fp = '%s/y_%s_train.txt' % (feature_pt, file_name[i])
            y = np.loadtxt(y_fp, delimiter=' ')
            if train_y is None:
                train_y = y
            else:
                train_y = np.concatenate((train_y, y), axis=1)
            y_fp = '%s/y_%s_test.txt' % (feature_pt, file_name[i])
            y = np.loadtxt(y_fp, delimiter=' ')
            if test_y is None:
                test_y = y
            else:
                test_y = np.concatenate((test_y, y), axis=1)
        self.x_train = train_features
        self.x_test = test_features
        self.y_train = train_y
        self.y_test = test_y

    def get_test_result(self, prob):
        prob_dict = {}
        align_dict = {}
        num = 0
        id = 1111
        for value in prob:
            if not prob_dict.has_key(id):
                prob_dict[id] = {}
            prob_dict[id][num] = value
            num += 1
            if num >= 10:
                id += 1
                num = 0
        num = [0 for i in range(10)]
        map_all = 0
        mrr_all = 0
        for key in prob_dict:
            # predict_uid = max(prob_dict[key], key = prob_dict[key].get)
            predict_sort = sorted(prob_dict[key].items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
            predict_uid = [v[0] for v in predict_sort]
            map_one = 0
            mrr_one = 0
            for i in range(10):
                if i >= len(predict_uid):
                    continue
                if 0 in predict_uid[:(i + 1)]:
                    num[i] += 1
                    map_one += 1
                if predict_uid[i] == 0:
                    mrr_one += 1.0 / (i + 1)
            map_one = map_one * 1.0 / 10
            map_all += map_one
            mrr_all += mrr_one
        for i in range(10):
            print "%d accuracy is %f." % (i + 1, num[i] * 1.0 / len(prob_dict))
        #print "map is %f." % (map_all / len(prob_dict))
        return mrr_all / len(prob_dict)

    def add_layer(self, inputs, in_size, out_size, n_layer, activation_function=None):
        layer_name = 'layer%s' % n_layer
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                Weights = tf.Variable(tf.truncated_normal([in_size, out_size]))
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

    def save_roc_data(self, true_tag, pred_tag):
        roc_fp = '%s/%s_%.2f.txt' % (self.config.get('FEATURE', 'multi_roc_pt'), self.feature_type, self.proportion)
        with open(roc_fp, 'wb') as fout:
            writer = csv.writer(fout)
            writer.writerow(true_tag)
            writer.writerow(pred_tag)

    def tf_train_test(self):
        #self.load_features(file_name)
        # sess = tf.InteractiveSession()
        in_units = self.x_train.shape[1]
        out_units = self.y_train.shape[1]
        h1_units = int((in_units + out_units) * 2 / 3)
        #h1_units = int((in_units * 2) ** 0.5)
        h2_units = h1_units
        epoch = 5000
        with tf.name_scope('inputs'):
            xs = tf.placeholder(tf.float32, [None, in_units], name='x_input')
            y_ = tf.placeholder(tf.float32, [None, out_units], name='y_input')

        layer1 = self.add_layer(xs, in_units, h1_units, n_layer=1, activation_function=tf.nn.relu)
        layer2 = self.add_layer(layer1, h1_units, h2_units, n_layer=2, activation_function=tf.nn.relu)
        prediction = self.add_layer(layer2, h2_units, out_units, n_layer=3)
        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(tf.square(y_ - prediction))
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

        for i in range(epoch):
            sess.run(train_step, feed_dict={xs: self.x_train, y_: self.y_train})
            """
            if i % 50 == 0:
                result = sess.run(merged, feed_dict={xs: self.x_train, y_: self.y_train})  # merged也是需要run的
                writer.add_summary(result, i)  # result是summary类型的，需要放入writer中，i步数（x轴）
            """
        accuracy = tf.reduce_mean(tf.square(tf.subtract(y_, prediction)), reduction_indices=[1])
        result = sess.run(accuracy, feed_dict={xs: self.x_test, y_: self.y_test})
        mrr = self.get_test_result(result)
        roc_auc = self.get_roc(result)
        self.save_roc_data(self.y_test, result)
        return mrr, roc_auc

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
        #plt.savefig('%s/pale/roc_%s_%.2f.png' % (self.config.get('FEATURE', 'roc_pt'), self.feature_type, self.proportion))
        return roc_auc

    def batch_test(self, config_fp):
        fout = open('%s/pale_result.csv' % self.config.get('FEATURE', 'result_pt'), 'wb')
        writer = csv.writer(fout)
        features = []
        for prop in self.frange(0.05, 1, 0.05):
            self.proportion = prop
            PreProcess(config_fp).get_multi_feature(proportion=prop)
            print '========== result ============\n'
            print 'train set propertion is %f' % prop
            print 'features are [embedding]'
            self.feature_type = 1
            self.load_features(['embedding'])
            mrr, roc_auc = self.tf_train_test()
            features.append([1, self.proportion, mrr, roc_auc])
        writer.writerows(features)
        fout.close()

    def print_evaluation(self):
        eval_fp = "%s/accuracy.csv" % self.config.get('FEATURE', 'data_pt')
        eval_data = csv.DictReader(open(eval_fp, 'r'), delimiter = ',')
        MRR = []
        TOP1 = []
        prop = []
        for row in eval_data:
            prop.append(float(row['train_set']))
            MRR.append(float(row['MRR']))
            TOP1.append(float(row['TOP1']))
        """
        #plt.subplot(211)
        plt.title("MRR")
        x1 = prop[0:19]
        y1 = MRR[0:19]
        x2 = prop[19:38]
        y2 = MRR[19:38]
        x3 = prop[38:]
        y3 = MRR[38:]
        plt.plot(x1, y1, label = 'now-embedding')
        plt.plot(x2, y2, label = 'now-embedding-profile')
        plt.plot(x3, y3, label = 'before-embedding')
        plt.legend()
        plt.xlabel('train set proportion')
        plt.ylabel('MRR')
        """
        #plt.subplot(212)
        plt.title("TOP1")
        x1 = prop[0:19]
        y1 = TOP1[0:19]
        x2 = prop[19:38]
        y2 = TOP1[19:38]
        x3 = prop[38:]
        y3 = TOP1[38:]
        plt.plot(x1, y1, label='now-embedding')
        plt.plot(x2, y2, label='now-embedding-profile')
        plt.plot(x3, y3, label='before-embedding')
        plt.legend()
        plt.xlabel('train set proportion')
        plt.ylabel('TOP1')


        plt.savefig('%s/top1.png' % self.config.get('FEATURE', 'data_pt'))

if __name__ == '__main__':
    config_fp = '/home/yangyaru/project/alignment/portrait/extract/myconf.ini'
    MultiModel(config_fp).batch_test(config_fp)
    #MultiModel(config_fp).print_evaluation()