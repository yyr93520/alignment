#! /usr/bin/python
# -*- coding: utf-8 -*-

import ConfigParser
import numpy as np
import csv
import random
import os

class PreProcess(object):

    def __init__(self, config_fp):
        self.feature_name = self.__class__.__name__
        self.feature_fp = None
        self.config = ConfigParser.ConfigParser()
        self.config.read(config_fp)
        self.source_embed_dict = self.get_embedding_dict('source', 'all')
        self.target_embed_dict = self.get_embedding_dict('target', 'all')
        self.source_feature_dict = self.get_feature_dict('Douban')
        self.target_feature_dict = self.get_feature_dict('Weibo')

    def get_small_data(self):
        douban_fp = '%s/douban.txt' % (self.config.get('FEATURE', 'data_pt'))
        adj_list = []
        adj_data = csv.reader(open(douban_fp, 'r'), delimiter = ' ')
        for row in adj_data:
            for j in range(1, len(row)):
                if len(row[j]) == 0:
                    continue
                else:
                    a = row[0]
                    b = row[j]
                    adj_list.append([a, b])
        new_data = []
        for i in range(500000):
            new_data.append(random.choice(adj_list))
        adj_dict = {}
        for pair in new_data:
            if not adj_dict.has_key(pair[0]):
                adj_dict[pair[0]] = set()
            adj_dict[pair[0]].add(pair[1])
        num = 0
        with open('douban_small.txt', 'w') as fout:
            for key in adj_dict:
                fout.write(key + ' ')
                num += len(adj_dict[key])
                fout.write(' '.join(adj_dict[key]))
                fout.write('\n')
        print num


    def get_embedding_dict(self, node_label, suffix):
        #print 'embedding start!'
        embed_dict = {}
        embedding_pt = self.config.get('FEATURE', 'embedding_pt')
        embedding_file = '%s/%s_%s.txt' % (embedding_pt, node_label, suffix)
        with open(embedding_file, 'r') as embedding:
            embedding.readline()
            for line in embedding.readlines():
                lst = line.strip().split(" ")
                embed_dict[lst[0]] = lst[1:]
        #print 'embedding end!'
        return embed_dict

    def get_feature_dict(self, node_label):

        feature_pt = self.config.get('FEATURE', 'feature_pt')
        userid_fp = feature_pt + '/' + node_label + '.UserID.txt'
        feature_fp = feature_pt + '/' + node_label + '.txt'
        feature_dict = {}
        #print 'feature start!'
        if not os.path.exists(feature_fp):
            return None
        with open(userid_fp, 'r') as fin1, open(feature_fp, 'r') as fin2:
            fin1.readline()
            for line in fin1.readlines():
                feature = fin2.readline().strip().split(' ')
                feature_dict[line.strip()] = feature
        #print 'feature end!'
        return feature_dict

    def write_list(self, file_name, list_name):
        with open(file_name, 'w') as fout:
            for row in list_name:
                if isinstance(row, list):
                    fout.write(' '.join(row))
                elif isinstance(row, str):
                    fout.write(row)
                else:
                    fout.write(str(row))
                fout.write('\n')

    def get_bi_feature(self, proportion = 0.8, train_sample = 2, test_sample = 9):
        #print "bi feature start!"
        align_fp = self.config.get('FEATURE', 'align_fp')
        align_list = []
        align_dict = {}
        negative_set = []

        #one to one style
        with open(align_fp, 'r') as fin:
            for line in fin.readlines():
                lst = line.strip().split(' ')
                source = lst[0]
                target = lst[1]
                align_dict[source] = [target]
                if not self.source_embed_dict.has_key(source) or not self.target_embed_dict.has_key(target):
                    # or not self.source_feature_dict.has_key(source) or not self.target_feature_dict.has_key(target):
                    continue
                align_list.append([source, target])
                negative_set.append(target)

        '''
        with open(align_fp, 'r') as fin:
            for line in fin.readlines():
                lst = line.strip().split(',')
                source = lst[0]
                target_lst = lst[1].split(';')
                align_dict[source] = target_lst
                for target in target_lst:
                    if not self.source_embed_dict.has_key(source) or not self.target_embed_dict.has_key(target):
                            #or not self.source_feature_dict.has_key(source) or not self.target_feature_dict.has_key(target):
                        continue
                    align_list.append([source, target])
                    negative_set.append(target)
        '''
        num = int(len(align_list) * proportion)
        y_train = []
        y_test = []
        embedding_train = []
        embedding_test = []
        feature_train = []
        feature_test = []
        align_test = []
        #negative_set = [v for v in self.target_embed_dict.keys() if v in self.target_feature_dict.keys()]
        index = 0
        for row in align_list:
            index += 1
            if index <= num:
                segment = 'train'
            else:
                segment = 'test'
            eval('y_' + segment).append(1)
            embed_tensor = self.source_embed_dict[row[0]] + self.target_embed_dict[row[1]]
            eval('embedding_' + segment).append(embed_tensor)
            #feature_tensor = self.source_feature_dict[row[0]] + self.target_feature_dict[row[1]]
            #eval('feature_' + segment).append(feature_tensor)
            if index > num:
                align_test.append([row[0], row[1]])
            for i in range(eval(segment + '_sample')):
                negative_sample = random.choice(negative_set)
                if negative_sample in align_dict[row[0]]:
                    continue
                eval('y_' + segment).append(0)
                embed_tensor = self.source_embed_dict[row[0]] + self.target_embed_dict[negative_sample]
                eval('embedding_' + segment).append(embed_tensor)
                #feature_tensor = self.source_feature_dict[row[0]] + self.target_feature_dict[negative_sample]
                #eval('feature_' + segment).append(feature_tensor)
                if index > num:
                    align_test.append([row[0], negative_sample])

        train_pt = self.config.get('FEATURE', 'bi_train_pt')
        self.write_list('%s/%s.txt' % (train_pt, 'y_train'), y_train)
        self.write_list('%s/%s.txt' % (train_pt, 'y_test'), y_test)
        self.write_list('%s/%s.txt' % (train_pt, 'embedding_train'), embedding_train)
        self.write_list('%s/%s.txt' % (train_pt, 'embedding_test'), embedding_test)
        #self.write_list('%s/%s.txt' % (train_pt, 'feature_train'), feature_train)
        #self.write_list('%s/%s.txt' % (train_pt, 'embedding_2_test'), embedding_2_test)
        self.write_list('%s/%s.txt' % (train_pt, 'feature_train'), feature_train)
        self.write_list('%s/%s.txt' % (train_pt, 'feature_test'), feature_test)
        self.write_list('%s/%s.txt' % (train_pt, 'align_test'), align_test)

    def get_multi_feature(self, proportion = 0.8, train_sample = 0, test_sample = 9):
        align_fp = self.config.get('FEATURE', 'align_fp')
        align_dict = {}
        negative_set = []
        align_list = []
        with open(align_fp, 'r') as fin:
            for line in fin.readlines():
                lst = line.strip().split(',')
                source = lst[0]
                target_lst = lst[1].split(';')
                align_dict[source] = target_lst
                for target in target_lst:
                    if not self.source_embed_dict.has_key(source) or not self.target_embed_dict.has_key(target):
                        continue
                    align_list.append([source, target])
                    negative_set.append(target)
        num = int(len(align_list) * proportion)
        y_embedding_train = []
        y_embedding_test = []
        embedding_train = []
        embedding_test = []

        index = 0
        for row in align_list:
            index += 1
            if index <= num:
                segment = 'train'
            else:
                segment = 'test'
            eval('embedding_' + segment).append(self.source_embed_dict[row[0]])
            eval('y_embedding_' + segment).append(self.target_embed_dict[row[1]])
            #eval('embedding_' + segment).append(row)


            for i in range(eval(segment + '_sample')):
                negative_sample = random.choice(negative_set)
                if negative_sample in align_dict[row[0]]:
                    continue
                eval('embedding_' + segment).append(self.source_embed_dict[row[0]])
                eval('y_embedding_' + segment).append(self.target_embed_dict[negative_sample])
                #eval('embedding_' + segment).append([row[0], negative_sample])


        train_pt = self.config.get('FEATURE', 'multi_train_pt')
        self.write_list('%s/%s.txt' % (train_pt, 'y_embedding_train'), y_embedding_train)
        self.write_list('%s/%s.txt' % (train_pt, 'y_embedding_test'), y_embedding_test)
        #self.write_list('%s/%s.txt' % (train_pt, 'y_embedding_2_train'), y_embedding_2_train)
        #self.write_list('%s/%s.txt' % (train_pt, 'y_embedding_2_test'), y_embedding_2_test)
        #self.write_list('%s/%s.txt' % (train_pt, 'embedding_2_train'), embedding_2_train)
        self.write_list('%s/%s.txt' % (train_pt, 'embedding_train'), embedding_train)
        self.write_list('%s/%s.txt' % (train_pt, 'embedding_test'), embedding_test)
        #self.write_list('%s/%s.txt' % (train_pt, 'embedding_2_train'), embedding_2_test)
        #self.write_list('%s/%s.txt' % (train_pt, 'align_test'), align_test)

if __name__ == '__main__':
    config_fp = '/home/yangyaru/project/alignment/portrait/extract/myconf.ini'
    #PreProcess(config_fp).get_align_list()
    #PreProcess(config_fp).get_bi_feature()
    PreProcess(config_fp).get_multi_feature()