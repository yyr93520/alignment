#! /usr/bin/python
# -*- coding: utf-8 -*-

import re

def transfer(infile, outfile):
    with open(outfile, 'w') as fout:
        with open(infile, 'r') as fin:
            for line in fin.readlines():
                lst = line.strip().split(',')
                new_lst = []
                for row in lst:
                    if re.match(r'^[0-9]+$', row):
                        new_lst.append(row)
                fout.write(' '.join(new_lst))
                fout.write('\n')


#transfer('douban_net', 'douban.txt')
transfer('weibo_retweet_net', 'weibo.txt')