import os
import random
import collections

import torch
import numpy as np
import pandas as pd

from data_loader.loader_base import DataLoaderBase


class DataLoader(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)

        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size
        #print(self.kg_file)
        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info(logging)


    def construct_data(self, kg_data):
        '''
            kg_data 为 DataFrame 类型
        '''
        # 1. 为KG添加逆向三元组，即对于KG中任意三元组(h, r, t)，添加逆向三元组 (t, r+n_relations, h)，
        #    并将原三元组和逆向三元组拼接为新的DataFrame，保存在 self.kg_data 中。
        #目的：数据增广（有向->无向）
        inverse_kg_data = kg_data.copy()
        #print(kg_data['r'])
        n_relations=len(kg_data['r'].unique())
        print(n_relations)
        inverse_kg_data['r'] += n_relations#
        inverse_kg_data = inverse_kg_data[['t', 'r', 'h']]
        print(kg_data.shape)
        self.kg_data = pd.concat([kg_data, inverse_kg_data])#ignore_index=True表示忽略原来的索引，重新生成索引
        print(self.kg_data.shape)

        # 2. 计算关系数，实体数和三元组的数量
        self.n_relations = len(self.kg_data['r'].unique())
        print("n_relations:",self.n_relations)
        #self.n_entities = max(len(self.kg_data['h']), len(self.kg_data['t'])) + 1
        tmp_entities = pd.concat([self.kg_data['h'], self.kg_data['t']]).unique()
        self.n_entities = len(tmp_entities)+1
        print("n_entities:",self.n_entities)
        self.n_kg_data = len(self.kg_data)
        print("n_kg_data:",self.n_kg_data)

        # 3. 根据 self.kg_data 构建字典 self.kg_dict ，其中key为h, value为tuple(t, r)，
        #    和字典 self.relation_dict，其中key为r, value为tuple(h, t)。
        self.kg_dict = collections.defaultdict(list)#保存以一个头实体为首的所有三元组
        self.relation_dict = collections.defaultdict(list)#保存以r为关系的所有三元组
        #构建KG_dict和relation_dict
        for i,row in self.kg_data.iterrows():#iterrows()函数将DataFrame的每一行迭代为(index, Series)对，可以通过row['h']来获取h的值
            h = row['h']
            r = row['r']
            t = row['t']
            self.kg_dict[h].append((t,r))
            self.relation_dict[r].append((h,t))
        #for h, r, t in self.kg_data.values:
        #    self.kg_dict[h].append((t, r))
        #    self.relation_dict[r].append((h, t))
        
        



    def print_info(self, logging):
        logging.info('n_users:      %d' % self.n_users)
        logging.info('n_items:      %d' % self.n_items)
        logging.info('n_entities:   %d' % self.n_entities)
        logging.info('n_relations:  %d' % self.n_relations)

        logging.info('n_cf_train:   %d' % self.n_cf_train)
        logging.info('n_cf_test:    %d' % self.n_cf_test)

        logging.info('n_kg_data:    %d' % self.n_kg_data)


