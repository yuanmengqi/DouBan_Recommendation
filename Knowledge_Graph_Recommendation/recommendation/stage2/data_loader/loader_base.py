import os
import time
import random
import collections

import torch
import numpy as np
import pandas as pd


class DataLoaderBase(object):

    def __init__(self, args, logging):
        self.args = args
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain

        self.data_dir = os.path.join(args.data_dir, args.data_name)
        #self.train_file = os.path.join(self.data_dir, 'train.txt')
        self.train_file = self.data_dir + '/train.txt'
        self.test_file = os.path.join(self.data_dir, 'test.txt')
        #self.kg_file = os.path.join(self.data_dir, "kg_final.txt")
        self.kg_file = self.data_dir + "/kg_final.txt"

        self.cf_train_data, self.train_user_dict = self.load_cf(self.train_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(self.test_file)
        self.statistic_cf()


    def load_cf(self, filename):
        user = []
        item = []
        user_dict = dict()#用户字典，key为用户id，value为该用户交互过的item的id列表

        lines = open(filename, 'r').readlines()
        for l in lines:
            #print(l)
            tmp = l.strip()#.strip()去除字符串首尾的空格
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:#非空
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))#每个用户交互的item

                for item_id in item_ids:
                    user.append(user_id)
                    item.append(item_id)
                user_dict[user_id] = item_ids

        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        return (user, item), user_dict


    def statistic_cf(self):
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])


    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()#去重
        return kg_data


    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)

        sample_pos_items = []
        while True:
            if len(sample_pos_items) == n_sample_pos_items:
                break

            pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_item_id = pos_items[pos_item_idx]
            if pos_item_id not in sample_pos_items:
                sample_pos_items.append(pos_item_id)
        return sample_pos_items


    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items


    def generate_cf_batch(self, user_dict, batch_size):#生成协同过滤的批数据
        exist_users = user_dict.keys()
        if batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, batch_size)#无放回
        else:
            # batch_user = [random.choice(list(exist_users)) for _ in range(batch_size)]
            batch_user = np.random.choice(list(exist_users), batch_size, replace=True).tolist()#又放回

        batch_pos_item, batch_neg_item = [], []
        for u in batch_user:
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)#为用户u采样一个正样本
            batch_neg_item += self.sample_neg_items_for_u(user_dict, u, 1)#为用户u采样一个负样本

        batch_user = torch.LongTensor(batch_user)
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item


    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]#从知识图谱字典中获取与给定head实体相关的三元组
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]#选中三元组的尾实体
            relation = pos_triples[pos_triple_idx][1]#选中三元组的关系

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails


    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:#不在正样本三元组中，也不在已采样的负样本三元组中
                sample_neg_tails.append(tail)
        return sample_neg_tails


    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):#生成知识图谱的批数据
        exist_heads = kg_dict.keys()
        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            # batch_head = [random.choice(list(exist_heads)) for _ in range(batch_size)]
            batch_head = np.random.choice(list(exist_heads), batch_size, replace=True).tolist()

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            #对给定的头实体h，采样一个正样本的关系和尾实体
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)#为头实体h采样一个正样本
            batch_relation += relation
            batch_pos_tail += pos_tail

            #针对给定的头实体h和采样得到的关系，得到一个负样本的尾实体
            neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 1, highest_neg_idx)
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail
