import torch
from torch import nn
from torch.nn import functional as F


def _L2_loss_mean(x):#计算L2范数的均值
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class KG_free(nn.Module):#KG_free类继承自nn.Module类

    def __init__(self, args, n_users, n_items):

        super(KG_free, self).__init__()#调用父类的初始化方法

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = args.embed_dim#嵌入维度
        self.l2loss_lambda = args.l2loss_lambda

        self.user_embed = nn.Embedding(self.n_users, self.embed_dim)#用户嵌入
        self.item_embed = nn.Embedding(self.n_items, self.embed_dim)#物品嵌入

        nn.init.xavier_uniform_(self.user_embed.weight)#对用户嵌入层的权重进行初始化，xaver_uniform_是均匀分布
        nn.init.xavier_uniform_(self.item_embed.weight)


    def calc_score(self, user_ids, item_ids):#计算评分
        """
        user_ids:   (n_users)
        item_ids:   (n_items)
        """
        user_embed = self.user_embed(user_ids)                              # (n_users, embed_dim)
        item_embed = self.item_embed(item_ids)                              # (n_items, embed_dim)
        cf_score = torch.matmul(user_embed, item_embed.transpose(0, 1))     # (n_users, n_items)
        #计算协调过滤评分，即用户与物品的内积
        #torch.matmul是矩阵乘法，transpose是转置
        return cf_score


    def calc_loss(self, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (batch_size)
        item_pos_ids:   (batch_size)
        item_neg_ids:   (batch_size)
        """
        user_embed = self.user_embed(user_ids)                              # (batch_size, embed_dim)
        item_pos_embed = self.item_embed(item_pos_ids)                      # (batch_size, embed_dim)
        item_neg_embed = self.item_embed(item_neg_ids)                      # (batch_size, embed_dim)
        #矩阵分解
        pos_score = torch.sum(user_embed * item_pos_embed, dim=1)           # (batch_size)
        neg_score = torch.sum(user_embed * item_neg_embed, dim=1)           # (batch_size)

        # BPR Loss
        cf_loss = (-1.0) * F.logsigmoid(pos_score - neg_score)
        cf_loss = torch.mean(cf_loss)

        # L2 Loss
        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_embed) + _L2_loss_mean(item_neg_embed)
        #三部分组成，用户嵌入，正样本嵌入，负样本嵌入
        loss = cf_loss + self.l2loss_lambda * l2_loss
        return loss


    def forward(self, *input, is_train):#前向传播
        if is_train:#训练模式
            return self.calc_loss(*input)#返回损失值
        else:#推断模式
            return self.calc_score(*input)#返回评分


