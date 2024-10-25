import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from parser1.parser_KG_free import *
from model.KG_free import KG_free
from utils.log_helper import *
from utils.metrics import *
from utils.model_helper import *
from data_loader.loader_KG_free import DataLoader


def evaluate(model, dataloader, Ks, device):
    test_batch_size = dataloader.test_batch_size
    train_user_dict = dataloader.train_user_dict
    test_user_dict = dataloader.test_user_dict

    model.eval()#模型设置为评估模式，不启用BatchNormalization和Dropout

    user_ids = list(test_user_dict.keys())#获取测试集中的所有用户id
    user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]#将用户id分批
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]#每个批次转换为longtensor

    n_items = dataloader.n_items#物品数量
    item_ids = torch.arange(n_items, dtype=torch.long).to(device)#包含所有物品id的tensor

    cf_scores = []#存储所有批次的推荐评分
    metric_names = ['precision', 'recall', 'ndcg']#评价指标：精确率，召回率，ndcg
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}#初始化一个字典，存储不同k下每个指标的值

    #创建进度条
    with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
        for batch_user_ids in user_ids_batches:
            batch_user_ids = batch_user_ids.to(device)

            with torch.no_grad():#评估模式下不计算梯度
                batch_scores = model(batch_user_ids, item_ids, is_train=False)       # (n_batch_users, n_items)
                
            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks, device)
            #计算评估指标

            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])
            pbar.update(1)#更新进度条，完成一个批次的评估

    cf_scores = np.concatenate(cf_scores, axis=0)#将所有批次的评分在垂直方向拼接起来，形成一个二维数组
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    return cf_scores, metrics_dict


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)#为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(args.seed)#为所有GPU设置种子，以使得结果是确定的

    log_save_id = create_log_id(args.save_dir)#创建一个日志id
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)#配置日志记录
    logging.info(args)#将传入的参数打印到日志中

    # GPU / CPU
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
    logging.info(f'Running on {str(device)}')

    # load data
    data = DataLoader(args, logging)
    print("data loaded")
    # construct model & optimizer
    model = KG_free(args, data.n_users, data.n_items)
    if args.use_pretrain == 1:
        model = load_model(model, args.pretrain_model_path)

    model.to(device)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # initialize metrics
    best_epoch = -1#最佳epoch
    best_recall = 0#最佳召回率

    Ks = eval(args.Ks)  # eval() Function for str->list
    k_min = min(Ks)
    k_max = max(Ks)

    epoch_list = []
    metrics_list = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in Ks}

    # train model
    for epoch in range(1, args.n_epoch + 1):
        model.train()

        # train cf
        time1 = time()
        total_loss = 0
        n_batch = data.n_cf_train // data.train_batch_size + 1

        for iter in range(1, n_batch + 1):
            time2 = time()
            batch_user, batch_pos_item, batch_neg_item = data.generate_cf_batch(data.train_user_dict, data.train_batch_size)
            batch_user = batch_user.to(device)
            batch_pos_item = batch_pos_item.to(device)
            batch_neg_item = batch_neg_item.to(device)
            batch_loss = model(batch_user, batch_pos_item, batch_neg_item, is_train=True)

            if np.isnan(batch_loss.cpu().detach().numpy()):#判断loss是否为nan
                logging.info('ERROR: Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter, n_batch))
                sys.exit()

            batch_loss.backward()#反向传播，计算梯度
            optimizer.step()#更新参数
            optimizer.zero_grad()#梯度清零
            total_loss += batch_loss.item()#累加loss

            if (iter % args.print_every) == 0:#每隔一定批次打印一次
                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_batch, time() - time2, batch_loss.item(), total_loss / iter))
        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_batch, time() - time1, total_loss / n_batch))

        # evaluate cf
        if (epoch % args.evaluate_every) == 0 or epoch == args.n_epoch:#每隔一定epoch评估一次/最后一次评估
            time3 = time()#用于计算评估时间
            _, metrics_dict = evaluate(model, data, Ks, device)
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
                epoch, time() - time3, metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))

            epoch_list.append(epoch)
            for k in Ks:
                for m in ['precision', 'recall', 'ndcg']:
                    metrics_list[k][m].append(metrics_dict[k][m])
            best_recall, should_stop = early_stopping(metrics_list[k_min]['recall'], args.stopping_steps)

            if should_stop:
                break

            if metrics_list[k_min]['recall'].index(best_recall) == len(epoch_list) - 1:#如果当前召回率最大值对应的epoch是最后一个epoch
                save_model(model, args.save_dir, log_save_id, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch#更新最佳epoch

    # save metrics
    metrics_df = [epoch_list]
    metrics_cols = ['epoch_idx']
    for k in Ks:
        for m in ['precision', 'recall', 'ndcg']:
            metrics_df.append(metrics_list[k][m])
            metrics_cols.append('{}@{}'.format(m, k))
    metrics_df = pd.DataFrame(metrics_df).transpose()
    metrics_df.columns = metrics_cols
    metrics_df.to_csv(args.save_dir + '/metrics{:d}.tsv'.format(log_save_id), sep='\t', index=False)
    print(args.save_dir + '/metrics{:d}.tsv'.format(log_save_id))

    # print best metrics
    best_metrics = metrics_df.loc[metrics_df['epoch_idx'] == best_epoch].iloc[0].to_dict()
    logging.info('Best CF Evaluation: Epoch {:04d} | Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        int(best_metrics['epoch_idx']), best_metrics['precision@{}'.format(k_min)], best_metrics['precision@{}'.format(k_max)], best_metrics['recall@{}'.format(k_min)], best_metrics['recall@{}'.format(k_max)], best_metrics['ndcg@{}'.format(k_min)], best_metrics['ndcg@{}'.format(k_max)]))


def predict(args):
    # GPU / CPU
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

    # load data
    data = DataLoader(args, logging)

    # load model
    model = KG_free(args, data.n_users, data.n_items)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)

    # predict
    Ks = eval(args.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    cf_scores, metrics_dict = evaluate(model, data, Ks, device)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    print('CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))


if __name__ == '__main__':
    args = parse_args()#参数解析
    train(args)
    # predict(args)
