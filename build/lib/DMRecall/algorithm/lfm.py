# latent factor model
# @author: WenYi
# @time: 2019-08-28
# @Contact: wenyi@cvte.com


import os
from operator import itemgetter
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import datetime
import math
from multiprocessing import Manager, Pool, cpu_count
from DMRecall.tools.utils import train_test_split


class LFM():
    def __init__(self, train, test, F, negative_ratio, sigma):
        """
        Args：
            train: dataframe, trainset
            test: dataframe, testset
            F: int, 隐变量的数目
            negative_ratio: int, 负采样的比例
            sigma: float, 初始化隐语义矩阵时高斯分布的参数
        """
        self.train, self.test = self.get_data(train), self.get_data(test)
        self.F = F
        self.negative_ratio, self.sigma = negative_ratio, sigma
        self.all_users, self.all_items, self.item_popularity = self.helper(train)
        self.items_pool = np.array(list(self.item_popularity.keys()))

        p = np.array(list(self.item_popularity.values()))
        self.p = p / p.sum()  # 负采样的概率
        self.P, self.Q = self.build_matrix()

    def get_data(self, data):
        """将数据转化为{user:{item:rate}}字典
        """
        if data is None:
            return None
        uir = defaultdict(dict)
        for username, courseid, _ in data.itertuples(index=False):
            uir[username][courseid] = 1
        return uir

    def helper(self, train_df):
        """获取一些属性
        """
        all_items = train_df['item_id'].unique()
        all_users = train_df['user_id'].unique()
        item_popularity = train_df['item_id'].value_counts().to_dict()
        print('total user:', len(all_users), 'total item:', len(all_items), 'total data:', len(train_df))
        return all_users, all_items, item_popularity

    def Random_Negative_Sampling(self, items):
        """随机负采样，返回字典{items:score}，包含正负样本
        """
        ret = dict()
        for item in items:
            ret[item] = 1

        # 采集负样本
        n_negative = 0

        while n_negative < self.negative_ratio * len(items):
            negative_sample = np.random.choice(self.items_pool, p=self.p)
            if negative_sample in ret:
                continue
            ret[negative_sample] = 0
            n_negative += 1
            if len(ret) >= len(self.items_pool) - 50:
                break
        return ret

    def build_matrix(self):
        """建立隐语义矩阵
        """
        P = dict()
        for user in self.train.keys():
            P[user] = np.random.normal(0, self.sigma, size=(self.F))

        Q = dict()
        for movie in set(self.items_pool):
            Q[movie] = np.random.normal(0, self.sigma, size=(self.F))
        return P, Q

    def predict(self, user, item):
        """将预测结果经过 sigmoid 再输出，限制到 0-1，向量
        """
        pre = np.dot(self.P[user], self.Q[item])
        # logit = 1.0 / (1 + np.exp(-pre))
        # return logit
        return pre

    def get_sample(self):
        """生成所有负样本"""
        tic = time.time()
        sample = {}
        for user, items in self.train.items():
            sample[user] = self.Random_Negative_Sampling(items)
        print('负采样完成，耗时{:.2f}min'.format((time.time() - tic) / 60))
        return sample

    def sample_multi(self, user_list):
        """用于多线程生成负样本的任务"""
        sample = {}
        for user in user_list:
            sample[user] = self.Random_Negative_Sampling(self.train[user])
        return sample

    def generate_sample(self):
        """多线程生成所有负样本
        """
        tic = time.time()
        sample = {}
        n = 16  # 将用户分成多少份
        all = list(self.train.keys())
        slice = math.ceil(len(all) / n)
        user_list = []
        for i in range(n):
            start = i * slice
            end = min((i + 1) * slice, len(all))
            user = all[start:end]
            user_list.append(user)

        p = Pool(cpu_count())  # 使用多少个进程
        multi_res = [p.apply_async(self.sample_multi, (users,)) for users in user_list]
        p.close()
        p.join()
        for res in multi_res: sample.update(res.get())
        print('负采样完成，耗时{:.2f}min'.format((time.time() - tic) / 60))
        return sample

    def fit_sample(self, epochs, lr, reg, N, lr_decay, verbose=False):
        sample = self.get_sample()  # 没有使用多进程，使用后会导致结果变差，暂时不知道原因。
        # sample = self.generate_sample()  # 多进程负采样
        print('开始训练')
        for step in range(epochs):
            tic = time.time()
            loss = []
            for user in sample.keys():
                for item, rui in sample[user].items():
                    user_latent = self.P[user]
                    item_latent = self.Q[item]

                    loss_ = (rui - self.predict(user, item)) ** 2
                    loss.append(loss_ / 2)

                    err = rui - self.predict(user, item)

                    self.P[user] += lr * (err * item_latent - reg * user_latent)
                    self.Q[item] += lr * (err * user_latent - reg * item_latent)
            print('epoch{}\t loss {:.5f}\t {:.2f} min'.format(step, np.mean(loss), (time.time() - tic) / 60))
            if (step + 1) % 1 == 0 and verbose:
                self.evaluate(N)
            if (step + 1) % 5 == 0:
                lr *= lr_decay
                # # 打印推荐的结果
                # for user in list(self.test.keys())[:5]:
                #     self.showRecommendResult(user, 10)

    def recommend(self, data_pre, k=10):
        """产生推荐结果，并获取topN
        :param data_pre: dict for user and their history item sequence
        :param k: top k recommend result for each user
        :return: dict user and user recommend result
        """
        result = {}
        for user, items in data_pre.items():
            rank_user = {}
            for item in self.all_items:
                if item in items:
                    continue
                rank_user[item] = self.predict(user, item)
            recommend_list = [item for (item, score) in sorted(rank_user.items(), key=itemgetter(1), reverse=True)[:k]]
            result[user] = recommend_list
        return result

    def evaluate(self, N):
        """评价测试集中的top@N的precision recall coverage
        """
        hit = 0
        n_recall = 0
        n_precision = 0
        recommend_items = []
        # 评价模型时注意只给测试集用户推荐
        for user, items in self.test.items():
            tu = set(items.keys())
            recommend_list = set(self.recommend(user, N))

            recommend_items += list(recommend_list)

            hit += len(tu & recommend_list)
            n_recall += len(tu)
            n_precision += len(recommend_list)

        recall = hit / (n_recall * 1.0)
        precision = hit / (n_precision * 1.0)
        coverage = len(set(recommend_items)) / (len(self.all_items) * 1.0)
        print('precision=%.4f\t recall=%.4f\t coverage=%.4f' % (precision, recall, coverage))
        return recall, precision, coverage

if __name__ == "__main__":
    tic = time.time()

    print('开始导入文件.')
    current_path = os.path.abspath('..')
    data_path = os.path.join(current_path, 'dataset', 'ml-1m', 'ratings.dat')
    data = pd.read_csv(data_path, sep="::", names=['user', 'item', 'ratings', 'dates'])
    # 日期转为"%Y-%m-%d"
    data['dates'] = data['dates'].apply(lambda x: datetime.datetime.fromtimestamp(x).strftime("%Y-%m-%d %D-%M-%S")[:10])
    data['ratings'] = data['ratings'].apply(lambda x: 1 if x == 5 or x == 4 else 0)
    data = data[data['ratings'] == 1]
    print(data.head())
    print(data.columns)
    train, test = train_test_split(data, 300)
    print(train.shape, test.shape)


    negative_ratio = 4
    sigma = 0.1
    F = 80
    lr = 0.02
    lr_decay = 1
    reg = 0.015
    N = 50
    epochs = 15

    print('epochs={} negative_ratio={} sigma={} F={} lr={}*{} reg={} N={}'.format(epochs, negative_ratio, sigma, F, lr, lr_decay, reg, N))

    lfm = LFM(train, test, F, negative_ratio, sigma)

    lfm.fit_sample(epochs=epochs, lr=lr, reg=reg, N=N, lr_decay=lr_decay, verbose=True)

    # for user in np.random.choice(list(lfm.test.keys()), 5):
    #     lfm.showRecommendResult(user, 10)

    print('共耗时：{:.2f}min'.format((time.time() - tic) / 60))
