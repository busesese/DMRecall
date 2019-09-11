# some tools function for this algorithm
# author: WenYi
# time: 2019-09-04

import datetime
import numpy as np


def train_test_split(data, day, min_count=1):
    """
    对数据集进行时间划分，其中data中必须有一列dates其格式为"%Y-%m-%d"
    :param data: 需要划分的数据集
    :param day: 测试集取的时间（距最后一天的前多少天为测试集）
    :param min_count: default=1, 过滤掉出现次数少于多少次的数据
    :return: datafrmae， train, test
    """
    if set(['user_id', 'item_id', 'dates']) <= set(list(data.columns)) == True:
        raise Exception("Maybe input data is not correct, please check the input include user_id, item_id,"
                        "dates columns")

    test_date = data['dates'].max()
    split_date = (datetime.datetime.strptime(test_date, "%Y-%m-%d") - datetime.timedelta(days=day)).strftime("%Y-%m-%d")
    train = data[data['dates'] <= split_date]
    test = data[data['dates'] > split_date]

    # 去重
    if 'times' in data.columns:
        train = train.sort_values(by=['user_id', 'dates', 'times'])
    else:
        train = train.sort_values(by=['user_id', 'dates'])
    train = train.drop_duplicates(subset=['user_id', 'item_id'])
    test = test.drop_duplicates(subset=['user_id', 'item_id'])

    # 过滤掉train中出现次数少于min_count的course
    if min_count != 1:
        course_supports = train.groupby('item_id').size()
        train = train[np.in1d(train['item_id'], course_supports[course_supports >= min_count].index)]

    if day == 0:
        return train, test

    # 过滤掉test在train中没有出现的user和item
    test = test[test['user_id'].isin(train['user_id'].unique())]
    test = test[test['item_id'].isin(train['item_id'].unique())]
    return train, test


if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv('../dataset/seewo-edu/user_history_click_course_1.csv')
    data = data[['username', 'courseid', 'dates']]
    data.columns = ['user_id', 'item_id', 'dates']

    train, test = train_test_split(data, day=7, min_count=1)
    print(train.shape, test.shape)


