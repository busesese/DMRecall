# 加载数据集

import os
import pandas as pd
from DMRecall.tools.utils import train_test_split

# def movielen_data():
#     root_path = os.path.dirname(__file__)
#     data_path = os.path.join(root_path, 'ml-1m/ratings.dat')
#     data = pd.read_table(data_path, sep='::', names=['user_id', 'item_id', 'ratings', 'dates'])
#     data = data[['user_id', 'item_id', 'dates']]
#     return data
#
#
# def seewo_data():
#     root_path = os.path.dirname(__file__)
#
#     data_path = os.path.join(root_path, 'seewo-edu/user_history_click_course_1.csv')
#     print(data_path)
#     data = pd.read_csv(data_path)
#     data = data[['username', 'courseid', 'dates']]
#     data.columns = ['user_id', 'item_id', 'dates']
#     return data


def mini_data():
    root_path = os.path.dirname(__file__)
    data_path = os.path.join(root_path, 'sample_data/sample.csv')
    print(data_path)
    data = pd.read_csv(data_path)
    data.columns = ['user_id', 'item_id', 'dates']
    return data


if __name__ == "__main__":
    df = mini_data()
    train, test = train_test_split(df, day=5)
    print(df.shape)
    print(train.shape, test.shape)