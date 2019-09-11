# -*- coding: utf-8 -*-
# build graph by networkx
# @author: WenYi
# @time:  2019-08-29
# @contact: wenyi@cvte.com

import networkx as nx


def bulidGraph(data):
    """
    bulid a graph
    :param data: DataFrame for user item history interaction must
    include [['user_id', 'item_id', 'dates']] three columns
    :return: G a graph
    """
    if set(['user_id', 'item_id', 'dates']) <= set(list(data.columns)) == False:
        raise Exception("Maybe input data is not correct, please check the input include user_id, item_id,"
                        "dates columns")

    data = data.sort_values(by=['user_id', 'dates'])
    groups = data.groupby('user_id')
    dic = {}

    # calculate the weight between items
    for name, group in groups:
        seq = groups.get_group(name)['item_id'].tolist()
        for j in range(len(seq) - 1):
            key = seq[j]
            if key not in dic:
                dic[key] = dict()
            key1 = seq[j+1]
            if key1 not in dic[key]:
                dic[key][key1] = 1
            else:
                dic[key][key1] += 1

    # normalize weight and construct data format:[(item1,item2, weight)...]
    result = []
    for key, val in dic.items():
        sums = 0
        for key1, val1 in val.items():
            sums += val1
        for key1, val1 in val.items():
            result.append((key, key1, val1 / sums))

    G = nx.DiGraph()
    G.add_weighted_edges_from(result)
    return G


if __name__ == '__main__':
    import pandas as pd
    data = pd.read_table('../dataset/ml-1m/ratings.dat', sep='::', names=['user_id', 'item_id', 'ratings', 'dates'])
    data = data[['user_id', 'item_id', 'dates']]
    print(data.head())
    G = bulidGraph(data)
    print(list(G.nodes))