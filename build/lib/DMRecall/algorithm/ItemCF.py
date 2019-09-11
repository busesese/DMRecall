from DMRecall.baseclass.recall import Recall
import numpy as np


class ItemCF(Recall):
    """
    ItemCF algorithm inherit the base class Recall
    """
    def __init__(self, data):
        super(ItemCF, self).__init__()
        self.data = data
        self.item_similarity = self.train()

    def train(self):
        """
        get the item user history dict and caculate the similarity between items
        :return:
        """
        # get the item user history dict
        item_to_users = dict()
        for index, rows in self.data.iterrows():
            uid = rows['user_id']
            item = rows['item_id']
            if item not in item_to_users:
                item_to_users[item] = dict()
                if uid not in item_to_users[item]:
                    item_to_users[item][uid] = 1

        # caculate the item similarity
        item_similarity = dict()
        for item1 in item_to_users:
            item_similarity[item1] = dict()
            users1 = set(item_to_users[item1].keys())
            for item2 in item_to_users:
                if item2 != item1:
                    users2 = set(item_to_users[item2].keys())
                    sim = len(users1 & users2)/np.sqrt((len(users1)*len(users2)))
                    if item2 not in item_similarity[item1]:
                        item_similarity[item1][item2] = sim
                    else:
                        item_similarity[item1][item2] += sim
        return item_similarity

    def predict(self, items, k=10):
        """
        predict result for a given user
        :param items: list, user recent behavior item list
        :param k: predict top k result
        :return: dict
        """
        if isinstance(items, list):
            return super(ItemCF, self).predict(items, k)
        else:
            raise TypeError(" parameter items must be a list")

    def recommend(self, data_pre, k=10):
        """
        given a user and recent behave item sequence
        :param data_pre:
        :param k:
        :return:
        """
        return super(ItemCF, self).recommend(data_pre, k)



if __name__ == '__main__':
    import pandas as pd
    data = pd.read_table('../dataset/ml-1m/ratings.dat', sep='::', names=['user_id', 'item_id', 'ratings', 'timestamp'])
    data = data[['user_id', 'item_id', 'timestamp']]
    print(data.head())
    items = [1193, 661, 914, 3048]
    # [3408, 2355, 1197, 1287, 2804, 594, 919, 595, 938, 2398]
    icf = ItemCF(data)
    print(icf.predict(items, k=20))