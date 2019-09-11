from DMRecall.baseclass.recall import Recall
import numpy as np
from gensim.models.word2vec import Word2Vec


class Item2Vec(Recall):
    """
    Item2vec algorithm based on《Item2Vec: Neural Item Embedding for Collaborative Filtering》
    """
    def __init__(self, data, size=100, window=5, workers=3, negative=5, min_count=2, k=10):
        """
        init parameters
        :param data: user history item sequence list like [[item1, item2,...],[item3,item2...]...]
        """
        self.data = data
        self.size = size
        self.window = window
        self.workers = workers
        self.negative = negative
        self.min_count = min_count
        self.k = k
        self.sentences, self.vocab = self.processData()
        self.item_similarity = self.train()

    def processData(self):
        """
        process data before train
        :return:
        """
        sentences = []
        # filter item behave less than self.min_count
        support = self.data.groupby('item_id').size()
        self.data = self.data[np.in1d(self.data['item_id'], support[support >= self.min_count].index)]
        vocab = list(map(str, self.data['item_id'].unique().tolist()))
        self.data = self.data.sort_values(by=['user_id', 'dates'])
        groups = self.data.groupby('user_id')
        for name, group in groups:
            seq = groups.get_group(name)['item_id'].tolist()
            seq = list(map(str, seq))
            sentences.append(seq)
        return sentences, vocab

    def train(self):
        """
        train the data and calculate the item similarity
        :return:
        """
        model = Word2Vec(self.sentences, size=self.size, window=self.window, workers=self.workers,
                         negative=self.negative, min_count=self.min_count)
        similarity_dic = dict()
        for item1 in self.vocab:
            similarity_dic[item1] = dict()
            sim = model.similar_by_word(item1, topn=self.k*10)
            for item, val in sim:
                if item not in similarity_dic[item1]:
                    similarity_dic[item1][item] = val
                else:
                    similarity_dic[item1][item] += val
        return similarity_dic

    def predict(self, items, k=10):
        """
        predict an given item sequence
        :param items: list of items  note: item must be str
        :param k: top k recommned item
        :return: list of top k recommend item
        """
        return super(Item2Vec, self).predict(items, k)

    def recommend(self, data_pre, k=10):
        """
        recommend result for given users
        :param data_pre: dict of user and history action items
        :param k: top k recommned item
        :return: dict of user top k recommend items
        """
        return super(Item2Vec, self).recommend(data_pre, k)

if __name__ == '__main__':
    import pandas as pd
    data = pd.read_table('../dataset/ml-1m/ratings.dat', sep='::', names=['user_id', 'item_id', 'ratings', 'timestamp'])
    data = data[['user_id', 'item_id', 'dates']]
    print(data.shape)
    i2v = Item2Vec(data)
    items = ['1193', '661', '914', '3048']
    print(i2v.predict(items))