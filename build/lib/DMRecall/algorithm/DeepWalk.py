from DMRecall.baseclass.recall import Recall
from gensim.models.word2vec import Word2Vec
import numpy as np
from DMRecall.tools.graph import bulidGraph
from multiprocessing import cpu_count
import time
from concurrent.futures import ProcessPoolExecutor


class DeepWalk(Recall):
    """
    DeepWalk algorithm
    """
    def __init__(self, data, iters=4, depth=11, size=100, window=5, workers=3, negative=5, min_count=1, k=10):
        """
        init parmerters for model
        :param data:
        :param iters:
        :param depth: int sequence length of deepwalk
        :param size: int embedding size of item
        :param window: int
        :param workers: number of workers for train model
        :param negative: negative sample numbers
        :param min_count: the item
        :param k:
        """
        super(DeepWalk, self).__init__()
        self.data = data
        self.iters = iters
        self.depth = depth
        self.size = size
        self.window = window
        self.workers = workers
        self.negative = negative
        self.min_count = min_count
        self.k = k
        self.G = self.build_graph()
        self.vocab = self.G.nodes()
        self.sentences = self.processData()
        self.item_similarity = self.train()

    def build_graph(self):
        """
        build a graph using input data
        :return: graph G
        """
        self.data = self.data[['user_id', 'item_id', 'dates']]
        self.data['user_id'] = self.data['user_id'].apply(lambda x: str(x))
        self.data['item_id'] = self.data['item_id'].apply(lambda x: str(x))
        return bulidGraph(self.data)

    def random_walk(self, args):
        """
        Graph random walk with given args
        :param args: tuple include two int variables start node and walk depth like (1,5)
        :return: list random walk result
        """
        item, depth = args
        result = [str(item)]
        while depth > 0:
            neighbors = list(self.G[item].keys())
            if len(neighbors) == 0:
                break
            else:
                weights = [item['weight'] for item in self.G[item].values()]
                item = np.random.choice(neighbors, size=1, p=weights)[0]
                result.append(str(item))
                depth -= 1
        return result

    def generate_random_walk_args(self):
        args_list = []
        for i in range(self.iters):
            for node in self.vocab:
                args_list.append((node, self.depth))
        return args_list

    def multiple_processing_result(self, args_list):
        """
        generate result using multiple processing
        :param args_list: list of tuple [(1,5),(3,5),...]
        :return: list[list]
        """
        NUM_CPU = cpu_count()-1
        result = []
        with ProcessPoolExecutor(max_workers=NUM_CPU) as executor:
            for r in executor.map(self.random_walk, args_list, chunksize=20):
                if len(r) >= 8:
                    result.append(r)
        return result

    def processData(self):
        """
        generate deep walk result
        :return:
        """
        args_list = self.generate_random_walk_args()
        result = self.multiple_processing_result(args_list)
        return result

    def train(self):
        model = Word2Vec(self.sentences, size=self.size, window=self.window, workers=self.workers,
                         negative=self.negative, min_count=self.min_count)
        similarity_dic = dict()
        for item1 in self.vocab:
            try:
                sim = model.similar_by_word(item1, topn=self.k*10)

                similarity_dic[item1] = dict()
                for item, val in sim:
                    if item not in similarity_dic[item1]:
                        similarity_dic[item1][item] = val
                    else:
                        similarity_dic[item1][item] += val
            except:
                continue
        return similarity_dic

    def predict(self, items, k=10):
        """
        predict an given item sequence
        :param items: list of items  note: item must be str
        :param k: top k recommned item
        :return: list of top k recommend item
        """
        return super(DeepWalk, self).predict(items, k)

    def recommend(self, data_pre, k=10):
        """
        recommend result for given users
        :param data_pre: dict of user and history action items
        :param k: top k recommned item
        :return: dict of user top k recommend items
        """
        return super(DeepWalk, self).recommend(data_pre, k)


if __name__ == '__main__':
    import pandas as pd
    start = time.time()
    data = pd.read_table('../dataset/sample_data/sample.csv', sep='::', names=['user_id', 'item_id', 'dates'])
    data = data[['user_id', 'item_id', 'dates']]
    print(data.shape)
    i2v = DeepWalk(data)
    items = ['1193', '661', '914', '3048']
    print(i2v.predict(items, k=20))
    print(time.time() - start)