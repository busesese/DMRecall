# base class for recall algorithm
# author: WenYi
# time: 2019-08-20


class Recall(object):
    """
    the base class for recall algorithm, other algorithm should inherit this
    class and rewrite some methods
    """
    def __init__(self, **kwargs):
        """
        :param data: input DataFrame include ['user_id', 'item_id', 'timestamp']
        """
        pass

    def processData(self):
        """
        process data before train to get the right data format
        :return:
        """
        pass

    def train(self):
        """
        train the data
        :return:
        """
        pass

    def predict(self, items, k=10):
        """
        predict result for a given user
        :param user: str, user id
        :param items: list, user recent behavior item list
        :param k: predict top k result
        :return: dict
        """
        result = dict()
        if isinstance(items, list):
            for item in items:
                if item in self.item_similarity:
                    for i, val in self.item_similarity[item].items():
                        if i not in items:
                            if i not in result:
                                result[i] = val
                            else:
                                result[i] += val
            return [i for i, val in sorted(result.items(), key=lambda x: x[1], reverse=True)[:k]]
        else:
            raise TypeError("Input parameter type is not list")

    def recommend(self, data_pre, k=10):
        """
        recommend result for given users
        :param data_pre: dict, include user id and user recent behavior item
        :param k: int predict top k result
        :return: dict, key = user and value = user result
        """
        result = dict()
        for uid, item_list in data_pre.items():
            pred = self.predict(item_list, k)
            result[uid] = pred
        return result