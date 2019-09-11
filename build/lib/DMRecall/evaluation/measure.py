# 常用测试指标：命中率，准确率，召回率，覆盖率，流行度
# author: WenYi
# time: 2019-09-05


def hits(origin, res):
    """
    calculate the hit number of recommend result
    :param origin: dict {user:[item,...]} the user click or behave items
    :param res: dict {user:[item,...]} the algorithm recommend result
    :return: int the number of res result in origin
    """
    hitCount = {}
    for user, items in origin.items():
        rs_items = res[user]
        hitCount[user] = len(set(items).intersection(set(rs_items)))
    return hitCount


def precision(origin, res, N):
    """
    the precision of recommend result
    :param origin: dict
    :param res: dict
    :param N: top N precision
    :return: float
    """
    sums = 0
    hit = hits(origin, res)
    for user, hitcount in hit.items():
        sums += hitcount
    return float(sums)/(len(hit) * N)


def recall(origin, res):
    """
    recall of recommend result
    :param origin: dict
    :param res: dict
    :return: float
    """
    hit = hits(origin, res)
    recallList = [float(hit[user])/len(origin[user]) for user in hit]
    return sum(recallList)/float(len(recallList))


def coverage(res, item_num):
    """
    the coverage percent of recommend result
    :param res: recommend result
    :param item_num: item num of train result
    :return: float
    """
    res_items = set()
    for user, items in res.items():
        res_items.update(items)
    return float(len(res_items))/item_num


def maps(origin, res, N):
    """
    :param origin:
    :param res:
    :param N:
    :return:
    """
    sum_prec = 0
    for user in res:
        hits = 0
        precision = 0
        for n, item in enumerate(res[user]):
            if origin[user].has_key(item):
                hits += 1.0
                precision += hits / (n + 1.0)
        sum_prec += precision / (min(len(origin[user]), N) + 0.0)
    return sum_prec / (len(res))


    # def ndcg(self):