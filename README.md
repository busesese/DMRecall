# 推荐算法召回框架V1.0.0
本框架是一个基于现有业务场景中经常用到的召回算法进行模块化封装，拥有统一的数据输入输出接口，将算法和评测等底层化更好的满足日常使用的需求，可以极大的提高算法的开发效率，将作为以后的一个底层框架支撑推荐及其他相关业务。

**框架特点：工程化快速，接口统一，接口友好度高，上手快，实验方便**

![image-20190910093424709](/Users/wenyi/Library/Application Support/typora-user-images/image-20190910093424709.png)



## 数据输入

统一的数据格式为pandas的DataFrame，对于实际场景中对于用户行为的召回，是一个常见的0-1场景即对用户历史喜欢的物品行为进行建模，构建相似度推荐算法。这里喜欢的定义需要根据实际的场景例如：以用户实际的点击表示用户的喜欢，用户的停留时间，购买等不同的定义输入数据需要用户自己构造。

构造数据的最终格式为：

| User_id | Item_id | Score | Dates      |
| ------- | ------- | ----- | ---------- |
| 1       | 1       | 1     | 2019-05-01 |
| 1       | 4       | 1     | 2019-05-02 |
| 2       | 3       | 1     | 2019-05-07 |

注：由于评分预测在实际场景中很少有实际真实评分数据，因此这里统一以0-1表示用户的喜好，因此score列全部为1。score列不是必要的一列，其他三列必须包含在数据里




## 快速上手
```python
from DMRecall.algorithm import ItemCF
from DMRecall.evaluation import measure
from DMRecall.tools import utils
import pandas as pd

# load data
df = pd.read_csv('seewo_edu.csv')
df.columns = ['user_id', 'item_id', 'score', 'dates']
train, test = utils.train_test_split(df, day=1)

# load model
model = ItemCF(train)

# model predict
train_dic = dict()
test_dic = dict()
groups = train.groupby('user_id')
for name, group in groups:
    train_dic[name] = groups.get_group(name)['item_id'].tolist()
groups = test.groupby('user_id')
for name, group in group:
    test_dic[name] = groups.get_group(name)['item_id'].tolist()

test_predict = model.recommend(train_dic, k=10)

# evaluate
rec = measure.recall(test_dic, test_predict)
pre = measure.precision(test_dic, test_predict, 10)
item_num = len(train['item_id'].unique())
cov = measure.coverage(test_predict, item_num)
print("Recall is %.3f, Precision is %.3f, Coverage is %.3f" % (rec, pre, cov))

```

## API

**DMRecall.alogrithm.ItemCF**(data):

data: dataframe input data for train model,must include ['user_id', 'item_id', 'dates'] columns

**ItemCF.predict**(items, k=10)

items: list of input item for predict,user recent item sequence

k: int predict top k result

return: list of predict result

