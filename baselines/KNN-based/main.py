import pickle
import numpy as np
from SKNN import SKNN
from STAN import STAN
import math
import time


train_data = pickle.load(open('../../data/retailrocket/train.txt', 'rb'))
# train_data = pickle.load(open('../../data/diginetica/train.txt', 'rb'))

train_id = train_data[0]
train_session = train_data[1]
train_timestamp = train_data[3]
train_predict = train_data[2]

for i, s in enumerate(train_session):
    train_session[i] += [train_predict[i]]

test_data = pickle.load(open('../../data/retailrocket/test.txt', 'rb'))
# test_data = pickle.load(open('../../data/diginetica/test.txt', 'rb'))

test_id = test_data[0]
test_session = test_data[1]
test_timestamp = test_data[3]
test_predict = test_data[2]

model = SKNN(session_id=train_id, session=train_session, session_timestamp=train_timestamp, sample_size=0, k=500)
# model = STAN(session_id=train_id, session=train_session, session_timestamp=train_timestamp,
#              sample_size=0, k=500, factor1=True, l1=3.54, factor2=True, l2=20*24*3600, factor3=True, l3=3.54)

testing_size = len(test_session)

R_5 = 0
R_10 = 0
R_20 = 0

MRR_5 = 0
MRR_10 = 0
MRR_20 = 0

NDCG_5 = 0
NDCG_10 = 0
NDCG_20 = 0
for i in range(testing_size):
    if i % 1000 == 0:
        print("%d/%d" % (i, testing_size))

    score = model.predict(session_id=test_id[i], session_items=test_session[i], session_timestamp=test_timestamp[i],
                          k=20)
    items = [x[0] for x in score]
    if test_predict[i] in items:
        rank = items.index(test_predict[i]) + 1
        MRR_20 += 1 / rank
        R_20 += 1
        NDCG_20 += 1 / math.log(rank + 1, 2)

        if rank <= 5:
            MRR_5 += 1 / rank
            R_5 += 1
            NDCG_5 += 1 / math.log(rank + 1, 2)

        if rank <= 10:
            MRR_10 += 1 / rank
            R_10 += 1
            NDCG_10 += 1 / math.log(rank + 1, 2)

MRR_5 = MRR_5 / testing_size
MRR_10 = MRR_10 / testing_size
MRR_20 = MRR_20 / testing_size
R_5 = R_5 / testing_size
R_10 = R_10 / testing_size
R_20 = R_20 / testing_size
NDCG_5 = NDCG_5 / testing_size
NDCG_10 = NDCG_10 / testing_size
NDCG_20 = NDCG_20 / testing_size

print("MRR@5: %f" % MRR_5)
print("MRR@10: %f" % MRR_10)
print("MRR@20: %f" % MRR_20)
print("R@5: %f" % R_5)
print("R@10: %f" % R_10)
print("R@20: %f" % R_20)
print("NDCG@5: %f" % NDCG_5)
print("NDCG@10: %f" % NDCG_10)
print("NDCG@20: %f" % NDCG_20)
print("training size: %d" % len(train_session))
print("testing size: %d" % testing_size)
