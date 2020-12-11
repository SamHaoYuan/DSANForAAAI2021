import pickle
import csv

train_data = pickle.load(open('../../data/retailrocket/train.txt', 'rb'))
train_id = train_data[0]
train_session = train_data[1]
train_timestamp = train_data[2]
train_predict = train_data[3]

test_data = pickle.load(open('../../data/retailrocket/test.txt', 'rb'))
test_id = test_data[0]
test_session = test_data[1]
test_timestamp = test_data[2]
test_predict = test_data[3]

sid = 0

iid = -1
train_seq_id = []
train_seq_timestamp = []
train_seq = []
for i in range(len(train_id)):
    if iid != train_id[i]:
        train_seq += [train_session[i] + [train_predict[i]]]
        iid = train_id[i]
        train_seq_id += [sid]
        sid += 1
        train_seq_timestamp += [train_timestamp[i]]

iid = -1
test_seq = []
test_seq_id = []
test_seq_timestamp = []
for i in range(len(test_id)):
    if iid != test_id[i]:
        test_seq += [test_session[i] + [test_predict[i]]]
        iid = test_id[i]
        test_seq_id += [sid]
        sid += 1
        test_seq_timestamp += [test_timestamp[i]]

cnt = 0
for s in train_seq + test_seq:
    cnt += len(s)
print("clicks: %d" % cnt)

with open('datasets/retailrocket/train.txt', 'w', newline='') as f:
    file = csv.writer(f, delimiter='\t')
    file.writerow(['UserId', 'ItemId', 'Time'])
    for i in range(len(train_seq_id)):
        for j in range(len(train_seq[i])):
            file.writerow([train_seq_id[i], train_seq[i][j], train_seq_timestamp[i]])

with open('datasets/retailrocket/valid.txt', 'w', newline='') as f:
    file = csv.writer(f, delimiter='\t')
    file.writerow(['UserId', 'ItemId', 'Time'])
    for i in range(len(test_seq_id)):
        for j in range(len(test_seq[i])):
            file.writerow([test_seq_id[i], test_seq[i][j], test_seq_timestamp[i]])

with open('datasets/retailrocket/test.txt', 'w', newline='') as f:
    file = csv.writer(f, delimiter='\t')
    file.writerow(['UserId', 'ItemId', 'Time'])
    for i in range(len(test_seq_id)):
        for j in range(len(test_seq[i])):
            file.writerow([test_seq_id[i], test_seq[i][j], test_seq_timestamp[i]])
