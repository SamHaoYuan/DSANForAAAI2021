# coding: utf-8
'''
Author: Chence Shi
Contact: chenceshi@pku.edu.cn
'''

import tensorflow as tf
import argparse
import numpy as np
import sys
import time
import math

from utils import *
from model import *
from sampler import *

parser = argparse.ArgumentParser(description='Sequential or session-based recommendation')
parser.add_argument('--model', type=str, default='fpmc', help='model: fossil/fpmc. (default: fpmc)')
parser.add_argument('--batch_size', type=int, default=128, help='batch size (default: 128)')
parser.add_argument('--seq_len', type=int, default=20, help='max sequence length (default: 20)')
parser.add_argument('--l2_reg', type=float, default=0.0, help='regularization scale (default: 0.0)')
parser.add_argument('--lr', type=float, default=0.005, help='initial learning rate for Adam (default: 0.01)')
parser.add_argument('--lr_decay', type=float, default=0, help='learning rate decay rate (default: 0.5)')
parser.add_argument('--emsize', type=int, default=100, help='dimension of item embedding (default: 100)')
parser.add_argument('--neg_size', type=int, default=1, help='size of negative samples (default: 1)')
parser.add_argument('--worker', type=int, default=1, help='number of sampling workers (default: 10)')
parser.add_argument('--seed', type=int, default=1111, help='random seed (default: 1111)')
parser.add_argument('--data', type=str, default='retailrocket', help='datasets set name (default: gowalla)')
parser.add_argument('--log_interval', type=int, default=1e2, help='log interval (default: 1e2)')
parser.add_argument('--eval_interval', type=int, default=1e2, help='eval/test interval (default: 1e3)')
parser.add_argument('--optim', type=str, default='adam', help='optimizer: sgd/adam (default: adam)')
parser.add_argument('--warm_up', type=int, default=0, help='warm up step (default: 0)')

# ****************************** unique arguments for FOSSIL *******************************************************
parser.add_argument('--alpha', type=float, default=0.2, help='alpha (default: 0.2)')
parser.add_argument('--order', type=int, default=1, help='order of Fossil (default: 1)')

# ****************************** unique arguments for FPMC *******************************************************
# None


args = parser.parse_args()
tf.set_random_seed(args.seed)

train_data, val_data, test_data, n_items, n_users = data_generator(args)

train_sampler = Sampler(
    data=train_data,
    n_items=n_items,
    n_users=n_users,
    batch_size=args.batch_size,
    max_len=args.seq_len,
    neg_size=args.neg_size,
    n_workers=args.worker,
    neg_method='rand')

val_data = prepare_eval_test(val_data, batch_size=100, max_test_len=20)
test_data = prepare_eval_test(test_data, batch_size=100, max_test_len=20)

checkpoint_dir = '_'.join(['save', args.data, args.model, str(args.lr), str(args.l2_reg), str(args.emsize)])

print(args)
print('#Item: ', n_items)
print('#User: ', n_users)

model_dict = {'fossil': FOSSIL, 'fpmc': FPMC}
assert args.model in ['fossil', 'fpmc']

model = model_dict[args.model](args, n_items, n_users)

lr = args.lr


def evaluate(source, sess):
    total_recall_20 = 0.0
    total_ndcg_20 = 0.0
    total_mrr_20 = 0.0

    total_recall_10 = 0.0
    total_ndcg_10 = 0.0
    total_mrr_10 = 0.0

    total_recall_5 = 0.0
    total_ndcg_5 = 0.0
    total_mrr_5 = 0.0

    count = 0.0
    for batch in source:
        feed_dict = {model.inp: batch[1], model.user: batch[0], model.pos: batch[2]}
        recall_20, mrr_20, ndcg_20, recall_10, mrr_10, ndcg_10, recall_5, mrr_5, ndcg_5 = sess.run(
            [model.recall_20, model.mrr_20, model.ndcg_20,
             model.recall_10, model.mrr_10, model.ndcg_10,
             model.recall_5, model.mrr_5, model.ndcg_5], feed_dict=feed_dict)
        count += len(batch[0])

        total_recall_20 += recall_20
        total_ndcg_20 += ndcg_20
        total_mrr_20 += mrr_20

        total_recall_10 += recall_10
        total_ndcg_10 += ndcg_10
        total_mrr_10 += mrr_10

        total_recall_5 += recall_5
        total_ndcg_5 += ndcg_5
        total_mrr_5 += mrr_5

    val_recall_20 = total_recall_20 / count
    val_ndcg_20 = total_ndcg_20 / count
    val_mrr_20 = total_mrr_20 / count

    val_recall_10 = total_recall_10 / count
    val_ndcg_10 = total_ndcg_10 / count
    val_mrr_10 = total_mrr_10 / count

    val_recall_5 = total_recall_5 / count
    val_ndcg_5 = total_ndcg_5 / count
    val_mrr_5 = total_mrr_5 / count

    return [val_recall_20, val_mrr_20, val_ndcg_20,
            val_recall_10, val_mrr_10, val_ndcg_10,
            val_recall_5, val_mrr_5, val_ndcg_5]


def main():
    global lr
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    all_val_recall = [-1]
    early_stop_cn = 0
    step_count = 0
    train_loss_l = 0.
    start_time = time.time()
    print('Start training...')
    try:
        while True:
            cur_batch = train_sampler.next_batch()
            inp = np.array(cur_batch[1])
            feed_dict = {model.inp: inp, model.lr: lr}
            feed_dict[model.pos] = np.array(cur_batch[2])
            feed_dict[model.neg] = np.array(cur_batch[3])
            feed_dict[model.user] = np.array(cur_batch[0])

            _, train_loss = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
            train_loss_l += train_loss
            step_count += 1

            if step_count % args.log_interval == 0:
                cur_loss = train_loss_l / args.log_interval
                elapsed = time.time() - start_time
                print('| Totol step {:10d} | lr {:02.5f} | ms/batch {:5.2f} | loss {:5.3f}'.format(
                    step_count, lr, elapsed * 1000 / args.log_interval, cur_loss))
                sys.stdout.flush()
                train_loss_l = 0.
                start_time = time.time()

            if step_count % args.eval_interval == 0 and step_count > args.warm_up:

                val_recall_20, val_mrr_20, val_ndcg_20, val_recall_10, val_mrr_10, val_ndcg_10, val_recall_5, val_mrr_5, val_ndcg_5 = evaluate(
                    val_data, sess)

                all_val_recall.append(val_recall_20)
                print('-' * 90)
                print(
                    '| End of step {:10d} \n| valid recall@20 {:8.5f} | valid mrr@20 {:8.5f} | valid ndcg@20 {:8.5f}'.format(
                        step_count, val_recall_20, val_mrr_20, val_ndcg_20))
                print(
                    '| valid recall@10 {:8.5f} | valid mrr@10 {:8.5f} | valid ndcg@10 {:8.5f}'.format(
                        val_recall_10, val_mrr_10, val_ndcg_10))
                print(
                    '| valid recall@5 {:8.5f} | valid mrr@5 {:8.5f} | valid ndcg@5 {:8.5f}'.format(
                        val_recall_5, val_mrr_5, val_ndcg_5))
                print('=' * 90)
                sys.stdout.flush()

                if all_val_recall[-1] <= all_val_recall[-2]:
                    lr = lr * args.lr_decay
                    lr = max(lr, 1e-6)
                    early_stop_cn += 1
                else:
                    early_stop_cn = 0
                    model.saver.save(sess, checkpoint_dir + '/model.ckpt')
                if early_stop_cn == 3:
                    print('Validation recall decreases in three consecutive epochs. Stop Training!')
                    sys.stdout.flush()
                    break
                start_time = time.time()
    except Exception as e:
        print(str(e))
        train_sampler.close()
        exit(1)
    train_sampler.close()
    print('Done')

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        model.saver.restore(sess, '{}/{}'.format(checkpoint_dir, 'model.ckpt'))
        print('Restore model successfully')
    else:
        print('Restore model failed!!!!!')
    test_recall_20, test_mrr_20, test_ndcg_20, test_recall_10, test_mrr_10, test_ndcg_10, test_recall_5, test_mrr_5, test_ndcg_5 = evaluate(
        test_data, sess)
    print('-' * 90)
    print('test recall@20 {:8.5f} | test mrr@20 {:8.5f} | test ndcg@20 {:8.5f}'.format(
        test_recall_20, test_mrr_20, test_ndcg_20))
    print('test recall@10 {:8.5f} | test mrr@10 {:8.5f} | test ndcg@10 {:8.5f}'.format(
        test_recall_10, test_mrr_10, test_ndcg_10))
    print('test recall@5 {:8.5f} | test mrr@5 {:8.5f} | test ndcg@5 {:8.5f}'.format(
        test_recall_5, test_mrr_5, test_ndcg_5))
    print('=' * 90)


if __name__ == '__main__':
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    main()
