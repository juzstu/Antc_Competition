# -*- coding：utf-8 -*-
# Author: Studog
# Time: 2018/6/13 23:20

import pandas as pd
import lightgbm as lgb
import numpy as np
import time
from sklearn import metrics
import bisect
import os


def get_tpr_from_fpr(fpr_array, tpr_array, target):
    fpr_index = np.where(fpr_array == target)
    assert target <= 0.01, 'the value of fpr in the custom metric function need lt 0.01'
    if len(fpr_index[0]) > 0:
        return np.mean(tpr_array[fpr_index])
    else:
        tmp_index = bisect.bisect(fpr_array, target)
        fpr_tmp_1 = fpr_array[tmp_index-1]
        fpr_tmp_2 = fpr_array[tmp_index]
        if (target - fpr_tmp_1) > (fpr_tmp_2 - target):
            tpr_index = tmp_index
        else:
            tpr_index = tmp_index - 1
        return tpr_array[tpr_index]


def eval_metric(pred, labels):
    fpr, tpr, _ = metrics.roc_curve(labels, pred, pos_label=1)
    tpr1 = get_tpr_from_fpr(fpr, tpr, 0.001)
    tpr2 = get_tpr_from_fpr(fpr, tpr, 0.005)
    tpr3 = get_tpr_from_fpr(fpr, tpr, 0.01)
    return 0.4*tpr1 + 0.3*tpr2 + 0.3*tpr3


def model_train(train, test, valid, model_type):
    params = {
        'learning_rate': 0.01,
        'boosting_type': model_type,
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 62,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'verbose': -1,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 3,
        'feature_fraction_seed': 2
		'is_unbalance': True
    }

    prob_ = np.zeros((valid.shape[0], 5))
    train_features = [t for t in valid.columns if t not in ['label', 'id']]
    # feat_imp = np.zeros((5, len(train_features)))
    submission_label = np.zeros((test.shape[0], 5))
    t = 0
    for d in train:
        print('第{}次训练...'.format(t+1))
        X_train, X_valid = d.loc[:, train_features], valid.loc[:, train_features]
        y_train, y_valid = d['label'], valid['label']
        print(X_train.shape, X_valid.shape)
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid)
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=3000,
                        valid_sets=lgb_valid,
                        verbose_eval=100,
                        early_stopping_rounds=100)
        # feat_imp[t] = gbm.feature_importance()
        prob_[:, t] = gbm.predict(X_valid)
        submission_label[:, t] = gbm.predict(test.loc[:, train_features])
        t += 1
    return pd.DataFrame(prob_), pd.DataFrame(submission_label)


if __name__ == "__main__":
    path = './deal_data/'

    train_ = (pd.read_csv(path + 'ant_train_sample_{}.csv'.format(i), encoding='utf8') for i in range(1, 6))

    test_ = pd.read_csv(path + 'ant_test_set.csv', encoding='utf8')
    valid_ = pd.read_csv(path + 'ant_valid_set.csv', encoding='utf8')

    prob_gbdt, submit_gbdt = model_train(train_, test_, valid_, 'gbdt')
    print(prob_gbdt.describe())
    print('Fold 1 Ant metric score is: {}'.format(eval_metric(prob_gbdt[0], valid_['label'])))
    print('Fold 2 Ant metric score is: {}'.format(eval_metric(prob_gbdt[1], valid_['label'])))
    print('Fold 3 Ant metric score is: {}'.format(eval_metric(prob_gbdt[2], valid_['label'])))
    print('Fold 4 Ant metric score is: {}'.format(eval_metric(prob_gbdt[3], valid_['label'])))
    print('Fold 5 Ant metric score is: {}'.format(eval_metric(prob_gbdt[4], valid_['label'])))
    final_prob = 0.14 * prob_gbdt[0] + 0.16 * prob_gbdt[1] + 0.4 * prob_gbdt[2] + 0.14 * prob_gbdt[3] + 0.16 * prob_gbdt[4]
    final_prob = final_prob.round(3)
    # print(final_prob.sort_values(ascending=False))
    score_ = eval_metric(final_prob, valid_['label'])
    print('Ant metric score is: {}'.format(score_))

    test_['score'] = 0.14 * submit_gbdt[0] + 0.16 * submit_gbdt[1] + 0.4 * submit_gbdt[2] + 0.14 * submit_gbdt[3] \
                     + 0.16 * submit_gbdt[4]
    test_['score'] = test_['score'].round(3)
    print(test_['score'].describe())
    print('test has {} records gt 0.5.'.format(sum(test_['score'] > 0.5)))
    submit = test_.loc[:, ['id', 'score']]
    date = time.strftime('%Y%m%d_%H%M%S')
    if not os.path.exists("./results"):
        os.mkdir('results')
    submit.to_csv('./results/submit_{}.csv'.format(date), index=False, encoding='utf8')
