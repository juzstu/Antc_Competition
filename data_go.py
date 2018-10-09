# -*- coding：utf-8 -*-
# Author: Studog
# Time: 2018/6/22 20:54

import pandas as pd
import datetime
import os
import time
import numpy as np
# 训练集 label {'0': 977884, '1': 12122, '-1': 4725}


def get_week_of_month(df):
    year = int(df[:4])
    month = int(df[4:6])
    day = int(df[6:])
    end = int(datetime.datetime(year, month, day).strftime("%W"))
    begin = int(datetime.datetime(year, month, 1).strftime("%W"))
    return end - begin + 1


def get_period_of_month(df):
    day = int(df[6:])
    if day <= 10:
        return 0
    elif day > 20:
        return 2
    else:
        return 1


def judge_is_weekend(df):
    if df > 4:
        return 1
    else:
        return 0


def transform_date(data_frame):
    data_frame['date_copy'] = data_frame['date']
    data_frame['date_copy'] = data_frame['date_copy'].astype(str)
    data_frame['week_in_month'] = data_frame['date_copy'].apply(get_week_of_month)
    data_frame['period_in_month'] = data_frame['date_copy'].apply(get_period_of_month)
    data_frame['day'] = data_frame['date_copy'].apply(lambda x: int(x[6:]))
    data_frame['date_copy'] = data_frame['date_copy'].apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:])
    data_frame['date_copy'] = pd.to_datetime(data_frame['date_copy'])
    data_frame['week_day'] = data_frame['date_copy'].apply(lambda x: x.weekday())
    data_frame['is_weekend'] = data_frame['week_day'].apply(judge_is_weekend)
    del data_frame['date_copy']
    return data_frame


def sample(data_frame, cols, neg_ratio):
    date_group = (dg for dg in data_frame.groupby(by='date'))
    temp_data = pd.DataFrame(columns=cols)
    for dg in date_group:
        dg_label_1 = dg[1][dg[1]['label'] != 0]
        dg_label_0 = dg[1][dg[1]['label'] == 0]
        dg_label_0_sample = dg_label_0.sample(n=dg_label_1.shape[0] * neg_ratio, random_state=1024)
        dg_label_0_sample = dg_label_0_sample.append(dg_label_1)
        dg_label_0_sample = dg_label_0_sample.sample(frac=1, random_state=1024)
        temp_data = temp_data.append(dg_label_0_sample.loc[:, cols])
    temp_data['label'] = np.abs(temp_data['label'])
    return temp_data


def fill_miss(data_frame, data_temp):
    for c in data_frame.columns:
        if not all(data_frame[c].notnull()):
            data_temp[c] = data_temp[c].fillna(data_frame[c].median())
    return data_temp


# 20170905-20170914, 20170915-20170924, 20170925-20171004, 20171005-20171014, 20171015-20171024, 20171025-20171105
start_time = time.time()
train = pd.read_csv('./data/atec_anti_fraud_train.csv', encoding='utf8')
test = pd.read_csv('./data/atec_anti_fraud_test_b.csv', encoding='utf8')
print('the origin shape of train data: {}.'.format(train.shape))

null_sum = train.isnull().sum()
null_sum_df = pd.DataFrame(null_sum, columns=['num'])
null_sum_df['ratio'] = null_sum_df['num'] / train.shape[0]
drop_list = list(null_sum_df[null_sum_df['ratio'] > 0.5].index)
print(drop_list)
train.drop(drop_list, axis=1, inplace=True)
test.drop(drop_list, axis=1, inplace=True)

train = transform_date(train)
test = transform_date(test)
columns = train.columns
train_1 = train[train['date'] <= 20170914]
train_2 = train[(train['date'] >= 20170915) & (train['date'] <= 20170924)]
train_3 = train[(train['date'] >= 20170925) & (train['date'] <= 20171004)]
train_4 = train[(train['date'] >= 20171005) & (train['date'] <= 20171014)]
train_5 = train[(train['date'] >= 20171015) & (train['date'] <= 20171024)]
valid = train[train['date'] >= 20171025]
valid = valid[valid['label'] != -1]

train_1_sample = sample(train_1, columns, 13)
train_2_sample = sample(train_2, columns, 13)
train_3_sample = sample(train_3, columns, 13)
train_4_sample = sample(train_4, columns, 13)
train_5_sample = sample(train_5, columns, 13)

train_1_sample['missing'] = train_1_sample.isnull().sum(axis=1).astype(float)
train_2_sample['missing'] = train_2_sample.isnull().sum(axis=1).astype(float)
train_3_sample['missing'] = train_3_sample.isnull().sum(axis=1).astype(float)
train_4_sample['missing'] = train_4_sample.isnull().sum(axis=1).astype(float)
train_5_sample['missing'] = train_5_sample.isnull().sum(axis=1).astype(float)

valid['missing'] = valid.isnull().sum(axis=1).astype(float)
test['missing'] = test.isnull().sum(axis=1).astype(float)

del train_1_sample['date']
del train_2_sample['date']
del train_3_sample['date']
del train_4_sample['date']
del train_5_sample['date']
del valid['date']
del test['date']

print(f'train_1_sample: {train_1_sample.shape}')
print(f'train_2_sample: {train_2_sample.shape}')
print(f'train_3_sample: {train_3_sample.shape}')
print(f'train_4_sample: {train_4_sample.shape}')
print(f'train_5_sample: {train_5_sample.shape}')
print('***************************************')
print(f'valid: {valid.shape}')
print(f'test: {test.shape}')

path = './deal_data'
if not os.path.exists(path):
    os.mkdir(path)

print('Start saving the final train, valid and test data ...')
train_1_sample.to_csv(path + '/ant_train_sample_1.csv', encoding='utf8', index=False)
train_2_sample.to_csv(path + '/ant_train_sample_2.csv', encoding='utf8', index=False)
train_3_sample.to_csv(path + '/ant_train_sample_3.csv', encoding='utf8', index=False)
train_4_sample.to_csv(path + '/ant_train_sample_4.csv', encoding='utf8', index=False)
train_5_sample.to_csv(path + '/ant_train_sample_5.csv', encoding='utf8', index=False)
valid.to_csv(path + '/ant_valid_set.csv', encoding='utf8', index=False)
test.to_csv(path + '/ant_test_set.csv', encoding='utf8', index=False)
print('Finish the mission of saving.')
print('feature work finish, total cost {} seconds.'.format(time.time() - start_time))
