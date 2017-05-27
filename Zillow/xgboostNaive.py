#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import gc
import os
from subprocess import call
from sklearn.preprocessing import LabelEncoder

cTest = True
if cTest:
    xgboostLr = 0.1
else:
    xgboostLr = 0.001

def readOrPickle(filename):
    pklFilename = filename + ".pkl"
    if os.path.isfile(pklFilename):
        return pd.read_pickle(pklFilename)
    else:
        df = pd.read_csv(filename)
        df.to_pickle(pklFilename)
        return df

print('Loading data ...')

folder = "data"

train = readOrPickle('%s/train_2016.csv' % folder)
print('Convert timestamps ...')
train['transactiondate'] = pd.to_datetime(train['transactiondate']).astype(np.int64)

prop = readOrPickle('%s/properties_2016.csv' % folder)
sample = readOrPickle('%s/sample_submission.csv' % folder)

print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')

TO_BOOL = ['hashottuborspa','fireplaceflag']
TO_STRING = ['propertycountylandusecode', 'propertyzoningdesc']

def convert_to_bool(data, columns):
    if len(columns) != 0:
        for c in columns:
            data[c] = (data[c] == True)
    return data


def replace_nan(data):
    data['taxdelinquencyyear'] = data['taxdelinquencyyear'].fillna(0)
    data['taxdelinquencyflag'] = data['taxdelinquencyflag'].fillna('N')
    data['taxdelinquencyflag'] = LabelEncoder().fit_transform(data['taxdelinquencyflag'])
    return data

df_train = replace_nan(convert_to_bool(df_train, TO_BOOL))

print('Creating test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, how='left', on='parcelid')

del prop, sample
gc.collect()

split = 80000
df_trainS, df_validS = df_train[:split], df_train[split:]

def remove_logerror_outliers(data):
    assert 'logerror' in data.columns, 'Data provided has no logerror column'
    print('...input shape' + str(data.shape))
    upl = np.percentile(data.logerror.values, 99)
    lol = np.percentile(data.logerror.values, 1)
    result = data[(data['logerror'] > lol) & (data['logerror'] < upl)]
    print('...output shape' + str(result.shape))
    return result

df_trainS = remove_logerror_outliers(df_trainS)

x_train = df_train.drop(['logerror'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

x_trainS = df_trainS.drop(['logerror'], axis=1)
y_trainS = df_trainS['logerror'].values

x_validS = df_validS.drop(['logerror'], axis=1)
y_validS = df_validS['logerror'].values

del df_train
gc.collect()

train_columns = x_train.columns

def prepareTrain(x_train):
	for c in x_train.dtypes[x_train.dtypes == object].index.values:
		x_train[c] = (x_train[c] == True)

prepareTrain(x_train)
prepareTrain(x_trainS)
prepareTrain(x_validS)

print('Building DMatrix...')

d_trainS = lgb.Dataset(x_trainS, label=y_trainS)
d_validS = lgb.Dataset(x_validS, label=y_validS)

del x_trainS, y_trainS, x_validS, y_validS
gc.collect()

print('Training ...')

params = {}
# params['eta'] = xgboostLr
# params['objective'] = 'reg:linear'
# params['eval_metric'] = 'mae'
# params['max_depth'] = 7
# params['silent'] = 1
# params['subsample'] = 0.5
# params['colsample_bytree'] = 0.5
# params['max_bins'] = 1024

params['learning_rate'] = xgboostLr
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'mae'
params['sub_feature'] = 0.5
params['num_leaves'] = 60
params['min_data'] = 500
params['min_hessian'] = 1
params['verbosity'] = -1
params['num_threads'] = 16
params['max_bin'] = 1024
# params['max_depth'] = 10

clfS = lgb.train(params, d_trainS, 10000, [d_validS], early_stopping_rounds=100, verbose_eval=True)

del d_trainS, d_validS
gc.collect()

d_train = lgb.Dataset(x_train, label=y_train)
del x_train, y_train
gc.collect()

clf = lgb.train(params, d_train, clfS.best_iteration, verbose_eval=True)

del d_train, clfS
gc.collect()

print('Building test set ...')

for c in df_test.dtypes[df_test.dtypes == object].index.values:
    df_test[c] = (df_test[c] == True)

sub = readOrPickle('%s/sample_submission.csv' % folder)
for index, date in enumerate(['20161001', '20161101', '20161201', '20171001', '20171101', '20171201']):
    print('Predicting on test %s...' % date)

    df_test['transactiondate'] = pd.to_datetime([date]).astype(np.int64)[0]

    x_test = df_test[train_columns]
    print('Start predict...')
    p_test = clf.predict(x_test.values.astype(np.float32, copy=False))
    print('End predict...')

    del x_test
    gc.collect()

    sub[sub.columns[index + 1]] = p_test

    del p_test
    gc.collect()

del df_test
gc.collect()

outFilename = 'xgb_starter.csv'

print('Writing csv ...')
sub.to_csv(outFilename, index=False, float_format='%.4f')
call(["gzip", outFilename])
