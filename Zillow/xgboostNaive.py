#!/usr/bin/env python3

import numpy as np
import pandas as pd
import xgboost as xgb
import gc
import os
from subprocess import call

cTest = False
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

print('Creating test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, how='left', on='parcelid')

del prop, sample
gc.collect()

x_train = df_train.drop(['logerror'], axis=1)
y_train = df_train['logerror'].values
print(x_train.shape, y_train.shape)

del df_train
gc.collect()

train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)


split = 80000
x_trainS, y_trainS, x_validS, y_validS = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print('Building DMatrix...')

d_trainS = xgb.DMatrix(x_trainS, label=y_trainS)
d_validS = xgb.DMatrix(x_validS, label=y_validS)

del x_trainS, y_trainS, x_validS, y_validS
gc.collect()

print('Training ...')

params = {}
params['eta'] = xgboostLr
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 7
params['silent'] = 1
params['subsample'] = 0.5
params['colsample_bytree'] = 0.5
params['max_bins'] = 1024

watchlistS = [(d_trainS, 'train'), (d_validS, 'valid')]
clfS = xgb.train(params, d_trainS, 10000, watchlistS, early_stopping_rounds=100, verbose_eval=10)

del d_trainS, d_validS
gc.collect()

d_train = xgb.DMatrix(x_train, label=y_train)
del x_train, y_train
gc.collect()

watchlist = [(d_train, 'train')]
clf = xgb.train(params, d_train, clfS.best_iteration, watchlist, verbose_eval=10)

del d_train, clfS
gc.collect()

print('Building test set ...')

x_test = df_test[train_columns]
for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

del df_test
gc.collect()

sub = readOrPickle('%s/sample_submission.csv' % folder)
for index, date in enumerate(['20161001', '20161101', '20161201', '20171001', '20171101', '20171201']):
    print('Predicting on test %s...' % date)

    x_test['transactiondate'] = pd.to_datetime([date]).astype(np.int64)[0]

    d_test = xgb.DMatrix(x_test)

    p_test = clf.predict(d_test)

    sub[sub.columns[index + 1]] = p_test

    del d_test, p_test
    gc.collect()

del x_test
gc.collect()

outFilename = 'xgb_starter.csv'

print('Writing csv ...')
sub.to_csv(outFilename, index=False, float_format='%.4f')
call(["gzip", outFilename])
