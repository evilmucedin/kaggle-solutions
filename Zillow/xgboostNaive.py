#!/usr/bin/env python3

import gc
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from subprocess import call
from sklearn.preprocessing import LabelEncoder

import numpy as np
import theano
import theano.tensor as T

import lasagne

cTest = True
if cTest:
    xgboostLr = 0.1
    num_epochs = 200
    numUnits1 = 100
    numUnits2 = 15
    nnLr = 0.0001
else:
    xgboostLr = 0.001
    num_epochs = 2000
    numUnits1 = 200
    numUnits2 = 30
    nnLr = 0.00001

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

def dump(df):
    for c in df.columns:
        print(c, df[c])

dump(prop)

sample = readOrPickle('%s/sample_submission.csv' % folder)

print('Binding to float32')

for c, dtype in zip(prop.columns, prop.dtypes):
    if dtype == np.float64:
        prop[c] = prop[c].astype(np.float32)

print('Creating training set ...')

df_train = train.merge(prop, how='left', on='parcelid')

del train
gc.collect()

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
    for c in ['taxdelinquencyflag', 'propertyzoningdesc', 'propertycountylandusecode']:
        data[c] = data[c].values.astype(np.str)
        data[c] = LabelEncoder().fit_transform(data[c])
    return data

df_train = replace_nan(convert_to_bool(df_train, TO_BOOL))

drop_columns = []
for x in df_train.columns:
    c = df_train[x]
    nan = 0
    notNan = 0
    for v in c.astype(np.float32, copy=False):
        if np.isnan(v):
            nan += 1
        else:
            notNan += 1
    if 0 != nan:
        print('Has Nan: %s nan=%d notNan=%d' % (x, nan, notNan))
    if 0 == notNan:
        drop_columns.append(x)

print('Drop: ', ','.join(drop_columns))
df_train = df_train.drop(drop_columns, axis=1)

df_train = df_train.iloc[np.random.permutation(len(df_train))]

print('Creating test set ...')

sample['parcelid'] = sample['ParcelId']
df_test = sample.merge(prop, how='left', on='parcelid')

del prop, sample
gc.collect()

split = (len(df_train)*9)//10
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

del df_train, df_trainS, df_validS
gc.collect()

train_columns = x_train.columns

print("Prepare x_test...")

for c in df_test.dtypes[df_test.dtypes == object].index.values:
    df_test[c] = (df_test[c] == True)

df_test['transactiondate'] = pd.to_datetime(['20161001']).astype(np.int64)[0]

x_test = df_test[train_columns]

del df_test
gc.collect()

def prepareTrain(x_train):
    for c in x_train.dtypes[x_train.dtypes == object].index.values:
        x_train[c] = (x_train[c] == True)

print("Prepare train and valid...")

prepareTrain(x_train)
prepareTrain(x_trainS)
prepareTrain(x_validS)

dropout = True

def build_mlp(vInput):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, len(train_columns)), input_var=vInput)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=numUnits1,
            nonlinearity=lasagne.nonlinearities.linear,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop if dropout else l_hid1, num_units=numUnits2,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.GlorotUniform())

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop if dropout else l_hid2, num_units=1,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.GlorotUniform())

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def train_nn(featuresTrain, labelsTrain, featuresTest, labelsTest):
    # Prepare Theano variables for inputs and targets
    input_var = T.fmatrix('inputs')
    target_var = T.fmatrix('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_mlp(input_var)

    train_prediction = lasagne.layers.get_output(network)
    predict_prediction = lasagne.layers.get_output(network, deterministic=True)

    params = lasagne.layers.get_all_params(network, trainable=True)
    train_loss = abs(train_prediction - target_var).mean() + 0.0001*lasagne.regularization.l2(train_prediction)
    predict_loss = abs(predict_prediction - target_var).mean()

    trainUpdates = lasagne.updates.nesterov_momentum(train_loss, params, learning_rate=nnLr, momentum=0.9)

    train_fn = theano.function([input_var, target_var], train_loss, updates=trainUpdates)
    val_fn = theano.function([input_var, target_var], predict_loss)
    predict_fn = theano.function([input_var], predict_prediction)
    
    # Finally, launch the training loop.
    print("Starting training...")
    batchSize = 1000
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(featuresTrain, labelsTrain, batchSize, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(featuresTrain, labelsTrain, batchSize, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1

        # And a full pass over the test data:
        test_err = 0
        test_batches = 0
        for batch in iterate_minibatches(featuresTest, labelsTest, batchSize, shuffle=False):
            inputs, targets = batch
            # print(inputs)
            err = val_fn(inputs, targets)
            # print(predict_fn(inputs))
            test_err += err
            test_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  test loss:\t\t{:.6f}".format(test_err / test_batches))
    
    print("Training done...")
    return predict_fn

def castF(x):
    casted = x.astype(np.float32, copy=False)
    casted[np.isnan(casted)] = 1000000
    casted[np.isinf(casted)] = 1000000
    return casted

x_trainSF = castF(x_trainS.values)

normMin = x_trainSF.min(axis=0)
normMax = x_trainSF.max(axis=0)
b = min(y_trainS)

print("Validation", x_validS.shape, b)

def transformFeatures(df):
    return ((castF(df) - normMin)/(normMax - normMin)) - 0.5

x_validSF = transformFeatures(x_validS)
y_validSF = castF(y_validS).reshape(len(y_validS), 1)

predict_fn = train_nn(transformFeatures(x_trainSF), castF(y_trainS).reshape(len(y_trainS), 1) - b, x_validSF, y_validSF - b)
nnPrediction = predict_fn(x_validSF) + b
print(nnPrediction, y_validSF)

print("Full prediction...")
nnPredictionFull = predict_fn(transformFeatures(castF(x_test.values))) + b

del x_trainSF, normMin, normMax, x_validSF, predict_fn
gc.collect()

print('Building DMatrix...')

d_trainS = lgb.Dataset(x_trainS, label=y_trainS)
d_validS = lgb.Dataset(x_validS, label=y_validS)

del x_trainS, y_trainS
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
splitImportance = clfS.feature_importance('split')
gainImportance = clfS.feature_importance('gain')
for iCol, col in enumerate(train_columns):
    print("%s\t%f\t%f" % (col, splitImportance[iCol], gainImportance[iCol]))
gdbPredictions = clfS.predict(x_validS.values.astype(np.float32, copy=False))
gdbBestIteration = clfS.best_iteration
print("Best iteration: %d" % gdbBestIteration)

del clfS
gc.collect()

minMae = 1e10
bestLambda = 0
for i in range(101):
    lmbd = float(i)/100
    predictions = lmbd*nnPrediction + (1.0 - lmbd)*gdbPredictions
    mae = abs(y_validSF - predictions).mean()
    if mae < minMae:
        minMae = mae
        bestLambda = lmbd
    print(lmbd, mae)
print("opt", bestLambda, minMae)

del d_trainS, d_validS, y_validSF, gdbPredictions, x_validS, predictions
gc.collect()

d_train = lgb.Dataset(x_train, label=y_train)
del x_train, y_train
gc.collect()

clf = lgb.train(params, d_train, verbose_eval=True)

del d_train
gc.collect()

print('Building test set ...')

sub = readOrPickle('%s/sample_submission.csv' % folder)
dates = ['20161001', '20161101', '20161201', '20171001', '20171101', '20171201']
for index, date in enumerate(dates):
    print('Predicting on test %s...' % date)

    x_test['transactiondate'] = pd.to_datetime([date]).astype(np.int64)[0]

    print('Start predict...')
    p_test = np.array(len(x_test), dtype=np.float32)
    batchSize = 1000
    batchBegin = 0
    iBatch = 0
    while batchBegin < len(x_test):
        iBatch += 1
        if iBatch % 100 == 0:
            print("\t...%d" % iBatch)
        batchEnd = min(len(x_test), batchBegin + batchSize)
        p_test[batchStart:batchEnd] = clf.predict(x_test[batchStart:batchEnd].values.astype(np.float32, copy=False))
        batchBegin = batchEnd
    print('End predict...')

    sub[sub.columns[index + 1]] = p_test

    del p_test
    gc.collect()

del x_test
gc.collect()

print('Start NN...')
subNN = readOrPickle('%s/sample_submission.csv' % folder)
for index, _ in enumerate(dates):
    subNN[subNN.columns[index + 1]] = bestLambda*nnPredictionFull + (1.0 - bestLambda)*sub[sub.columns[index + 1]]

suffix = "_test" if cTest else ""
outFilename = 'xgb_starter%s.csv' % suffix

print('Writing csv ...')
sub.to_csv(outFilename, index=False, float_format='%.4f')
call(["gzip", "-q", outFilename])

outFilenameNN = 'xgb_starterNN%s.csv' % suffix

print('Writing csv nn ...')
subNN.to_csv(outFilenameNN, index=False, float_format='%.4f')
call(["gzip", "-q", outFilenameNN])
