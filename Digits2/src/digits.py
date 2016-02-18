#!/usr/bin/env python3

import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model build in Lasagne.

numUnits = 2000

def build_mlp(vInput=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 784), input_var=vInput)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=numUnits,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=numUnits,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

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

def iterate_minibatches2(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

def loadDataset():
    dtype = np.float32
    data = np.loadtxt("../trainAll.csv", dtype=dtype, delimiter=',', skiprows=1)
    print("Data: ", data.shape)

    labelsTrain = data[:,0].astype(np.int32)
    featuresTrain = data[:, 1:]
    data = None

    amean = np.mean(featuresTrain)
    featuresTrain = featuresTrain - amean
    astd = np.std(featuresTrain)
    featuresTrain = featuresTrain / astd

    featuresTest = np.loadtxt("../test.csv", dtype=dtype, delimiter=',', skiprows=1)
    print("TestData: ", featuresTest.shape)
    featuresTest = (featuresTest - amean) / astd
    featuresTest = featuresTest[:, 1:]
    
    return labelsTrain, featuresTrain, featuresTest


def main():
    print("Loading data...")
    labelsTrain, featuresTrain, featuresTest = loadDataset()

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_mlp(input_var)

    prediction = lasagne.layers.get_output(network)
    pureloss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    pureloss = pureloss.mean()

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    loss = pureloss + 0.0001*lasagne.regularization.l2(test_prediction)

    params = lasagne.layers.get_all_params(network, trainable=True)
    trainUpdates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.002, momentum=0.9)

    train_fn = theano.function([input_var, target_var], loss, updates=trainUpdates)
    pureTrain_fn = theano.function([input_var, target_var], pureloss)

    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))
    
    def save():
        preds = []
        for batch in iterate_minibatches2(featuresTest, batchSize, shuffle=False):
            preds.extend(predict_fn(batch))
        
        subm = np.empty((len(featuresTest), 2))
        subm[:, 0] = np.arange(1, len(featuresTest) + 1)
        subm[:, 1] = preds
        np.savetxt('../submission.csv', subm, fmt='%d', delimiter=',', header='ImageId,Label', comments='')

    # Finally, launch the training loop.
    print("Starting training...")
    num_epochs = 1000
    batchSize = 200
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_pureErr = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(featuresTrain, labelsTrain, batchSize, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_pureErr += pureTrain_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(featuresTrain, labelsTrain, batchSize, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  pure training loss:\t\t{:.6f}".format(train_pureErr / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
        save()
    
    print("Training done...")

    save()
    
if __name__ == '__main__':
    main()
