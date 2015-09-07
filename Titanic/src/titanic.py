#!/usr/bin/env python3
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from concurrent.futures import ThreadPoolExecutor 
import xgboost
from time import process_time
from numpy.f2py.auxfuncs import isinteger

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('../train.csv', header=0)        # Load the train file into a dataframe
medianAge = train_df['Age'].dropna().median()
medianFare = train_df['Fare'].dropna().median()

def isinteger(s):
    try:
        if len(s) == 0:
            return False
        for ch in s:
            if ch < '0' or ch > '9':
                return False
            return True
    except:
        return False
    
def hasA(s):
    try:
        return s.find("A") >= 0
    except:
        return False

def hasB(s):
    try:
        return s.find("B") >= 0
    except:
        return False

def hasC(s):
    try:
        return s.find("C") >= 0
    except:
        return False

def hasD(s):
    try:
        return s.find("D") >= 0
    except:
        return False

def hasE(s):
    try:
        return s.find("E") >= 0
    except:
        return False

def hasF(s):
    try:
        return s.find("F") >= 0
    except:
        return False

def hasMr(s):
    try:
        return s.find("Mr.") >= 0
    except:
        return False

def hasMrs(s):
    try:
        return s.find("Mrs.") >= 0
    except:
        return False

def hasMiss(s):
    try:
        return s.find("Miss.") >= 0
    except:
        return False

def hasPC(s):
    try:
        return s.find("PC ") >= 0
    except:
        return False

def hasST(s):
    try:
        return s.find("ST") >= 0
    except:
        return False

def hasSpace(s):
    try:
        return s.find(" ") >= 0
    except:
        return False

def lenSp(s):
    try:
        return len(list(filter(lambda ch: ch == ' ', s)))
    except:
        return False

def lenQuote(s):
    try:
        return len(list(filter(lambda ch: ch == '"', s)))
    except:
        return False


def prepare(df):
    # I need to convert all strings to integer classifiers.
    # I need to fill in the missing values of the data and make it complete.
    
    # female = 0, Male = 1
    df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    df['CabinType'] = df['Cabin'].isnull().astype(int)
    df['CabinType2'] = df['Cabin'].apply(hasSpace).astype(int)
    df['CabinA'] = df['Cabin'].apply(hasA).astype(int)
    df['CabinB'] = df['Cabin'].apply(hasB).astype(int)
    df['CabinC'] = df['Cabin'].apply(hasC).astype(int)
    df['CabinD'] = df['Cabin'].apply(hasD).astype(int)
    df['CabinE'] = df['Cabin'].apply(hasE).astype(int)
    df['CabinF'] = df['Cabin'].apply(hasF).astype(int)
    df['TicketType'] = df['Ticket'].apply(isinteger).astype(int)
    df['TicketPC'] = df['Ticket'].apply(hasPC).astype(int)
    df['TicketST'] = df['Ticket'].apply(hasST).astype(int)
    df['TicketSp'] = df['Ticket'].apply(hasSpace).astype(int)
    df['HasMr'] = df['Name'].apply(hasMr).astype(int)
    df['HasMrs'] = df['Name'].apply(hasMrs).astype(int)
    df['HasMiss'] = df['Name'].apply(hasMiss).astype(int)
    df['NameLen'] = df['Name'].apply(len).astype(int)
    df['NameSpLen'] = df['Name'].apply(lenSp).astype(int)
    df['NameQuoteLen'] = df['Name'].apply(lenQuote).astype(int)
    
    # Embarked from 'C', 'Q', 'S'
    # Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.
    
    # All missing Embarked -> just make them embark from most common place
    if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
        df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values
    
    ports = list(enumerate(np.unique(df['Embarked'])))    # determine all values of Embarked,
    dictPorts = { name : i for i, name in ports }              # set up a dictionary in the form  ports : index
    df.Embarked = df.Embarked.map( lambda x: dictPorts[x]).astype(int)     # Convert all Embark strings to int
    
    # All the ages with no data -> make the median of all Ages
    df['HasAge'] = df.Age.isnull().astype(int)
    if len(df.Age[ df.Age.isnull() ]) > 0:
        df.loc[ (df.Age.isnull()), 'Age'] = medianAge
    if len(df.Fare[ df.Fare.isnull() ]) > 0:
        df.loc[ (df.Fare.isnull()), 'Fare'] = medianFare
    
    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
    return df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


train_df = prepare(train_df)
train_df.to_csv("../trainConverted.csv")

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values

trainFeatures, testFeatures, trainTarget, testTarget = train_test_split(train_data[0::, 1::], train_data[0::, 0], test_size=0.2)

bestNIterations = 1
bestScore = 0
bestC = None

global bestScore, bestNIterations, bestC

tp = ThreadPoolExecutor(2)

for nIterations in [1, 3, 5, 6, 7, 10, 20, 40, 50, 75, 100, 150, 200, 250, 400, 500, 600, 700, 1000]:
    for c in [RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, xgboost.XGBClassifier]:
    # for c in [RandomForestClassifier]:
    # for c in [xgboost.XGBClassifier]:    
        def calc(nIterations, c):
            try:
                rfE = c(n_estimators=nIterations, n_jobs=2)
            except:
                rfE = c(n_estimators=nIterations)
            before = process_time()
            rfE.fit(trainFeatures, trainTarget)
            after = process_time()
            score = rfE.score(testFeatures, testTarget)
            rfE = None
            print(c, nIterations, score, after - before)
            global bestScore, bestNIterations, bestC
            if score > bestScore:
                bestScore = score
                bestNIterations = nIterations
                bestC = c
                
        tp.submit(calc, nIterations, c)  
            
tp.shutdown()
print("best", bestC, bestNIterations, bestScore)

print('Training...')
forest = bestC(n_estimators=bestNIterations)
forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])

print('Predicting...')
# TEST DATA
test_df = pd.read_csv('../test.csv', header=0)        # Load the test file into a dataframe
ids = test_df['PassengerId'].values

test_df = prepare(test_df)
test_df.to_csv("../testConverted.csv")

test_data = test_df.values
fOut = open("../debug", "w")
for x in test_data:
    for y in x:
        print(y, file=fOut)
fOut.close()

def _assert_all_finite(X):
    """Like assert_all_finite, but only for ndarray."""
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    for ix, x in enumerate(X):
        for iy, y in enumerate(x):
            if not np.isfinite(y):
                raise ValueError("%d %d %f %d %d" % (ix, iy, y, len(X), len(x)))
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r. %f" % (X.dtype, X.sum()))

_assert_all_finite(test_data)

output = [[ids[index], 1 if (x[1] > 0.5) else 0] for index, x in enumerate(forest.predict_proba(test_data))]
predictions_file = open("../myfirstforest.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(output)
predictions_file.close()
print('Done.')