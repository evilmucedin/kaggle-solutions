#!/usr/bin/env python3
import re
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
dfRawTrain = pd.read_csv('../train.csv', header=0)        # Load the train file into a dataframe
medianAge = dfRawTrain['Age'].dropna().median()
medianFare = dfRawTrain['Fare'].dropna().median()
dfTrain = dfRawTrain.copy()

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
    
def readFile(filename):
    fSurvivors = open(filename)
    survivorsText  = ""
    for line in fSurvivors:
        survivorsText += line
    return survivorsText.lower()

survivorsText = readFile("../Titanic Survivors.html")
victimsText = readFile("../Victims of the Titanic Disaster.html")     
deathText = readFile("../Titanic Death Certificates.shtml")     

iRe = re.finditer(">([^>]*) - death certificate</a></li>", deathText)
deaths = []
for match in iRe:
    group = match.group(0)
    deaths.append(group)

def splitName(s):
    return s.lower().replace('"', '', 1000).replace('(', ' ', 1000).replace(')', ' ', 1000).replace('master.', ' ', 1000).replace('mr.', ' ', 1000).replace("mrs.", ' ', 1000).replace('miss.', ' ', 1000).replace(",", ' ', 1000).split()

def inList(name):
    try:
        parts = splitName(name)
        for p in parts:
            # print(p, name)
            if survivorsText.find(p) < 0:
                return False
        return True
    except Exception as e:
        print(e)
        return False

def inList2(name):
    try:
        parts = splitName(name)[0:2]
        for p in parts:
            # print(p, name)
            if survivorsText.find(p) < 0:
                return False
        return True
    except Exception as e:
        print(e)
        return False

def inListCount(name):
    try:
        count = 0
        parts = splitName(name)
        for p in parts:
            if survivorsText.find(p) >= 0:
                count += 1
        return count
    except Exception as e:
        print(e)
        return 0

def inVictimsList(name):
    try:
        parts = splitName(name)
        for p in parts:
            # print(p, name)
            if victimsText.find(p) < 0:
                return False
        return True
    except Exception as e:
        print(e)
        return False

def inVictimsList2(name):
    try:
        parts = splitName(name)[0:2]
        for p in parts:
            # print(p, name)
            if victimsText.find(p) < 0:
                return False
        return True
    except Exception as e:
        print(e)
        return False

def inVictimsListCount(name):
    try:
        count = 0
        parts = splitName(name)
        for p in parts:
            if victimsText.find(p) >= 0:
                count += 1
        return count
    except Exception as e:
        print(e)
        return 0

def inDeathList(name):
    try:
        parts = splitName(name)
        for p in parts:
            if deathText.find(p) < 0:
                return False
        return True
    except Exception as e:
        print(e)
        return False

def inDeathListCount(name):
    try:
        count = 0
        parts = splitName(name)
        for p in parts:
            if deathText.find(p) >= 0:
                count += 1
        return count
    except Exception as e:
        print(e)
        return 0

def inDeathListCountItems(name):
    try:
        parts = splitName(name)
        maxCount = 0
        for d in deaths:
            count = 0
            for p in parts:
                if d.find(p) >= 0:
                    count += 1
            if count > maxCount:
                maxCount = count
        return maxCount
    except Exception as e:
        print(e)
        return 0

def inDeathListItems(name):
    try:
        parts = splitName(name)
        for d in deaths:
            foundAll = True
            for p in parts:
                if d.find(p) < 0:
                    foundAll = False
                    break
            if foundAll:
                return True
        return False
    except Exception as e:
        print(e)
        return False

def inDeathListItems2(name):
    try:
        parts = splitName(name)[0:2]
        for d in deaths:
            foundAll = True
            for p in parts:
                if d.find(p) < 0:
                    foundAll = False
                    break
            if foundAll:
                return True
        return False
    except Exception as e:
        print(e)
        return False

def nameCount(name):
    try:
        parts = splitName(name)
        return len(parts)
    except Exception as e:
        print(e)
        return 0

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
    df['NameInListAll'] = df['Name'].apply(nameCount).astype(int)
    df['NameInListHits'] = df['Name'].apply(inListCount).astype(int)
    df['NameInList'] = df['Name'].apply(inList).astype(int)
    df['NameInList2'] = df['Name'].apply(inList2).astype(int)
    df['NameInVictimsListHits'] = df['Name'].apply(inVictimsListCount).astype(int)
    df['NameInVictimsList'] = df['Name'].apply(inVictimsList).astype(int)
    df['NameInVictimsList2'] = df['Name'].apply(inVictimsList2).astype(int)
    df['NameInListDeathHits'] = df['Name'].apply(inDeathListCount).astype(int)
    df['NameInListDeath'] = df['Name'].apply(inDeathList).astype(int)
    df['NameInListDeathHitsItems'] = df['Name'].apply(inDeathListCountItems).astype(int)
    df['NameInListDeathItem'] = df['Name'].apply(inDeathListItems).astype(int)
    df['NameInListDeathItem2'] = df['Name'].apply(inDeathListItems2).astype(int)
    
    # Embarked from 'C', 'Q', 'S'
    # Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.
    
    # All missing Embarked -> just make them embark from most common place
    if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
        df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values
    
    ports = list(enumerate(np.unique(df['Embarked'])))    # determine all values of Embarked,
    dictPorts = { name : i for i, name in ports }              # set up a dictionary in the form  ports : index
    df.Embarked = df.Embarked.map( lambda x: dictPorts[x]).astype(int)     # Convert all Embark strings to int
    
    # All the ages with no data -> make the median of all Ages
    df['HasNoAge'] = df.Age.isnull().astype(int)
    if len(df.Age[ df.Age.isnull() ]) > 0:
        df.loc[ (df.Age.isnull()), 'Age'] = medianAge
    if len(df.Fare[ df.Fare.isnull() ]) > 0:
        df.loc[ (df.Fare.isnull()), 'Fare'] = medianFare
    
    # Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
    return df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


dfTrain = prepare(dfTrain)
dfTrain.to_csv("../trainConverted.csv")

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
trainData = dfTrain.values

allTrainFeatures = trainData[0::, 1::]
allTrainTarget = trainData[0::, 0]
trainFeatures, testFeatures, trainTarget, testTarget = train_test_split(allTrainFeatures, allTrainTarget, test_size=0.2)

bestNIterations = 1
bestScore = 0
bestC = None
bestModel = None

global bestScore, bestNIterations, bestC, bestModel

tp = ThreadPoolExecutor(2)

def createModel(cl, nIt):
    try:
        rfE = c(n_estimators=nIterations, learning_rate=0.01, max_depth=3)
    except:
        rfE = c(n_estimators=nIterations)
    return rfE

for nIterations in [1, 3, 5, 6, 7, 10, 20, 40, 50, 75, 100, 150, 200, 250, 400, 500, 600, 700, 1000, 1500, 2000]:
    for c in [RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, xgboost.XGBClassifier]:
        def calc(nIterations, c):
            rfE = createModel(c, nIterations)
            before = process_time()
            rfE.fit(trainFeatures, trainTarget)
            after = process_time()
            score = rfE.score(testFeatures, testTarget)
            print(c, nIterations, score, after - before)
            global bestScore, bestNIterations, bestC, bestModel
            if score > bestScore:
                bestScore = score
                bestNIterations = nIterations
                bestC = c
                bestModel = rfE
            rfE = None
                
        tp.submit(calc, nIterations, c)  
            
tp.shutdown()
print("best", bestC, bestNIterations, bestScore)

print('Training...')
forest = createModel(bestC, bestNIterations)
forest = forest.fit(allTrainFeatures, allTrainTarget)

allTrainPredicted = forest.predict_proba(allTrainFeatures)
for i in range(len(allTrainPredicted)):
    if abs(allTrainPredicted[i][1] - allTrainTarget[i]) > 0.2:
        print(dfTrain.iloc[i], dfRawTrain.iloc[i], "predictred=", allTrainPredicted[i][1], "fact=", allTrainTarget[i])

print('Predicting...')
# TEST DATA
dfTest = pd.read_csv('../test.csv', header=0)        # Load the test file into a dataframe
ids = dfTest['PassengerId'].values

dfTest = prepare(dfTest)
dfTest.to_csv("../testConverted.csv")

testData = dfTest.values
# fOut = open("../debug", "w")
# for x in testData:
#     for y in x:
#         print(y, file=fOut)
# fOut.close()

def assertAllFinite(X):
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

assertAllFinite(testData)

trainProba = bestModel.predict_proba(testFeatures)
values = []
for i, data in enumerate(testTarget):
    if 1.0 == data:
        values.append(trainProba[i][1])
values = sorted(values)
medianValue = values[len(values)//2]
print("medianValue", medianValue)

testProba = forest.predict_proba(testData)
np.savetxt('../testProba.csv', testProba, delimiter='\t')
output = [[ids[index], 1 if (x[1] > medianValue - 0.01) else 0] for index, x in enumerate(testProba)]
fPredictions = open("../myfirstforest.csv", "w")
csvPredictions = csv.writer(fPredictions)
csvPredictions.writerow(["PassengerId", "Survived"])
csvPredictions.writerows(output)
fPredictions.close()
print('Done.')