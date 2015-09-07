#!/usr/bin/env python3
import pickle
from time import process_time
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from numpy import genfromtxt, savetxt
from concurrent.futures import ThreadPoolExecutor 
import xgboost

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt('../train.csv', delimiter=',', dtype='f8', skip_header=1)[1:]

    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    
    trainFeatures, testFeatures, trainTarget, testTarget = train_test_split(train, target, test_size=0.2)
    print("data loaded")

#     rf = xgboost.XGBClassifier()
#     rf.fit(trainFeatures, trainTarget)
#     score = rf.score(train, target)
#     test = genfromtxt('../test.csv', delimiter=',', dtype='f8', skip_header=1)
#     savetxt('../xgboost.csv', rf.predict_proba(test), delimiter='\t')
#     print("ok ", score)
    
    tp = ThreadPoolExecutor(2)

    global bestScore, bestNIterations, bestC
    bestNIterations = 1
    bestScore = 0
    bestC = None
    for nIterations in [1, 5, 10, 50, 100, 150, 200, 250, 400, 500, 700, 1000]:
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
                if score > bestScore:
                    global bestScore, bestNIterations, bestC
                    bestScore = score
                    bestNIterations = nIterations
                    bestC = c
                    
            if c == GradientBoostingClassifier:
                if nIterations < 100:
                    tp.submit(calc, nIterations, c)  
            elif c == AdaBoostClassifier:
                if nIterations < 1000:
                    tp.submit(calc, nIterations, c)
            else:  
                tp.submit(calc, nIterations, c)
                
    tp.shutdown()
    print("best", bestC, bestNIterations, bestScore)
                
    rf = bestC(n_estimators=bestNIterations)
    rf.fit(train, target)
    with open('bestModel.pickle', 'wb') as f:
        pickle.dump(rf, f)
    
    test = genfromtxt('../test.csv', delimiter=',', dtype='f8', skip_header=1)
    savetxt('../test2.csv', test, delimiter='\t')
    predicted_probs = [[index + 1, x[1]] for index, x in enumerate(rf.predict_proba(test))]
    savetxt('../submission.csv', predicted_probs, delimiter=',', fmt='%d,%f', header='MoleculeId,PredictedProbability', comments = '')

if __name__=="__main__":
    main()
