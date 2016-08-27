import pandas, numpy, statistics
from sklearn import metrics, svm, feature_selection, cross_validation


def main():
    considerNoise = True
    enableFS = False

    waveDataRaw = loadWaveData(considerNoise)

    f1_scoreList = []

    for x in range(10):

        waveTestSet, waveTrainSet = generateTrainAndTestSets(waveDataRaw)

        model = createClassifier()

        featuresName = selectFeatures(waveTrainSet, model, enableFS)

        model = trainClassifier(waveTrainSet, featuresName, model)

        result, f1_score = classifyAndEvaluate(model, waveTestSet, featuresName)

        f1_scoreList.append(f1_score)

    print("mean: ", statistics.mean(f1_scoreList))
    print("sda: ", statistics.stdev(f1_scoreList))


def createClassifier():

    model = svm.SVC(kernel="linear")

    return model

def generateTrainAndTestSets(waveDataRaw):
    #Generate random trainig and test sets with 70% and 30% of data

    trainIndex = numpy.random.rand(len(waveDataRaw)) < 0.7
    waveTrainSet = waveDataRaw[trainIndex]
    waveTestSet = waveDataRaw[~trainIndex]
    return waveTestSet, waveTrainSet


def selectFeatures(waveTrainSet, model, enableFS):
    setLen = len(waveTrainSet.columns)

    featuresName = list(waveTrainSet.columns.values)
    featuresName = featuresName[:setLen - 2]

    if enableFS:

        #Recursive feature selection

        recursiveFS = feature_selection.RFECV(estimator=model, step=1, cv=cross_validation.StratifiedKFold(waveTrainSet.ix[:, setLen-1], 2),
                      scoring='f1_weighted')
        recursiveFS.fit(waveTrainSet[featuresName], waveTrainSet.ix[:, setLen-1])

        #Get all features names ranked as 1

        featuresName = list(map(lambda x: x[1],filter(lambda x: x[0]==1,zip(recursiveFS.ranking_, featuresName))))

        print("Selected features:")
        print(featuresName)

    return featuresName

def classifyAndEvaluate(model, waveTestSet, featuresName):
    setLen = len(waveTestSet.columns)

    result = list(model.predict(waveTestSet[featuresName]))

    print("Result:")
    print(metrics.classification_report(waveTestSet.ix[:, setLen - 1], result))

    return result, metrics.f1_score(waveTestSet.ix[:, setLen - 1], result)


def trainClassifier(waveTrainSet, featuresName, model):
    setLen = len(waveTrainSet.columns)

    model.fit(waveTrainSet[featuresName], waveTrainSet.ix[:, setLen-1])

    return model


def loadWaveData(considerNoise):

    file = "waveform-+noise.data" if considerNoise else "waveform.data"

    waveFileName = "C:\\Users\\gustavo.v.machado\\Downloads\\" + file
    waveDataRaw = pandas.read_csv(waveFileName, sep=',', header=None, names=None)


    return waveDataRaw


if __name__ == "__main__":
    main()
