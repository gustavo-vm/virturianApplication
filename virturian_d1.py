import pandas, numpy
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def main():
    considerNoise = True

    waveDataRaw = loadWaveData(considerNoise)

    trainIndex = numpy.random.rand(len(waveDataRaw)) < 0.7

    waveTrainSet = waveDataRaw[trainIndex]
    waveTestSet = waveDataRaw[~trainIndex]

    featuresName = selectFeatures(waveTrainSet)

    model = createAndTrainClassifier(waveTrainSet, featuresName)

    classifyAndEvaluate(model, waveTestSet, featuresName)

def selectFeatures(waveTrainSet):

    featuresName = list(waveTrainSet.columns.values)
    featuresName = featuresName[:len(featuresName) - 2]

    return featuresName

def classifyAndEvaluate(model, waveTestSet, featuresName):
    setLen = len(waveTestSet.columns)

    result = list(model.predict(waveTestSet[featuresName]))

    print(metrics.classification_report(waveTestSet.ix[:, setLen - 1], result))


def createAndTrainClassifier(waveTrainSet, featuresName):
    setLen = len(waveTrainSet.columns)

    RFModel = RandomForestClassifier(n_estimators=100)
    RFModel.fit(waveTrainSet[featuresName], waveTrainSet.ix[:, setLen-1])

    print("Features by importance:")
    print(sorted(zip(map(lambda x: round(x, 4), RFModel.feature_importances_), featuresName),reverse=True))

    return RFModel


def loadWaveData(considerNoise):

    file = "waveform-+noise.data" if considerNoise else "waveform.data"

    waveFileName = "C:\\Users\\gustavo.v.machado\\Downloads\\" + file
    waveDataRaw = pandas.read_csv(waveFileName, sep=',', header=None, names=None)


    return waveDataRaw


if __name__ == "__main__":
    main()
