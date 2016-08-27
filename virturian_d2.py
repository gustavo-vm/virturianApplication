import pandas, numpy, statistics
from sklearn import naive_bayes, cross_validation, metrics

def main():
    robotRawData = loadRobotData()

    f1_scoreList = []

    for x in range(10):

        robotTestSet, robotTrainSet = generateTrainAndTestSets(robotRawData)

        model = createClassifier()

        featuresName = selectFeatures(robotTrainSet)

        model = trainClassifier(robotTrainSet, featuresName, model)

        result, f1_score = classifyAndEvaluate(model, robotTestSet, featuresName)

        f1_scoreList.append(f1_score)

    print("mean: ", statistics.mean(f1_scoreList))
    print("sda: ", statistics.stdev(f1_scoreList))


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

def selectFeatures(waveTrainSet):
    setLen = len(waveTrainSet.columns)

    featuresName = list(waveTrainSet.columns.values)
    featuresName = featuresName[:setLen - 2]

    return featuresName

def createClassifier():

    model = naive_bayes.GaussianNB()

    return model
def generateTrainAndTestSets(robotRawData):
    #Generate random trainig and test sets with 70% and 30% of data

    trainIndex = numpy.random.rand(len(robotRawData)) < 0.8
    TrainSet = robotRawData[trainIndex]
    TestSet = robotRawData[~trainIndex]
    return TestSet, TrainSet

def loadRobotData():

    fileToFormatName = "C:\\Users\\gustavo.v.machado\\Downloads\\lp1.data"

    filename = formatData(fileToFormatName)

    robotDataRaw = pandas.read_csv(filename, sep=',', header=None, names=None)

    return robotDataRaw

def formatData(fileName):

    formattedFileName = "F"+fileName.split("\\")[-1]


    with open(fileName) as fileToFormat:
        content = fileToFormat.readlines()

    formattedFile = open(formattedFileName, "a")
    formattedFile.seek(0)
    formattedFile.truncate()

    localCounter = 0
    label = ""
    registryString = ""

    for line in content:

        if localCounter == 0:
            registryString = ""
            label = line.split()[0]
        elif localCounter >= 1 and localCounter <= 15:
            registryString += ",".join(line.split()) + ","

        localCounter += 1

        if localCounter == 18:
            registryString += label + '\n'
            formattedFile.write(registryString)
            localCounter = 0
    formattedFile.close()

    return formattedFileName


if __name__ == "__main__":
    main()
