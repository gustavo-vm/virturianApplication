library(FSelector)
library(randomForest)

#Load data

setwd("C:\\Users\\gustavo.v.machado\\Downloads")

waveDataRaw = read.csv("waveform.data",header=FALSE,sep=",")

#Prepare: scale features and rename classification variable

waveFeatures = scale(waveDataRaw[,1:21])
waveLabel = waveDataRaw[,22]

waveDataRaw = as.data.frame(cbind(waveFeatures,LABEL=waveLabel))
waveDataRaw$LABEL = as.factor(waveDataRaw$LABEL)

#Separate training set from test set

set.seed(596)
trainIndex = sample(seq_len(nrow(waveDataRaw)), size = floor(0.7 * nrow(waveDataRaw)))

waveTrainSet = waveDataRaw[trainIndex, ]
waveTestSet = waveDataRaw[-trainIndex, ]

#Feature selection with CFS

topFeatures = cfs(LABEL ~.,waveTrainSet)

waveTrainSelFeat = waveTrainSet[, topFeatures]
waveTestSelFeat = waveTestSet[, topFeatures]

#Generate classification model based on random forest

waveTrain = as.data.frame(cbind(waveTrainSelFeat,LABEL=waveTrainSet$LABEL))
waveTrain$LABEL = as.factor(waveTrain$LABEL)

model = randomForest(LABEL ~ .,data=waveTrain,ntree=100,proximity=T)

result = predict(model, waveTestSelFeat)

correctClassificationRate = sum(diag(table(result, waveTestSet$LABEL)))/nrow(waveTestSet)
correctClassificationRate

