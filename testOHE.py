from cProfile import label

from pyspark.mllib.linalg import SparseVector
import numpy as np
from pyspark import SparkConf,SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import PCA
from collections import defaultdict
import os
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD,LogisticRegressionModel
from math import log,exp
import matplotlib.pyplot as plt

os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"]="python3"

conf = SparkConf().setMaster("local")
sc = SparkContext(conf=conf)
sqlContext =SQLContext(sc)

######################################################


# one hot encoding function
def one_hot_encoding(rawFeats,OHEDict,numOHEFeats):
    validFeatureTuples =[]
    for (featID, value) in rawFeats:
        try:
            validFeatureTuples.append((OHEDict[(featID, value)], 1))
        except KeyError:
            pass
    return SparseVector(numOHEFeats, validFeatureTuples)


# auto create ohe dictionary
def create_one_hot_dict(input_df):

    retObj = (input_df.flatMap(lambda row:row)
               .distinct()
               .zipWithIndex()
               .collectAsMap())
    return retObj


raw_df = sc.textFile("/home/hadoop/PycharmProjects/data_ctr_criteo/test.txt").map(lambda x: x.replace('\t',','))
#raw_df.saveAsTextFile("subtrain_preprocess.txt")
weights =[.8,.1,.1]
seed =42
raw_train_df, raw_validation_df, raw_test_df = raw_df.randomSplit(weights, seed)

def parse_point(point):
    feats = point.split(",")[1:]
    return [(idx,value) for (idx,value) in enumerate(feats)]

parsedTrainFeat = raw_train_df.map(parse_point)
parsedValidFeat = raw_validation_df.map(parse_point)
####### PCA data ###############
pca = PCA(k=4, inputCol="features",outputCol="pcafeatures")

ctrOHEDict_train = create_one_hot_dict(parsedTrainFeat)

ctrOHEDict_valid = create_one_hot_dict(parsedValidFeat)
numCtrOHEFeats = len(ctrOHEDict_train.keys())

def pca_data(data,OHEDict):
    df = sqlContext.createDataFrame(data,["features"])
    model_pca= pca.fit(df)

    data_pca = model_pca.transform(df).map(lambda x: parse_point(x))
    numOHEFeats = len(data_pca.keys())
    print(" THIS SHIT :",data_pca)
    return one_hot_encoding(data_pca,OHEDict,numOHEFeats)

# df_train
# df_valid =sqlContext.createDataFrame(parsedTrainFeat,["features"])

# model_pca_train = pca.fit(df_train)
# data_pca_train = model_pca_train.transform(df_train).map(parse_point).select("pcafeatures")
# data_pca_train.show()

# model_pca_valid = pca.fit(df_valid)
# data_pca_valid = model_pca_valid.transform(df_valid)

#### tinh chieu dai OHEDict #################

def parseOHEPoint(point,OHEDict):
    print("\n\n",parse_point(point),"\n\n")
    pca_data(one_hot_encoding(parse_point(point), OHEDict))
    # return LabeledPoint(point.split(',')[0],pca_data(one_hot_encoding(parse_point(point),OHEDict)),OHEDict)

OHETrainData = raw_train_df.map(lambda point:parseOHEPoint(point,ctrOHEDict_train))


# OHETrainData.cache()

# def bucketFeatByCount(featCount):
#     """Bucket the counts by powers of two."""
#     for i in range(11):
#         size = 2 ** i
#         if featCount <= size:
#             return size
#     return -1
#
# featCounts = (OHETrainData
#               .flatMap(lambda lp: lp.features.indices)
#               .map(lambda x: (x, 1))
#               .reduceByKey(lambda x, y: x + y))
# featCountsBuckets = (featCounts
#                      .map(lambda x: (bucketFeatByCount(x[1]), 1))
#                      .filter(lambda x: x[0] != -1)
#                      .reduceByKey(lambda x, y: x + y)
#                      .collect())
# print (featCountsBuckets)

OHEValidationData = raw_validation_df.map(lambda point: parseOHEPoint(point, ctrOHEDict_valid, 4))
# OHEValidationData.cache()

############################
### hash data ##############
############################
# from collections import defaultdict
# import hashlib
#
# def hashFunction(numBuckets, rawFeats, printMapping = False):
#     mapping = {}
#     for ind, category in rawFeats:
#         featureString = category + str(ind)
#         featureString = featureString.encode('utf-8')
#         mapping[featureString] = int(int(hashlib.md5(featureString).hexdigest(),16)%numBuckets)
#         if(printMapping): print(mapping)
#         sparseFeatures = defaultdict(float)
#         for bucket in mapping.values():
#             sparseFeatures[bucket] +=1.0
#         return dict(sparseFeatures)
#
# def parseHashPoint(point,numBuckets):
#     fields = point.split(",")
#     label = fields[0]
#     features = parse_point(point)
#     return LabeledPoint(label,SparseVector(numBuckets,hashFunction(numBuckets,features)))
#
# numBucketCTR = 2**15
# hashTrainData = raw_train_df.map(lambda point: parseHashPoint(point,numBucketCTR))
# hashTrainData.cache()
# hashValidationData = raw_validation_df.map(lambda point:parseHashPoint(point,numBucketCTR))
# hashTrainData.cache()

# training CTR #############

numIters = 50
stepSize = 10.
regParam = 1e-10
regType = 'l2'
includeIntercept = True
model0 = LogisticRegressionWithSGD.train(OHETrainData,iterations=numIters,step=stepSize,regParam=regParam,regType=regType,intercept=includeIntercept)
sortedWeights = sorted(model0.weights)
#model0.save(sc,"/home/hadoop/PycharmProjects/ctr_predict/model0")

## train with hashdata#############

def getP(x, w, intercept):

    rawPrediction = w.dot(x) + intercept

    # Bound the raw prediction value
    rawPrediction = min(rawPrediction, 20)
    rawPrediction = max(rawPrediction, -20)
    return 1 / (1 + exp(-rawPrediction))

def computeLogLoss(p, y):
    epsilon = 10e-12
    if p == 0:
        p += epsilon
    elif p == 1:
        p -= epsilon
    if y == 1:
        return -log(p)
    else:
        return -log(1 - p)

def evaluateResults(model, data):
    """Calculates the log loss for the data given the model.

    Args:
        model (LogisticRegressionModel): A trained logistic regression model.
        data (RDD of LabeledPoint): Labels and features for each observation.

    Returns:
        float: Log loss for the data.
    """
    return (data
            .map(lambda lp: (lp.label, getP(lp.pcafeatures, model.weights, model.intercept)))
            .map(lambda x: computeLogLoss(x[1], x[0]))
            .mean())


# for stepSize in stepSizes:
#     for regParam in regParams:
#         model0 = (LogisticRegressionWithSGD
#                  .train(hashTrainData, numIters, stepSize, regParam=regParam, regType=regType,
#                         intercept=includeIntercept))
#         logLossVa = evaluateResults(model0, hashValidationData)
#         print ('\tstepSize = {0:.1f}, regParam = {1:.0e}: logloss = {2:.3f}'
#                .format(stepSize, regParam, logLossVa))
#         if (logLossVa < bestLogLoss):
#             bestModel = model0
#             bestLogLoss = logLossVa

# model0 = LogisticRegressionModel.load(sc, "model0")
###log loss,predict function#################





# trainingPredictions = hashTrainData.map(lambda lp: getP(lp.features, model0.weights, model0.intercept))

# baseline log loss #################

classOneFracTrain = OHETrainData.map(lambda lp: lp.label).mean()
print (classOneFracTrain)

logLossTrBase = OHETrainData.map(lambda lp: computeLogLoss(classOneFracTrain, lp.label)).mean()
print('Baseline Train Logloss = {0:.3f}\n'.format(logLossTrBase))

# ## evaluate model ###################
#
# def evaluateResult(model,data):
#     return (data.map(lambda x: (x.label,getP(x.features,model.weights,model.intercept))).
#             map(lambda x:computeLogLoss(x[1],x[0])).mean())
# logLossTrLR0 = evaluateResult(model0,OHETrainData)
# print("OHE Features Train LogLoss :\n baseline ={0:.3f}\n loglossTrLR0 = {0:.3f} ".format(logLossTrBase,logLossTrLR0) )

#########################
### visualize data#######
#########################
def ROC(label,result):
    from sklearn.utils import shuffle
    from sklearn.metrics import roc_curve, auc
    import pylab as pl
    # Compute ROC curve and area the curve
    Y = np.array(label)
    fpr, tpr, thresholds = roc_curve(Y, result)
    roc_auc = auc(fpr, tpr)
    print ("Area under the ROC curve : %f" % roc_auc)

    # Plot ROC curve
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('Receiver operating characteristic')
    pl.legend(loc="lower right")
    pl.show()


labelsAndScores = OHEValidationData.map(lambda lp:
                                            (lp.label, getP(lp.pcafeatures, model0.weights, model0.intercept)))


labelsAndWeights = labelsAndScores.collect()
labelsAndWeights.sort(key=lambda x: x[0], reverse=True)
X_label = np.array([k for (k, v) in labelsAndWeights])
Y_result = np.array([v for (k, v) in labelsAndWeights])
ROC(X_label,Y_result)


