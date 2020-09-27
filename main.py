import pandas as pd
pd.options.mode.chained_assignment = None
import random
import math


def train(rawData, activationType, alpha, errorCutoff):
    weights = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]

    # weights = [0.005241676415419116, 0.0025179227865492035]


    for cycle in range(500):
        # print("Weights", weights)
        print(cycle)
        totalErr = 0
        for currInput in rawData:
            net = 0
            for index in range(len(weights)):
                net += weights[index]*currInput[index]
            out = hardActivate(net) if activationType == 'hard' else softActivate(net)
            error = currInput[len(weights)] - out
            totalErr += error**2
            learn = alpha*error
            for index in range(len(weights)):
                weights[index] += learn * currInput[index]
        print(totalErr, weights)
        if totalErr <= errorCutoff:
            return weights
    return weights




def hardActivate(net):
    if net > 0:
        return 1
    if net < 0:
        return 0
    return 0.5

def softActivate(net):
    gain = 0.2

    return 1/(1+(math.exp(-gain*net)))



def normalize(rawData):
    for x in range(len(rawData.columns) - 1):
        columnData = rawData[rawData.columns[x]]
        maxVal = max(columnData)
        minVal = min(columnData)
        for y in range(len(columnData)):
            columnData[y] = (columnData[y]-minVal)/(maxVal-minVal)

    rawData.insert(2, 'bias', [1]*len(columnData))
    return rawData

def test(weights, testData):
    totalWrong = 0
    for row in testData:
        net=0
        for index in range(len(row)-1):
            net+= row[index]*weights[index]
        result = 1 if net > 0 else 0
        if result != row[len(weights)]:
            totalWrong+=1
    print(totalWrong)




if __name__ == '__main__':
    rawDataA = normalize(pd.read_csv('./Pr1_Data_ALT/DatasetA.csv', header=None))
    rawDataATest = normalize(pd.read_csv('./Pr1_Data_ALT/datasetATest.csv', header=None))
    rawDataB = normalize(pd.read_csv('./Pr1_Data_ALT/DatasetB.csv', header=None))
    rawDataBTest = normalize(pd.read_csv('./Pr1_Data_ALT/datasetBtest.csv', header=None))
    rawDataC = normalize(pd.read_csv('./Pr1_Data_ALT/DatasetC.csv', header=None))


    output = train(rawDataB.to_numpy(), 'soft', 0.01, 0.00005)

    # 0000019

    test(output, rawDataBTest.to_numpy())





