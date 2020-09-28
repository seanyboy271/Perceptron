import pandas as pd
pd.options.mode.chained_assignment = None
import random
import math
from matplotlib import pyplot as plt
import numpy as np


def train(rawData, activationType, alpha, errorCutoff):
    weights = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]


    for cycle in range(5000):
        # print("Weights", weights)
        # print(cycle)
        totalErr = 0
        for currInput in rawData:
            net = 0
            for index in range(len(weights)):
                net += weights[index]*currInput[index]
            out = hardActivate(net) if activationType == 'hard' else softActivate(net)
            error = currInput[len(weights)] - out
            # if error != 0:
            #     print(error)
            totalErr += error**2
            learn = alpha*error
            for index in range(len(weights)):
                weights[index] += learn * currInput[index]
        # print(totalErr, weights)
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
    print('test result', totalWrong)

def plotData(model, rawData, fileName):
    plt.xlabel('Weight')
    plt.ylabel("Cost")
    x = np.linspace(0,1,100)
    y = (-1*model[0]*x + -1*model[2])*(1/model[1])

    data = pd.DataFrame({"X Value": rawData[0], "Y Value":rawData[1], "Category": rawData[2]})
    groups = data.groupby("Category")
    for name, group in groups:
        plt.plot(group['X Value'], group['Y Value'], marker='o', linestyle='', label='name')

    plt.plot(x, y)
    plt.savefig(fileName)
    plt.clf()


if __name__ == '__main__':
    rawDataA75 = normalize(pd.read_csv('Pr1_Data_ALT/DatasetA75.csv', header=None))
    rawDataA25 = normalize(pd.read_csv('Pr1_Data_ALT/DatasetA25.csv', header=None))
    rawDataB75 = normalize(pd.read_csv('Pr1_Data_ALT/DatasetB75.csv', header=None))
    rawDataB25 = normalize(pd.read_csv('Pr1_Data_ALT/DatasetB25.csv', header=None))
    rawDataC75 = normalize(pd.read_csv('Pr1_Data_ALT/DatasetC75.csv', header=None))
    rawDataC25 = normalize(pd.read_csv('Pr1_Data_ALT/DatasetC25.csv', header=None))


    ##Training models on 75% data
    dataAHardModel75 = train(rawDataA75.to_numpy(), 'hard', 0.1, 0.00005)
    dataBHardModel75 = train(rawDataB75.to_numpy(), 'hard', 1 * (10 ** -6), 200)
    dataCHardModel75 = train(rawDataC75.to_numpy(), 'hard', 1 * (10 ** -6), 700)

    dataASoftModel75 = train(rawDataA75.to_numpy(), 'soft', 0.3, 0.00005)
    dataBSoftModel75 = train(rawDataB75.to_numpy(), 'soft', 1 * (10 ** -3), 200)
    dataCSoftModel75 = train(rawDataC75.to_numpy(), 'soft', 1 * (10 ** -4), 700)

    #Training on 25%
    dataAHardModel25 = train(rawDataA25.to_numpy(), 'hard', 0.1, 0.00005)
    dataBHardModel25 = train(rawDataB25.to_numpy(), 'hard', 1 * (10 ** -5), 200)
    dataCHardModel25 = train(rawDataC25.to_numpy(), 'hard', 1 * (10 ** -6), 700)

    dataASoftModel25 = train(rawDataA25.to_numpy(), 'soft', 0.3, 0.00005)
    dataBSoftModel25 = train(rawDataB25.to_numpy(), 'soft', 1 * (10 ** -3), 200)
    dataCSoftModel25 = train(rawDataC25.to_numpy(), 'soft', 1 * (10 ** -4), 700)


    print('A')
    test(dataAHardModel75, rawDataA25.to_numpy())
    test(dataASoftModel75, rawDataA25.to_numpy())
    print('B')
    test(dataBHardModel75, rawDataB25.to_numpy())
    test(dataBSoftModel75, rawDataB25.to_numpy())
    print('C')
    test(dataCHardModel75, rawDataC25.to_numpy())
    test(dataCSoftModel75, rawDataC25.to_numpy())


    print('A')
    test(dataAHardModel25, rawDataA75.to_numpy())
    test(dataASoftModel25, rawDataA75.to_numpy())
    print('B')
    test(dataBHardModel25, rawDataB75.to_numpy())
    test(dataBSoftModel25, rawDataB75.to_numpy())
    print('C')
    test(dataCHardModel25, rawDataC75.to_numpy())
    test(dataCSoftModel25, rawDataC75.to_numpy())

    #Plot all of the data
    plotData(dataAHardModel75, rawDataA75, 'results/DatasetA75HardTrain.png')
    plotData(dataAHardModel75, rawDataA25, 'results/DatasetA75HardTest.png')

    plotData(dataBHardModel75, rawDataB75, 'results/DatasetB75HardTrain.png')
    plotData(dataBHardModel75, rawDataB25, 'results/DatasetB75HardTest.png')

    plotData(dataCHardModel75, rawDataC75, 'results/DatasetC75HardTrain.png')
    plotData(dataCHardModel75, rawDataC25, 'results/DatasetC75HardTest.png')

    plotData(dataASoftModel75, rawDataA75, 'results/DatasetA75SoftTrain.png')
    plotData(dataASoftModel75, rawDataA25, 'results/DatasetA75SoftTest.png')

    plotData(dataBSoftModel75, rawDataB75, 'results/DatasetB75SoftTrain.png')
    plotData(dataBSoftModel75, rawDataB25, 'results/DatasetB75SoftTest.png')

    plotData(dataCSoftModel75, rawDataC75, 'results/DatasetC75SoftTrain.png')
    plotData(dataCSoftModel75, rawDataC25, 'results/DatasetC75SoftTest.png')

    plotData(dataAHardModel25, rawDataA25, 'results/DatasetA25HardTrain.png')
    plotData(dataAHardModel25, rawDataA75, 'results/DatasetA25HardTest.png')

    plotData(dataBHardModel25, rawDataB25, 'results/DatasetB25HardTrain.png')
    plotData(dataBHardModel25, rawDataB75, 'results/DatasetB25HardTest.png')

    plotData(dataCHardModel25, rawDataC25, 'results/DatasetC25HardTrain.png')
    plotData(dataCHardModel25, rawDataC75, 'results/DatasetC25HardTest.png')

    plotData(dataASoftModel25, rawDataA25, 'results/DatasetA25SoftTrain.png')
    plotData(dataASoftModel25, rawDataA75, 'results/DatasetA25SoftTest.png')

    plotData(dataBSoftModel25, rawDataB25, 'results/DatasetB25SoftTrain.png')
    plotData(dataBSoftModel25, rawDataB75, 'results/DatasetB25SoftTest.png')

    plotData(dataCSoftModel25, rawDataC25, 'results/DatasetC25SoftTrain.png')
    plotData(dataCSoftModel25, rawDataC75, 'results/DatasetC25SoftTest.png')


