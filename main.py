from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import sklearn.metrics
from sklearn.linear_model import LinearRegression


def mse(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    differences = np.subtract(actual, predicted)
    squared_differences = np.square(differences)
    return squared_differences.mean()


def readfile(file, cols):
    if not cols:
        return pd.read_csv(file, usecols=cols)
    else:
        return pd.read_csv(file)


def heatmap(dataf):
    heatmap = sns.heatmap(dataf.corr(), annot=True, cmap="Greens")
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    plt.show()


def allCor(dataf):
    # all cooreleation graphs together
    for i in dataf:
        for k in dataf:
            plt.scatter(dataf[i], dataf[k])
    plt.show()


def allCorSeperated(dataf):
    # all cooreleation graphs seperated
    for i in dataf:
        for k in dataf:
            plt.scatter(dataf[i], dataf[k])
            plt.xlabel(i)
            plt.ylabel(k)
            plt.show()


def allCorComparison(dataf):
    sns.pairplot(dataf)


def specCor(dataf, namex, namey):
    plt.scatter(dataf[namex], dataf[namey])
    plt.xlabel(namex)
    plt.ylabel(namey)
    plt.show()


def linearReg(dataf, namex, namey):
    sns.regplot(x=dataf[namex], y=dataf[namey], data=dataf, line_kws={
        "color": "#4B0082"}, scatter_kws={"color": "#9932CC"})
    plt.title("Regressão Linear")
    plt.show()


def calcReg(dataf, namex, namey):
    x = dataf[[namex]]
    y = dataf[[namey]]

    rl = LinearRegression().fit(x, y)

    # beta 0
    print(rl.intercept_)

    # beta 1
    print(rl.coef_)

    return rl, x, y


def getBetas(dataf, namex, namey):
    return np.polyfit(dataf[namex], dataf[namey], deg=1)


def getPredicted(rl, x, y):
    # Predict Y value based on given X value
    y_predicted = rl.predict(x)

    # model evaluation
    # mse
    mse = sklearn.metrics.mean_squared_error(y, y_predicted)
    print(mse)
    print(f'Y = {rl.intercept_[0]} + {rl.coef_[0][0]} + {mse}')


# main - Variables Declarations
namex = "fixed acidity"
namey = "citric acid"

dataf = readfile('./assets/winequality-red.csv',
                 cols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# dispersion graph
print(dataf)
print(dataf.corr())

# all graphs
heatmap(dataf)
allCor(dataf)

# allCorSeperated(dataf)

allCorComparison(dataf)
specCor(dataf, namex, namey)
linearReg(dataf, namex, namey)

rl, x, y = calcReg(dataf, namex, namey)
getPredicted(rl, x, y)

# b, a = getBetas(dataf, namex, namey)

plt.show()
