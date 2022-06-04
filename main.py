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


def allcortogether(dataf):
    # all cooreleation graphs together
    for i in dataf:
        for k in dataf:
            plt.scatter(dataf[i], dataf[k])
    plt.show()


def allcorseperated(dataf):
    # all cooreleation graphs seperated
    for i in dataf:
        for k in dataf:
            plt.scatter(dataf[i], dataf[k])
            plt.xlabel(i)
            plt.ylabel(k)
            plt.show()


def allcorcomparison(dataf):
    sns.pairplot(dataf)


def speccor(dataf, namex, namey):
    plt.scatter(dataf[namex], dataf[namey])
    plt.xlabel(namex)
    plt.ylabel(namey)
    plt.show()


def linearreg(dataf, namex, namey):
    sns.regplot(x=dataf[namex], y=dataf[namey], data=dataf, line_kws={
        "color": "#4B0082"}, scatter_kws={"color": "#9932CC"})
    plt.show()


def calcreg(dataf, namex, namey):
    x = dataf[[namex]]
    y = dataf[[namey]]
    rl = LinearRegression().fit(x, y)
    # beta 1
    print(rl.intercept_)
    # beta 2
    print(rl.coef_)
    return rl, x, y


def getbetas(dataf, namex, namey):
    return np.polyfit(dataf[namex], dataf[namey], deg=1)


def getpredicted(rl, x, y):
    # mse
    # Predict
    y_predicted = rl.predict(x)
    # model evaluation
    mse = sklearn.metrics.mean_squared_error(y, y_predicted)
    plt.scatter(x, y, color='red')
    plt.plot(x, y_predicted, color='blue')
    plt.title('Y vs X')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    print(mse)
    print(f'Y = {rl.intercept_[0]} + {rl.coef_[0][0]} + {mse}')


namex = "fixed acidity"
namey = "citric acid"

dataf = readfile('./assets/winequality-red.csv',
                 cols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

# graficos de dispersao
print(dataf)
print(dataf.corr())

# all graphs
heatmap(dataf)
allcortogether(dataf)
# allcorseperated(dataf)
allcorcomparison(dataf)
speccor(dataf, namex, namey)
linearreg(dataf, namex, namey)
rl, x, y = calcreg(dataf, namex, namey)
# b, a = getbetas(dataf, namex, namey)
getpredicted(rl, x, y)
plt.show()
