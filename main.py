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

def readfile(file,cols):
    if not cols:
        return pd.read_csv(file, usecols=cols)
    else:
        return pd.read_csv(file)

dataf=readfile('./assets/winequality-red.csv',cols=[0,1,2,3,4,5,6,7,8,9,10,11])

# graficos de dispersao
print(dataf)
print(dataf.corr())

# heat map
heatmap=sns.heatmap(dataf.corr(),annot=True,cmap="Greens")
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
plt.show()

# all cooreleation graphs together
for i in dataf:
    for k in dataf:
        plt.scatter(dataf[i],dataf[k])
plt.show()

# all cooreleation graphs seperated
""" for i in dataf:
    for k in dataf:
        plt.scatter(dataf[i],dataf[k])
        plt.xlabel(i) 
        plt.ylabel(k) 
        plt.show() """

sns.pairplot(dataf)

plt.scatter(dataf["fixed acidity"],dataf["citric acid"])

plt.xlabel("citric acid")
plt.ylabel("fixed acidity")
plt.show()

sns.regplot(x=dataf["fixed acidity"], y=dataf["citric acid"],data=dataf, line_kws={"color": "#4B0082"}, scatter_kws={"color": "#9932CC"})
plt.show()


x = dataf[['fixed acidity']]
y = dataf[['citric acid']]

rl = LinearRegression().fit(x, y)

# beta 1
print(rl.intercept_)

# beta 2
print(rl.coef_)

b, a = np.polyfit(dataf["fixed acidity"], dataf["citric acid"], deg=1)
plt.show()

# mse
# Predict
y_predicted = rl.predict(x)
  
# model evaluation
mse = sklearn.metrics.mean_squared_error(y,y_predicted)

plt.scatter(x, y, color = 'red')
plt.plot(x, y_predicted, color = 'blue')
plt.title('mark1 vs mark2')
plt.xlabel('mark1')
plt.ylabel('mark2')
plt.show()

print(mse)
print(f'Y = {rl.intercept_[0]} + {rl.coef_[0][0]} + {mse}')