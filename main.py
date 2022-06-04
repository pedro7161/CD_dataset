from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st

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

# all cooreleation graphs togheter
for i in dataf:
    for k in dataf:
        plt.scatter(dataf[i],dataf[k])
plt.show()

# all cooreleation graphs seperated
for i in dataf:
    for k in dataf:
        plt.scatter(dataf[i],dataf[k])
        plt.xlabel(i) 
        plt.ylabel(k) 
        plt.show()

# for i in dataf:
#     for k in dataf:
#        sns.regplot(dataf[i], y=dataf[k]);
plt.scatter(dataf["fixed acidity"],dataf["citric acid"])

plt.xlabel("citric acid")
plt.ylabel("fixed acidity")
plt.show()
sns.regplot(x=dataf["fixed acidity"], y=dataf["citric acid"],data=dataf);
plt.show()
b, a = np.polyfit(dataf["fixed acidity"], dataf["citric acid"], deg=1)
xseq = np.linspace(0, 10, num=100)
plt.plot(xseq, a + b * xseq, color="k", lw=2.5);
plt.show()
print("beta 1 = ",a,"beta 2 = ",b,"erro = ",xseq)
# b, a = np.polyfit(dataf["fixed acidity"],dataf["citric acid"], deg=1)

# dataf.plot(dataf, dataf["fixed acidity"] + dataf["citric acid"] * dataf, color="k", lw=2.5);
# plt.show()


# # exempolo da net
# rng = np.random.default_rng(1234)
# # Generate data
# x = rng.uniform(0, 10, size=100)
# y = x + rng.normal(size=100)

# # Initialize layout
# fig, ax = plt.subplots(figsize = (9, 9))

# # Add scatterplot
# ax.scatter(x, y, s=60, alpha=0.7, edgecolors="k")

# # Fit linear regression via least squares with numpy.polyfit
# # It returns an slope (b) and intercept (a)
# # deg=1 means linear fit (i.e. polynomial of degree 1)
# b, a = np.polyfit(x, y, deg=1)

# # Create sequence of 100 numbers from 0 to 100 
# xseq = np.linspace(0, 10, num=100)

# # Plot regression line
# ax.plot(xseq, a + b * xseq, color="k", lw=2.5);
# plt.show()
