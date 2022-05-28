
from matplotlib import pyplot as plt
from operator import index

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
# fazer a matriz de correlaoçao o heatmap,aplicar o modelo de relaçao linear
def readfile(file,cols):
    return pd.read_csv(file, usecols=cols)

dataf=readfile('pokemon.csv',cols=[1,2,3,4])

print(dataf)
# for i in dataf:
#     print(i,st.norm.interval(0.9, loc=dataf[i].mean(0),scale=dataf[i].std(0)))

# sns.displot(dataf,kind="kde")
# for i in dataf:
#     sns.displot(dataf[i],kind="kde")
# plt.show()

