
from asyncio.windows_events import NULL
from matplotlib import pyplot as plt
from operator import index

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
def readfile(file,cols):
    if cols!=NULL:
        return pd.read_csv(file, usecols=cols)
    else:
        return pd.read_csv(file)

dataf=readfile('./projeto/assets/winequality-red.csv',cols=[0,1,2,3,4,5,6,7,8,9,10,11])

print(dataf)
print(dataf.corr())

heatmap=sns.heatmap(dataf.corr(),annot=True,cmap="Greens")
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
plt.show()
