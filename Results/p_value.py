import pandas as pd
import scipy.stats
import numpy as np

data = pd.read_csv('train_loss_old_datasets.csv')
data = data.iloc[:, 1:]
number_of_classifers = data.shape[1]
df = pd.DataFrame(np.zeros((number_of_classifers, number_of_classifers)))
df.columns = data.iloc[:, 0:data.shape[1]].columns.values
df.index = data.iloc[:, 0:data.shape[1]].columns.values
for i in range(0, data.shape[1]):
    for j in range(0, data.shape[1]):
        if i != j:
            df.iloc[i, j] = scipy.stats.wilcoxon(data.iloc[:, i], data.iloc[:, j], zero_method='wilcox')[
                1]  # pratt is conservative and wilcox is more standard

df.to_csv("wilcox_pval.csv")
