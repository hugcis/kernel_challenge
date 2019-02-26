import numpy as np
import pandas as pd
from os.path import join

np.random.seed(333)

DATA_DIR = 'data'

X_train = pd.concat(
    [pd.read_csv(join(DATA_DIR, 'Xtr{}.csv'.format(i)), 
                 index_col=0) for i in range(3)]
).reset_index(drop=True)

Y_train = pd.concat(
    [pd.read_csv(join(DATA_DIR, 'Ytr{}.csv'.format(i)), 
                 index_col=0) for i in range(3)]
).reset_index(drop=True)

X_train_mat = pd.concat([
    pd.read_csv(join(DATA_DIR, 'Xtr{}_mat100.csv'.format(i)), 
                delimiter=' ', 
                header=None) 
    for i in range(3)
]).reset_index(drop=True)

val_index = np.random.choice(range(X_train.shape[0]), 
                             size=int(0.2*X_train.shape[0]), 
                             replace=False)
train_index = [i for i in range(X_train.shape[0]) if i not in val_index]


X_train.reindex(val_index, ).to_csv(join(DATA_DIR, 'X_val.csv'))
X_train.reindex(train_index).to_csv(join(DATA_DIR, 'X_train.csv'))

Y_train.reindex(val_index).to_csv(join(DATA_DIR, 'Y_val.csv'))
Y_train.reindex(train_index).to_csv(join(DATA_DIR, 'Y_train.csv'))

X_train_mat.reindex(val_index).to_csv(join(DATA_DIR, 'X_val_mat.csv'))
X_train_mat.reindex(train_index).to_csv(join(DATA_DIR, 'X_train_mat.csv'))
