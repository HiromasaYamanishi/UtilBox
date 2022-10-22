from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

def stratified_continuous(df, target_col, n_folds=5, q=20):
    '''
    stratify kfold for continuous target
    input: df[pandas.DataFrame], target_col[str], n_folds[Optional[int]], q[Optional[int]]
    output: df[pandas.DataFrame]
    '''
    X= df.loc[~df.columns.isin[target_col]]
    y = df.loc[:, target_col]
    y_cat = pd.qcut(y, q=q).codes
    fold = StratifiedKFold(n_splits=n_folds, random_state=71,shuffle=True)
    index = np.zeros(y.shape[0])
    for i,(X_ind, y_ind) in enumerate(fold.split(X, y_cat)):
        index[y_ind]=i
    df['valid'] = index
    return df

    a