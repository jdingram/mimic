import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer

def final_cleaning(ids, target, train, test=None):

    if type(ids) == list:
        ids.append(target)
        drop_cols = ids.copy()
    else:
        drop_cols = [ids, target]
            
    # Shuffle
    train = train.sample(frac=1).reset_index(drop=True)
    
    # Split features and labels
    X_train = train.drop(columns=drop_cols)
    y_train = np.array(train[target].tolist())

    # Get feature names
    feature_names = np.array(list(X_train.columns))
    
    # Impute missing values
    imputer = Imputer(strategy = 'median')
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    
    # Scale each feature to have mean 0 and std dev of 1
    scaler = StandardScaler() 
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    

    if type(test) == pd.DataFrame:
        test = test.sample(frac=1).reset_index(drop=True)
        X_test = test.drop(columns=drop_cols)
        y_test = np.array(test[target].tolist())
        X_test = imputer.transform(X_test)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test, feature_names

    else:
        return X_train, y_train, feature_names
    