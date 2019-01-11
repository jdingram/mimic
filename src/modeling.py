import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer

# Set up paths & import src functions
project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
src_folder = os.path.join(project_root, 'src')
sys.path.insert(0, src_folder)
from s3_storage import *

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


def final_run(X_train, y_train, best_params, classifier, model_name):
    
    # Create the model
    model = classifier(**best_params)

    # Train the model
    model.fit(X_train, y_train)
    
    # Save model
    to_s3(obj=model, bucket='mimic-jamesi', filepath='models/{}'.format(model_name))
    