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
    
    '''
    
    The purpose of this function is to create X and y Numpy arrays from
    the training and test Pandas DataFrames.
    
    Several operations are performed:
        1. The target variable is removed from the training and test DataFrames
           and saved in separate arrays (y) to comply with the format used in 
           Machine Learning models. The IDs are also removed as these are randomly
           generated and unique therefore would be detrimental for creating
           generalised classification models.
        2. Missing values are imputed (using Median values) in order for the ML
           models to work correctly. Median is used as the strategy as it is less
           affected by outliers as the default 'Mean'.
        3. The features are scaled so that the mean is 0 with a standard deviation
           of 1. While this isn't needed for all models, it improves the training
           of certain models, particularly Neural Networks
     
     Parameters:
        1. ids - list of the IDs in the input DataFrames (eg, 'subject_id')
        2. target - the name of the target variable
        3. train - the training DataFrame
        4. test - the test DataFrame (optional)
     
     The outputs are as follows:
        1. X_train - feature training set
        2. X_test - feature test set
        3. y_train - target variable for the training set
        4. y_test - target variable for the test set
        5. feature_names - the features from the feature set
        
    '''

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
    

    # Apply the above operations on the test DataFrame
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
    
    '''
    
    The purpose of this function is to train a ML model with a given
    set of hyperparameters, and save the resulting model in AWS S3 as a
    Pickle file. It is intended to be the final training run, after the
    optimal hyperparameters have been found.
    
    Parameters:
        1. X_train - feature training set
        2. y_train - the target variable for the training set
        3. best_params - (dict) the hyperparameters that will be used
           for the model training
        4. classifier - the name of the ML model/ library that will be trained
        5. model_name - used as the name when saving the trained model on S3
    
    '''
    
    # Create the model and fit with the chosen hyperparameters
    model = classifier(**best_params)
    model.fit(X_train, y_train)
    
    # Save model to S3
    to_s3(obj=model,
          bucket='mimic-jamesi',
          filepath='models/{}'.format(model_name))
    
    