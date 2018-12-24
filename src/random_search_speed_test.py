import sys
import boto3
import botocore
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

startTime = datetime.now()
print('---> START TIME: ', startTime)

### --- RUNS
iterations = int(sys.argv[1])
print('--->ITERATIONS:', iterations)

### --- FUNCTIONS

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
    
    # Impute missing values
    imputer = SimpleImputer(strategy = 'median')
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    
    # Scale each feature to have mean 0 and std dev of 1
    scaler = StandardScaler() 
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    

    if test:
        test = test.sample(frac=1).reset_index(drop=True)
        X_test = test.drop(columns=drop_cols)
        y_test = np.array(test[target].tolist())
        X_test = imputer.transform(X_test)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_train, y_test

    else:
        return X_train, y_train

def run_random_search(model, random_grid, scoring, cv, n_iter, X_train, y_train):
    
    # Use the random grid to search for best hyperparameters
    m = model
    print('--> Model defined')

    random_search_model = RandomizedSearchCV(estimator = m, scoring=scoring,
                                   param_distributions = random_grid,
                                   n_iter = n_iter, cv = cv, verbose=0,
                                   random_state=8, n_jobs = -1,
                                   return_train_score=True)
    print('--> Random search defined')

    # Fit the random search model
    random_search_model.fit(X_train, y_train)
    print('--> Fitting done')

    # Print the best CV score
    print('--> Best CV Score: ', random_search_model.best_score_)
    
    return random_search_model

### --- IMPORT DATA

s3 = boto3.resource('s3')
s3.Object('mimic-jamesi', 'acute_respiratory_failure_train.csv').download_file('acute_respiratory_failure_train.csv')
train = pd.read_csv('acute_respiratory_failure_train.csv', index_col=0)

X_train, y_train = final_cleaning(ids = ['subject_id', 'hadm_id'], target = 'target', train = train)
print('--> Cleaning done')

### --- RUN RF TEST

# define the grid search parameters
n_estimators = list(np.arange(20, 3000, 5))
max_features = list(np.arange(2, X_train.shape[1]))
max_depth = list(np.arange(1, 100))
max_depth.append(None)
min_samples_split = list(np.arange(2, 250))
min_samples_leaf = list(np.arange(1, 250))
bootstrap = [True, False]

# Create the random grid
rf_random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

print('--> Grid defined')

# Run the random search model
df = run_random_search(model=RandomForestRegressor(), random_grid=rf_random_grid, scoring='roc_auc', cv=4, n_iter=iterations, X_train=X_train, y_train=y_train)

print('DF LENGTH: ', len(pd.DataFrame(df.cv_results_)))

print('---> TOTAL TIME: ', datetime.now() - startTime)

