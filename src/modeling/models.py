'''

Functions that train various models

'''

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

def train_lgb(X_train, X_test, y_train, y_test, n_folds, params, eval_metric, early_stopping_rounds):

    '''

    Trains a Light GBM, outputting the evaluation metrics

    '''
    
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = False, random_state = 50)
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(X_train.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []

    print('LGB starting')
        
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(X_train):
        
        # Training data for the fold
        train_features  = X_train[train_indices]
        train_labels = [x for i,x in enumerate(y_train) if i in train_indices]
        # Validation data for the fold
        valid_features = X_train[valid_indices]
        valid_labels = [x for i,x in enumerate(y_train) if i in valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(**params)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], early_stopping_rounds = early_stopping_rounds, verbose=500)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
    
    # Overall validation score
    valid_auc = roc_auc_score(y_train, out_of_fold)

    # Overall training score
    train_auc = np.mean(train_scores)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(train_auc)
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})
    
    print(metrics)

    # Find test score
    #predictions = #########
    #test_score = #########
    
    return metrics, train_auc, valid_auc