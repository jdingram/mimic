import sys
import os
import boto3
import botocore
import pickle
import pandas as pd
import numpy as np


def from_s3(bucket, filepath, index_col=None):

    '''

    Function that pulls a file from a specified AWS S3 bucket.

    The AWS key and secret key must already be configured before this will
    run on any machine

    '''

    new_filename = filepath.split('/')[-1]
    s3 = boto3.client('s3')
    s3.download_file(bucket, filepath, new_filename)

    if filepath.split('.')[-1] == 'csv':
        obj = pd.read_csv(new_filename, index_col=index_col)
    elif filepath.split('.')[-1] == 'npy':
        obj = np.load(new_filename)
    else:
        new_filename = new_filename.split('.')[0]
        with open(new_filename, 'rb') as file:  
            obj = pickle.load(file)

    os.remove(new_filename)

    return obj


def to_s3(obj, bucket, filepath):

    '''

    Function that saves either a DataFrame, np array or trained model onto S3.

    The AWS key and secret key must already be configured before this will
    run on any machine

    '''

    s3 = boto3.client('s3')

    if type(obj) == pd.DataFrame:
        obj.to_csv('out_file.csv')
        s3.upload_file('out_file.csv', bucket, filepath)
        os.remove('out_file.csv')

    elif type(obj) == np.ndarray:
        np.save('out_file', obj)
        s3.upload_file('out_file.npy', bucket, filepath)
        os.remove('out_file.npy')
        
    else:
        with open('out_file.pkl', 'wb') as file:  
            pickle.dump(obj, file)
        s3.upload_file('out_file.pkl', bucket, filepath)
        os.remove('out_file.pkl')
        
        