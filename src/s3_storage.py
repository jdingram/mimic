import sys
import os
import boto3
import botocore
import pandas as pd
import numpy as np


def from_s3(bucket, filename, index_col=None):
	
	'''
	
	Function that pulls a file from a specified AWS S3 bucket, returning a 
	Pandas dataframe.

	The AWS key and secret key must already be configured before this will
	run on any machine

	'''
	
	s3 = boto3.resource('s3')
	s3.Object(bucket, filename).download_file(filename)
	
	if filename.split('.')[-1] == 'csv':
		obj = pd.read_csv(filename, index_col=index_col)
	elif filename.split('.')[-1] == 'npy':
		obj = np.load(filename)

	os.remove(filename)
	
	return obj


def to_s3(obj, bucket, filename):
	
	'''
	
	Function that saves a Pandas dataframe into a specified AWS S3 bucket.

	The AWS key and secret key must already be configured before this will
	run on any machine

	'''
	
	s3 = boto3.client('s3')
	
	if type(obj) == pd.DataFrame:
		obj.to_csv('out_file.csv')
		s3.upload_file('out_file.csv', bucket, filename)
		os.remove('out_file.csv')

	elif type(obj) == np.ndarray:
		np.save('out_file', obj)
		s3.upload_file('out_file.npy', bucket, filename)
		os.remove('out_file.npy')