import sys
import os
import boto3
import botocore
import pandas as pd


def from_s3(bucket, filename, index_col=None):
	
	'''
	
	Function that pulls a file from a specified AWS S3 bucket, returning a 
	Pandas dataframe.

	The AWS key and secret key must already be configured before this will
	run on any machine

	'''
	
	s3 = boto3.resource('s3')
	s3.Object(bucket, filename).download_file(filename)
	df = pd.read_csv(filename, index_col=index_col)
	os.remove(filename)
	
	return df


def to_s3(df, bucket, filename):
	
	'''
	
	Function that saves a Pandas dataframe into a specified AWS S3 bucket.

	The AWS key and secret key must already be configured before this will
	run on any machine

	'''
	
	s3 = boto3.client('s3')
	df.to_csv('out_file.csv')
	s3.upload_file('out_file.csv', bucket, filename)
	os.remove('out_file.csv')