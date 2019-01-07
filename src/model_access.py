import os
import sys
import pickle
import pandas as pd

def save_pickle(model, pickle_filename):

	# Save to file in the current working directory
	project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
	filename = os.path.join(project_root, 'models', "{}.pkl".format(pickle_filename)) 
	with open(filename, 'wb') as file:  
	    pickle.dump(model, file)


def open_pickle(pickle_filename):
	
	project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
	filename = os.path.join(project_root, 'models', "{}.pkl".format(pickle_filename))

	# Load from file
	with open(filename, 'rb') as file:  
		pickle_model = pickle.load(file)

	return pickle_model