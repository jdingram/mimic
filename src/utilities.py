import pandas as pd

def df_empty(columns, dtypes, index=None):

	'''creates an empty pandas dataframe'''
	
	assert len(columns)==len(dtypes)
	df = pd.DataFrame(index=index)
	for c,d in zip(columns, dtypes):
	    df[c] = pd.Series(dtype=d)
	return df