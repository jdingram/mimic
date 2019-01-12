import pandas as pd

def df_empty(columns, dtypes, index=None):

    '''
    Creates an empty pandas dataframe.
    
    The column names must be specified as a list in the 'columns'
    parameter, and the corresponding data types must be passed as a
    list into the 'dtypes' parameter.
    
    '''

    assert len(columns)==len(dtypes)
    df = pd.DataFrame(index=index)
    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df

def lowercase_columns(df):
    
    ''' Takes a Pandas DataFrame and makes the column names lowercase'''
    
    cols = df.columns
    cols_lower = [c.lower() for c in cols]
    df.columns = cols_lower
    return df