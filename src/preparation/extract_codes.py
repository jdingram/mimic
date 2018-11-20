'''
Functions that extract specific codes (items, NDC codes, etc) from specific dataframes
'''

import pandas as pd

def find_ndc_codes(df, name):

    '''
    Function that extracts NDC drug codes from a list of prescriptions
    '''
    
    print('==========')
    print("  {}  ".format(name))
    print('==========')
    
    # Fill missing NDC codes
    df.fillna(value = 'missing', inplace=True)
    print(str(df.loc[df['ndc'] != 'missing', 'drug'].count()) + ' out of ' + str(df['drug'].count()) + ' prescriptions have NDC codes')
    
    # Extract unique NDC codes
    ndc = df.loc[df['ndc'] != 'missing', 'ndc'].unique().tolist()
    print(str(len(ndc)) + ' unique NDC codes')
    print('----------')

    # Print the unique drugs that were present
    print(df['drug'].unique())
    print('----------')
    
    return ndc