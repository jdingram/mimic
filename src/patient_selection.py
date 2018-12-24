# Import libraries
import os
import sys
import pandas as pd
import numpy as np


def get_diagnosis_groups(diagnosis, optional_exclusions=False):

	# Import data
	admissions = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data', 'admission_diagnosis_table.csv')), index_col=0)

	# ==== 1 ==== Find all patients diagnosed with the selected condition
	subject_adm = admissions[admissions['diagnosis_icd9'] == diagnosis]
	subject_adm.drop(columns=['diagnosis_name', 'diagnosis_icd9'], inplace=True)
	subject_adm.drop_duplicates(inplace=True)

	# Find list of subject ids
	subject_ids = subject_adm.subject_id.unique().tolist()

	# ==== 2 ==== Find full potential comparison group
	base_adm = admissions[~admissions['subject_id'].isin(subject_ids)]
	base_adm.drop(columns=['diagnosis_name', 'diagnosis_icd9'], inplace=True)
	base_adm.drop_duplicates(inplace=True)

	# ==== 3 ==== Optional exclusions
	if optional_exclusions:

	    if 'first_diagnosis_only' in optional_exclusions:
	        subject_adm = (subject_adm.sort_values(by=['subject_id', 'admission_number'], ascending=[True, True])
	                                  .groupby('subject_id')
	                                  .first().reset_index())
	        base_adm = (base_adm[base_adm['admission_number']==1])

	    if 'exclude_newborns' in optional_exclusions:
	        subject_adm = subject_adm[subject_adm['admission_type']!='NEWBORN']
	        base_adm = base_adm[base_adm['admission_type']!='NEWBORN']

	    if 'exclude_deaths' in optional_exclusions:
	        subject_adm = subject_adm[subject_adm['hospital_expire_flag']==0]
	        base_adm = base_adm[base_adm['hospital_expire_flag']==0]

	return subject_adm, base_adm



def add_chart_data(df):

	readings = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(),os.pardir,'data','admission_final_readings.csv'))
	                       ,index_col=0)

	df = pd.merge(df, readings, how='left',
	              left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])

	return df



def add_profile_data(df, profile_data):
    
    keep_cols = ['subject_id', 'hadm_id'] + profile_data
    
    data = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(),os.pardir,'data','admission_diagnosis_table.csv')),
                       index_col=0)
    
    data = data.loc[:, keep_cols]
    data.drop_duplicates(inplace=True)
    
    
    df = pd.merge(df, data, how='left',
              left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])
    
    return df