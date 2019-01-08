# Import libraries
import os
import sys
import pandas as pd
import numpy as np

# Set up paths
project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
src_folder = os.path.join(project_root, 'src')

# Import src functions
sys.path.insert(0, src_folder)
from s3_storage import *


def get_diagnosis_groups(diagnosis, optional_exclusions=None):

	'''

	The purpose of this function is for a given diagnosis, return 2 dataframes:
		1) subject_adm: This contains the patients who were diagnosed with
		   the condition
		2) base_adm: This contains all the other patients from the Mimic database
		   who were never diagnosed with the given condition.

	There are additional optional exlusions that can be applied. These should
	be passed as a list into the optional_exclusions argument. This argument can
	be omitted if no exclusions are needed:
		1) first_diagnosis_only - this means that only one admission per patient
		   will be included in the output dataframes. In the subject dataframe,
		   the first admission where they were diagnosed with the condition will
		   be included (not necessarily their first admission overall, if they 
		   were not diagnosed on their first admission)
		2) exclude_newborns - excludes all admissions with admission_type ==
		   'NEWBORN'
		3) exclude_deaths - excludes all admissions that resulted in the
		   patient dying

	'''

	# Import data
	admissions = from_s3(bucket='mimic-jamesi',
						 filename='admission_diagnosis_table.csv',
						 index_col=0)

	# ==== 1 ==== Find all patients diagnosed with the selected condition
	subject_adm = admissions[admissions['diagnosis_icd9'] == diagnosis]
	subject_adm.drop(columns=['diagnosis_name', 'diagnosis_icd9'], inplace=True)
	subject_adm.drop_duplicates(inplace=True)

	# Find list of subject ids so they can be excluded from the comparison group
	subject_ids = subject_adm.subject_id.unique().tolist()

	# ==== 2 ==== Find full potential comparison group
	base_adm = admissions[~admissions['subject_id'].isin(subject_ids)]
	base_adm.drop(columns=['diagnosis_name', 'diagnosis_icd9'], inplace=True)
	base_adm.drop_duplicates(inplace=True)

	# ==== 3 ==== Optional exclusions
	if optional_exclusions:

	    if 'first_diagnosis_only' in optional_exclusions:
	        subject_adm = (subject_adm.sort_values(by=['subject_id', 'admission_number'],
	        									   ascending=[True, True])
	                                  .groupby('subject_id')
	                                  .first()
	                                  .reset_index())
	        base_adm = (base_adm[base_adm['admission_number']==1])

	    if 'exclude_newborns' in optional_exclusions:
	        subject_adm = subject_adm[subject_adm['admission_type']!='NEWBORN']
	        base_adm = base_adm[base_adm['admission_type']!='NEWBORN']

	    if 'exclude_deaths' in optional_exclusions:
	        subject_adm = subject_adm[subject_adm['hospital_expire_flag']==0]
	        base_adm = base_adm[base_adm['hospital_expire_flag']==0]

	return subject_adm, base_adm



def add_chart_data(df):

    '''

    This function takes a given dataframe of admissions and merges on the final
    chart event readings for these admissions.

    The admission dataframe must include subject_id and hadm_id in order to
    identify the admissions

    '''

    readings = from_s3(bucket='mimic-jamesi',
                       filename='first_reading.csv',
                       index_col=0)

    df = pd.merge(df, readings,
                  how='left',
                  left_on=['subject_id', 'hadm_id'],
                  right_on=['subject_id', 'hadm_id'])

    return df



def add_profile_data(df, profile_data):

	'''

	This function takes a given dataframe of admissions and merges on additional
	profile data.

	The profile data comes from the admission_diagnosis_table dataset. The
	columns that are needed from this dataset must be passed as a list into the
	profile_data argument.

	The admission dataframe must include subject_id and hadm_id in order to
	identify the admissions.

	'''

	keep_cols = ['subject_id', 'hadm_id'] + profile_data

	admissions = from_s3(bucket='mimic-jamesi',
						 filename='admission_diagnosis_table.csv',
						 index_col=0)

	admissions = admissions[keep_cols]
	admissions.drop_duplicates(inplace=True)

	df = pd.merge(df, admissions,
		          how='left',
	              left_on=['subject_id', 'hadm_id'],
	              right_on=['subject_id', 'hadm_id'])

	return df


