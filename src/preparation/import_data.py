'''

This module is for importing the data

'''

import pandas as pd
import numpy as np
import psycopg2 as p


def get_table(host, dbname, schema, table, columns, where='row_id>0'):

	'''
	Function for importing the Mimic data from a Postgres database.

	'''

	# Select DB location
	con = p.connect("host={} dbname={}".format(host, dbname))
	cur = con.cursor()

	# Execute query
	cur.execute("SELECT {} FROM {}.{} WHERE {}".format(columns, schema, table, where))

	if cur.rowcount > 0:
		# Get rows and column names
		rows=cur.fetchall()
		column_names = [desc[0] for desc in cur.description]

		# Create DataFrame
		df = pd.DataFrame(rows)
		df.columns = column_names

		# De-Dupe
		df.drop_duplicates(inplace=True)

		return df
	else:
		print('No rows returned')


def get_data_simple(query):
	# Select DB location
	con = p.connect("host=localhost dbname=mimic")
	cur = con.cursor()

	# Execute query
	cur.execute(query)

	# Get rows and column names
	rows=cur.fetchall()
	column_names = [desc[0] for desc in cur.description]

	# Create DataFrame
	df = pd.DataFrame(rows)
	df.columns = column_names

	# De-Dupe
	df.drop_duplicates(inplace=True)
	return df


def get_patient_admissions_diagnoses(subjects):

	'''

	Function that pulls all patient, admission and diagnosis data for a list of subjects

	'''

	# Get patient data
	s = tuple(set(subjects))
	patient_data = get_table(host = 'localhost',
	                              dbname = 'mimic',
	                              schema = 'mimiciii',
	                              table = 'patients',
	                              columns = 'DISTINCT subject_id, dob, dod, gender, expire_flag',
	                              where = "subject_id IN {}".format(s))

	# Get admission data
	admission_data = get_table(host = 'localhost',
	                              dbname = 'mimic',
	                              schema = 'mimiciii',
	                              table = 'admissions',
	                              columns = 'DISTINCT subject_id, hadm_id, admittime, dischtime, deathtime, admission_type,\
	                                         ethnicity, hospital_expire_flag, diagnosis',
	                              where = "subject_id IN {}".format(s))

	# Number of admissions per patient
	admission_data['total_admissions'] = admission_data.groupby('subject_id').hadm_id.transform('nunique')

	# Add a rolling admission count
	admission_data.sort_values(by=['subject_id', 'admittime'], ascending=[True, True], inplace=True)
	admission_data['admission_number'] = admission_data.groupby('subject_id').cumcount() + 1

	# Get diagnoses
	a = tuple(set(admission_data.hadm_id.tolist()))
	diagnoses = get_table(host = 'localhost',
	                      dbname = 'mimic',
	                      schema = 'mimiciii',
	                      table = 'diagnoses_icd',
	                      columns = 'DISTINCT subject_id, hadm_id, icd9_code',
	                      where = "hadm_id IN {} AND hadm_id > 0".format(a))

	# Drop disgnoses where icd9_code is null
	diagnoses = diagnoses[~diagnoses['icd9_code'].isnull()]

	# Get disgnosis names
	d = tuple(set(diagnoses.icd9_code.tolist()))
	diagnoses_n = get_table(host = 'localhost',
	                          dbname = 'mimic',
	                          schema = 'mimiciii',
	                          table = 'd_icd_diagnoses',
	                          columns = 'DISTINCT icd9_code, short_title',
	                          where = "icd9_code IN {}".format(d))

	# Merge diagnosis names onto icd9 codes
	diagnoses = pd.merge(diagnoses, diagnoses_n, how='left', left_on='icd9_code', right_on='icd9_code')

	# Merge admissions onto patients
	df = pd.merge(patient_data, admission_data, how='left', left_on='subject_id', right_on='subject_id')

	# Calculate age at date of admission, setting all negative values to 89 (due to patients > 89 having age obscured)
	df['age_on_admission'] = (((df['admittime'] - df['dob']).dt.days)/365).astype(int)
	df['age_on_admission_shifted'] = np.where(df['age_on_admission'] < 0, 1, 0)
	df.loc[(df['age_on_admission'] < 0), 'age_on_admission'] = 89

	# Add age bucket
	df.loc[df['age_on_admission'] < 45, 'age_adm_bucket'] = '1. <45'
	df.loc[(df['age_on_admission'] >= 45) & (df['age_on_admission'] < 60), 'age_adm_bucket'] = '2. 45-60'
	df.loc[(df['age_on_admission'] >= 60) & (df['age_on_admission'] < 75), 'age_adm_bucket'] = '3. 60-75'
	df.loc[(df['age_on_admission'] >= 75) & (df['age_on_admission'] < 89), 'age_adm_bucket'] = '4. 75-89'
	df.loc[df['age_on_admission'] == 89, 'age_adm_bucket'] = '5. 89'

	# Add simplified ethnicity field
	df.loc[df['ethnicity'].str.contains('WHITE'), 'ethnicity_simple'] = 'WHITE'
	df.loc[df['ethnicity'].str.contains('BLACK'), 'ethnicity_simple'] = 'BLACK'
	df.loc[df['ethnicity'].str.contains('HISPANIC'), 'ethnicity_simple'] = 'HISPANIC'
	df.loc[df['ethnicity'].str.contains('ASIAN'), 'ethnicity_simple'] = 'ASIAN'
	df.loc[df['ethnicity'].str.contains('UNABLE TO OBTAIN'), 'ethnicity_simple'] = 'UNABLE TO OBTAIN'
	df.loc[df['ethnicity'].str.contains('PATIENT DECLINED TO ANSWER'), 'ethnicity_simple'] = 'PATIENT DECLINED TO ANSWER'
	df.loc[df['ethnicity'].str.contains('UNKNOWN/NOT SPECIFIED'), 'ethnicity_simple'] = 'UNKNOWN/NOT SPECIFIED'
	df.loc[df['ethnicity_simple'].isna(), 'ethnicity_simple'] = 'OTHER'

	# Merge on diagnoses
	df = pd.merge(df, diagnoses, how='left', left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])

	# Rename columns
	df.rename(columns={'diagnosis': 'entry_diagnosis',
	                  'icd9_code': 'diagnosis_icd9',
	                  'short_title': 'diagnosis_name'}, inplace=True)

	# Re-order columns
	ordered_columns = ['subject_id',
	                  'gender',
	                  'dob',
	                  'dod',
	                  'expire_flag',
	                  'total_admissions',
	                  'admission_number',
	                  'hadm_id',
	                  'entry_diagnosis',
	                  'age_on_admission',
	                  'age_adm_bucket',
	                  'age_on_admission_shifted',
	                  'admittime',
	                  'dischtime',
	                  'deathtime',
	                  'admission_type',
	                  'ethnicity',
	                  'ethnicity_simple',
	                  'hospital_expire_flag',
	                  'diagnosis_icd9',
	                  'diagnosis_name']

	df = df[ordered_columns]

	return df

def get_admission_data(admissions):

	'''

	Function that pulls patient, admission and diagnosis data for a list of admissions
	Data accurate at time of admission

	'''

	# Get patient data
	a = tuple(set(admissions))
	s = get_table(host = 'localhost',
	          dbname = 'mimic',
	          schema = 'mimiciii',
	          table = 'admissions',
	          columns = 'DISTINCT subject_id',
	          where = "hadm_id IN {}".format(a)).subject_id.tolist()

	s = tuple(set(s))

	patient_data = get_table(host = 'localhost',
	                              dbname = 'mimic',
	                              schema = 'mimiciii',
	                              table = 'patients',
	                              columns = 'DISTINCT subject_id, dob, gender',
	                              where = "subject_id IN {}".format(s))

	# Get admission data
	admission_data = get_table(host = 'localhost',
	                              dbname = 'mimic',
	                              schema = 'mimiciii',
	                              table = 'admissions',
	                              columns = 'DISTINCT subject_id, hadm_id, admittime, dischtime, deathtime, admission_type,\
	                                         ethnicity, hospital_expire_flag, diagnosis',
	                              where = "hadm_id IN {}".format(a))

	# Get diagnoses
	diagnoses = get_table(host = 'localhost',
	                      dbname = 'mimic',
	                      schema = 'mimiciii',
	                      table = 'diagnoses_icd',
	                      columns = 'DISTINCT subject_id, hadm_id, icd9_code',
	                      where = "hadm_id IN {} AND hadm_id > 0".format(a))

	# Drop disgnoses where icd9_code is null
	diagnoses = diagnoses[~diagnoses['icd9_code'].isnull()]

	# Get disgnosis names
	d = tuple(set(diagnoses.icd9_code.tolist()))
	diagnoses_n = get_table(host = 'localhost',
	                          dbname = 'mimic',
	                          schema = 'mimiciii',
	                          table = 'd_icd_diagnoses',
	                          columns = 'DISTINCT icd9_code, short_title',
	                          where = "icd9_code IN {}".format(d))

	# Merge diagnosis names onto icd9 codes
	diagnoses = pd.merge(diagnoses, diagnoses_n, how='left', left_on='icd9_code', right_on='icd9_code')

	# Merge admissions onto patients
	df = pd.merge(patient_data, admission_data, how='left', left_on='subject_id', right_on='subject_id')

	# Calculate age at date of admission, setting all negative values to 89 (due to patients > 89 having age obscured)
	df['age_on_admission'] = (((df['admittime'] - df['dob']).dt.days)/365).astype(int)
	df['age_on_admission_shifted'] = np.where(df['age_on_admission'] < 0, 1, 0)
	df.loc[(df['age_on_admission'] < 0), 'age_on_admission'] = 89

	# Add age bucket
	df.loc[df['age_on_admission'] < 45, 'age_adm_bucket'] = '1. <45'
	df.loc[(df['age_on_admission'] >= 45) & (df['age_on_admission'] < 60), 'age_adm_bucket'] = '2. 45-60'
	df.loc[(df['age_on_admission'] >= 60) & (df['age_on_admission'] < 75), 'age_adm_bucket'] = '3. 60-75'
	df.loc[(df['age_on_admission'] >= 75) & (df['age_on_admission'] < 89), 'age_adm_bucket'] = '4. 75-89'
	df.loc[df['age_on_admission'] == 89, 'age_adm_bucket'] = '5. 89'

	# Add simplified ethnicity field
	df.loc[df['ethnicity'].str.contains('WHITE'), 'ethnicity_simple'] = 'WHITE'
	df.loc[df['ethnicity'].str.contains('BLACK'), 'ethnicity_simple'] = 'BLACK'
	df.loc[df['ethnicity'].str.contains('HISPANIC'), 'ethnicity_simple'] = 'HISPANIC'
	df.loc[df['ethnicity'].str.contains('ASIAN'), 'ethnicity_simple'] = 'ASIAN'
	df.loc[df['ethnicity'].str.contains('UNABLE TO OBTAIN'), 'ethnicity_simple'] = 'UNABLE TO OBTAIN'
	df.loc[df['ethnicity'].str.contains('PATIENT DECLINED TO ANSWER'), 'ethnicity_simple'] = 'PATIENT DECLINED TO ANSWER'
	df.loc[df['ethnicity'].str.contains('UNKNOWN/NOT SPECIFIED'), 'ethnicity_simple'] = 'UNKNOWN/NOT SPECIFIED'
	df.loc[df['ethnicity_simple'].isna(), 'ethnicity_simple'] = 'OTHER'

	# Merge on diagnoses
	df = pd.merge(df, diagnoses, how='left', left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])

	# Rename columns
	df.rename(columns={'diagnosis': 'entry_diagnosis',
	                  'icd9_code': 'diagnosis_icd9',
	                  'short_title': 'diagnosis_name'}, inplace=True)

	# Re-order columns
	ordered_columns = ['subject_id',
	                  'gender',
	                  'dob',
	                  'hadm_id',
	                  'entry_diagnosis',
	                  'age_on_admission',
	                  'age_adm_bucket',
	                  'age_on_admission_shifted',
	                  'admittime',
	                  'dischtime',
	                  'deathtime',
	                  'admission_type',
	                  'ethnicity',
	                  'ethnicity_simple',
	                  'hospital_expire_flag',
	                  'diagnosis_icd9',
	                  'diagnosis_name']

	df = df[ordered_columns]

	return df

	
def get_chartevents(host, dbname, admissions, reading):

	'''
	Function for importing chartevents for a list of admissions

	'''

	# Select DB location
	con = p.connect("host={} dbname={}".format(host, dbname))
	cur = con.cursor()

	admissions=tuple(set(admissions))

	# Execute query
	cur.execute("SELECT\
	                a.subject_id\
	                ,a.hadm_id\
	                ,a.itemid\
	                ,b.valuenum\
	            FROM\
	                ((SELECT subject_id\
	                        ,hadm_id\
	                        ,itemid\
	                        ,{}(charttime) AS reading_time\
	                FROM mimiciii.chartevents\
	                WHERE hadm_id IN {}\
	                GROUP BY 1,2,3) a\
	                LEFT JOIN\
	                (SELECT hadm_id\
	                        ,itemid\
	                        ,charttime\
	                        ,valuenum\
	                FROM mimiciii.chartevents) b\
	                ON a.hadm_id = b.hadm_id AND a.itemid = b.itemid AND a.reading_time = b.charttime)\
	            WHERE valuenum IS NOT null".format(reading, admissions))

	if cur.rowcount > 0:
	    # Get rows and column names
	    rows=cur.fetchall()
	    column_names = [desc[0] for desc in cur.description]

	    # Create DataFrame
	    df = pd.DataFrame(rows)
	    df.columns = column_names

	    # De-Dupe
	    df.drop_duplicates(inplace=True)

	    return df
	else:
	    print('No rows returned')


def get_labevents(host, dbname, admissions, reading):

	'''
	Function for importing labevents for a list of admissions

	'''

	# Select DB location
	con = p.connect("host={} dbname={}".format(host, dbname))
	cur = con.cursor()

	admissions=tuple(set(admissions))

	# Execute query
	cur.execute("SELECT\
	                a.subject_id\
	                ,a.hadm_id\
	                ,a.itemid\
	                ,b.valuenum\
	            FROM\
	                ((SELECT subject_id\
	                        ,hadm_id\
	                        ,itemid\
	                        ,{}(charttime) AS reading_time\
	                FROM mimiciii.labevents\
	                WHERE hadm_id IN {}\
	                GROUP BY 1,2,3) a\
	                LEFT JOIN\
	                (SELECT hadm_id\
	                        ,itemid\
	                        ,charttime\
	                        ,valuenum\
	                FROM mimiciii.labevents) b\
	                ON a.hadm_id = b.hadm_id AND a.itemid = b.itemid AND a.reading_time = b.charttime)\
	            WHERE valuenum IS NOT null".format(reading, admissions))

	if cur.rowcount > 0:
	    # Get rows and column names
	    rows=cur.fetchall()
	    column_names = [desc[0] for desc in cur.description]

	    # Create DataFrame
	    df = pd.DataFrame(rows)
	    df.columns = column_names

	    # De-Dupe
	    df.drop_duplicates(inplace=True)

	    return df
	else:
	    print('No rows returned')
