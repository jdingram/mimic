'''

Funcitons that create some common datasets, saving them as csv files in the data folder

'''

# Import libraries
import os
import sys

import pandas as pd
import numpy as np
import psycopg2 as p

# Set up paths
project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
src_folder = os.path.join(project_root, 'src')

src_preparation_folder = os.path.join(src_folder, 'preparation')

# Import src functions
sys.path.insert(0, src_preparation_folder)
from import_data import get_data_simple


def create_chart_and_lab_table(reading):

    # Lab events
    lab = get_data_simple(query = "SELECT\
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
                                    GROUP BY 1,2,3) a\
                                    LEFT JOIN\
                                    (SELECT hadm_id\
                                            ,itemid\
                                            ,charttime\
                                            ,valuenum\
                                    FROM mimiciii.labevents) b\
                                    ON a.hadm_id = b.hadm_id AND a.itemid = b.itemid AND a.reading_time = b.charttime)\
                                WHERE valuenum IS NOT null".format(reading))

    lab_lookup = get_data_simple(query = "SELECT DISTINCT\
                                                itemid\
                                                ,label\
                                      FROM mimiciii.d_labitems")

    lab_lookup['dbsource'] = 'lab'

    lab_df = pd.merge(lab, lab_lookup, left_on='itemid', right_on='itemid', how='left')

    # Chart events
    chart = get_data_simple(query = "SELECT\
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
                                    GROUP BY 1,2,3) a\
                                    LEFT JOIN\
                                    (SELECT hadm_id\
                                            ,itemid\
                                            ,charttime\
                                            ,valuenum\
                                    FROM mimiciii.chartevents) b\
                                    ON a.hadm_id = b.hadm_id AND a.itemid = b.itemid AND a.reading_time = b.charttime)\
                                WHERE valuenum IS NOT null".format(reading))

    chart_lookup = get_data_simple(query = "SELECT DISTINCT\
                                                itemid\
                                                ,label\
                                                ,dbsource\
                                      FROM mimiciii.d_items")

    chart_df = pd.merge(chart, chart_lookup, left_on='itemid', right_on='itemid', how='left')

    # Append
    df = pd.DataFrame(columns=['subject_id', 'hadm_id', 'itemid', 'valuenum', 'label', 'dbsource'])
    df = df.append(chart_df)
    df = df.append(lab_df)

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df


def create_admission_diagnosis_table():

    '''

    Creates the summary dataset of all admissions and diagnoses, saving the output as a csv file in the data folder

    '''

    # Get patient data
    patient_data = get_data_simple(query = "SELECT DISTINCT subject_id, dob, dod, gender, expire_flag\
                                            FROM mimiciii.patients")

    # Get admission data
    admission_data = get_data_simple(query = "SELECT DISTINCT subject_id, hadm_id, admittime, dischtime,\
                                                              deathtime, admission_type, ethnicity,\
                                                              hospital_expire_flag, diagnosis\
                                            FROM mimiciii.admissions")

    # Number of admissions per patient
    admission_data['total_admissions'] = admission_data.groupby('subject_id').hadm_id.transform('nunique')

    # Add a rolling admission count
    admission_data.sort_values(by=['subject_id', 'admittime'], ascending=[True, True], inplace=True)
    admission_data['admission_number'] = admission_data.groupby('subject_id').cumcount() + 1

    # Get diagnosis data
    diagnoses = get_data_simple(query = "SELECT DISTINCT subject_id, hadm_id, icd9_code\
                                         FROM mimiciii.diagnoses_icd")

    # Drop disgnoses where icd9_code is null
    diagnoses = diagnoses[~diagnoses['icd9_code'].isnull()]

    # Get disgnosis names
    diagnoses_n = get_data_simple(query = "SELECT DISTINCT icd9_code, short_title\
                                          FROM mimiciii.d_icd_diagnoses")

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

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df