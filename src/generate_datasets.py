import os
import sys
import pandas as pd
import numpy as np

# Set up paths & import functions
project_root = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
src_folder = os.path.join(project_root, 'src')
sys.path.insert(0, src_folder)
from s3_storage import *
from utilities import *

def create_admission_diagnosis_table():

    '''

    This is intended as a single use function, designed to pull out the most important
    patient, admission and diagnosis data into a single dataset. Once this
    dataset is created, it can be used as the basis for later analysis without
    needing to use the raw data.

    The dataset includes:
      1) The most important patient demographic data, namely:
        a) subject_id
        b) Gender
        c) Date of birth
        d) Date of death (if applicable)
        e) Expire flag
      2) Admission data:
        a) Total admissions
        b) Admission number (so that a patient's first admission can be
           differentiated from their 2nd, 3rd, 4th etc admission)
        c) hadm_id
        d) Entry Diagnosis (From the admissions table. Not the definitive
           diagnosis, which would be pulled later from the diagnosis table, this
           is the preliminary diagnosis upon admission)
        e) Age on admission (either the actual age or if it has been obscured
           in the original database to comply with HIPAA then it is listed as
           89. See https://mimic.physionet.org/mimictables/patients/)
        f) Age on admission - bucketed (Same age as above but bucketed in the 
           ranges <45, 45-60, 60-75, 75-89, 89)
        g) Age on admission shifted (flag to indicate whether the age is the
           actual age or whether it has been obscured under HIPAA)
        h) admittime (Admission date and time)
        i) dischtime (Discharge date and time)
        j) deathtime (Death date and time - if applicable)
        k) Admission type (Elective, Urgent, Emergency, Newborn)
        l) Ethnicity (taken from admission table as this changes for some
           patients across admissions)
        m) Ethnicity simple (cleaned version of the above to reduce and
           simplify ethnicity buckets. For detailed analysis of ethnicity this
           shouldn't be used)
        n) Hospital expire flag
      3) Diagnosis data
        a) diagnosis_icd9 (1 row per diagnosis)
        b) diagnosis name

    Because there are new rows for each ADMISSION and DIAGNOSIS, this dataset is
    therefore at the DIAGNOSIS level

    '''

    # Get patient data and perform manual cleaning    
    patient_data = from_s3('mimic-jamesi', 'raw_data/PATIENTS.csv')
    patient_data = lowercase_columns(patient_data)
    patient_data = patient_data[['subject_id', 'dob', 'dod', 'gender', 'expire_flag']]
    patient_data = patient_data[~patient_data['subject_id'].isna()]
    patient_data = patient_data.drop_duplicates()
    patient_data['dob'] = pd.to_datetime(patient_data['dob'])
    patient_data['dod'] = pd.to_datetime(patient_data['dod'])
    
    
    # Get admission data and perform manual cleaning
    admission_data = from_s3('mimic-jamesi', 'raw_data/ADMISSIONS.csv')
    admission_data = lowercase_columns(admission_data)
    admission_data = admission_data[['subject_id', 'hadm_id', 'admittime', 'dischtime',
                                     'deathtime', 'admission_type', 'ethnicity',
                                     'hospital_expire_flag', 'diagnosis']]
    admission_data = admission_data.drop_duplicates()
    admission_data = admission_data[(~admission_data['subject_id'].isna()) &
                                    (~admission_data['hadm_id'].isna())]
    admission_data['admittime'] = pd.to_datetime(admission_data['admittime'])
    admission_data['dischtime'] = pd.to_datetime(admission_data['dischtime'])
    admission_data['deathtime'] = pd.to_datetime(admission_data['deathtime'])

    # Number of admissions per patient
    admission_data['total_admissions'] = (admission_data.groupby('subject_id')
                                                        .hadm_id
                                                        .transform('nunique'))

    # Add rolling admission count so a patient's 1st, 2nd, 3rd, last etc
    # admission can be identified
    admission_data.sort_values(by=['subject_id', 'admittime'],
                               ascending=[True, True], inplace=True)
    admission_data['admission_number'] = (admission_data.groupby('subject_id')
                                                       .cumcount() + 1)

    # Get diagnosis data and perform manual cleaning
    diagnoses = from_s3('mimic-jamesi', 'raw_data/DIAGNOSES_ICD.csv')
    diagnoses = lowercase_columns(diagnoses)
    diagnoses = diagnoses[['subject_id', 'hadm_id', 'icd9_code']]
    diagnoses = diagnoses.drop_duplicates()
    diagnoses = diagnoses[~diagnoses['icd9_code'].isnull()]

    # Get disgnosis names so they can be merged onto icd9_code
    diagnoses_n = from_s3('mimic-jamesi', 'raw_data/D_ICD_DIAGNOSES.csv')
    diagnoses_n = lowercase_columns(diagnoses_n)
    diagnoses_n = diagnoses_n[['icd9_code', 'short_title']]
    diagnoses_n = diagnoses_n.drop_duplicates()
    diagnoses = pd.merge(diagnoses, diagnoses_n,
                         how='left', left_on='icd9_code', right_on='icd9_code')

    # Merge admissions onto patients
    df = pd.merge(patient_data, admission_data,
                  how='left', left_on='subject_id', right_on='subject_id')

    # Calculate age at date of admission, setting all negative values to 89
    # (due to patients > 89 having age obscured)
    df['age_on_admission'] = (((df['admittime'] - df['dob']).dt.days)/365).astype(int)
    df['age_on_admission_shifted'] = np.where(df['age_on_admission'] < 0, 1, 0)
    df.loc[(df['age_on_admission'] < 0), 'age_on_admission'] = 89

    # Add age bucket for easier analysis
    df.loc[df['age_on_admission'] < 45, 'age_adm_bucket'] = '1. <45'
    df.loc[(df['age_on_admission'] >= 45) & (df['age_on_admission'] < 60),'age_adm_bucket'] = '2. 45-60'
    df.loc[(df['age_on_admission'] >= 60) & (df['age_on_admission'] < 75), 'age_adm_bucket'] = '3. 60-75'
    df.loc[(df['age_on_admission'] >= 75) & (df['age_on_admission'] < 89), 'age_adm_bucket'] = '4. 75-89'
    df.loc[df['age_on_admission'] == 89, 'age_adm_bucket'] = '5. 89'

    # Add simplified ethnicity field for easier analysis
    df.loc[df['ethnicity'].str.contains('WHITE'), 'ethnicity_simple'] = 'WHITE'
    df.loc[df['ethnicity'].str.contains('BLACK'), 'ethnicity_simple'] = 'BLACK'
    df.loc[df['ethnicity'].str.contains('HISPANIC'), 'ethnicity_simple'] = 'HISPANIC'
    df.loc[df['ethnicity'].str.contains('ASIAN'), 'ethnicity_simple'] = 'ASIAN'
    df.loc[df['ethnicity'].str.contains('UNABLE TO OBTAIN'), 'ethnicity_simple'] = 'UNABLE TO OBTAIN'
    df.loc[df['ethnicity'].str.contains('PATIENT DECLINED TO ANSWER'), 'ethnicity_simple'] = 'PATIENT DECLINED TO ANSWER'
    df.loc[df['ethnicity'].str.contains('UNKNOWN/NOT SPECIFIED'), 'ethnicity_simple'] = 'UNKNOWN/NOT SPECIFIED'
    df.loc[df['ethnicity_simple'].isna(), 'ethnicity_simple'] = 'OTHER'

    # Merge on diagnoses
    df = pd.merge(df, diagnoses, how='left',
                  left_on=['subject_id', 'hadm_id'],
                  right_on=['subject_id', 'hadm_id'])

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

    # Ensure the final output is clean by de-duping and reseting the index
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df