'''

Functions that select patients for modeling

'''

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Set up paths
src_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
src_preparation_folder = os.path.join(src_folder, 'preparation')
src_processing_folder = os.path.join(src_folder, 'processing')
src_modeling_folder = os.path.join(src_folder, 'modeling')

sys.path.insert(0, src_processing_folder)
from stats import plot_KDE
from stats import plot_perc_bar_chart
from stats import compare_groups
from stats import graph_comparisons


def select_test_groups(diagnosis, first_diagnosis_only, exclude_newborns, exclude_deaths, match_control, show_stats_graphs):

    '''

    Function that for a given dignosis selects the patients who were diagnosed with the condition along with an appropriate base group

    '''

    # Import data
    admissions = pd.read_csv(os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'data', 'admission_diagnosis_table.csv')), index_col=0)
    
    keep_cols = ['subject_id', 'gender', 'expire_flag', 'total_admissions', 'admission_number', 'hadm_id',
                'age_on_admission', 'age_adm_bucket', 'admission_type', 'ethnicity_simple', 'hospital_expire_flag']

    # ==== 1 ==== Find all patients diagnosed with the selected condition
    subject_adm = admissions[admissions['diagnosis_icd9'] == diagnosis]

    # Find list of subject ids
    subject_ids = subject_adm.subject_id.unique().tolist()

    # ==== 2 ==== Find full potential comparison group
    base_adm = admissions[~admissions['subject_id'].isin(subject_ids)]

    # -- Optional exclusions
    if first_diagnosis_only:
        subject_adm = (subject_adm.sort_values(by=['subject_id', 'admission_number'], ascending=[True, True])
                                  .groupby('subject_id')
                                  .first().reset_index())
        base_adm = (base_adm[base_adm['admission_number']==1])

    if exclude_newborns:
        subject_adm = subject_adm[subject_adm['admission_type']!='NEWBORN']
        base_adm = base_adm[base_adm['admission_type']!='NEWBORN']

    if exclude_deaths:
        subject_adm = subject_adm[subject_adm['hospital_expire_flag']==0]
        base_adm = base_adm[base_adm['hospital_expire_flag']==0]

    # Keep only necessary cols and de-dupe
    subject_adm = subject_adm[keep_cols].drop_duplicates()
    base_adm = base_adm[keep_cols].drop_duplicates()

    if match_control:
        # Subjects
        subject_segments = (subject_adm.groupby(['age_adm_bucket', 'gender'])
                                       .agg({'hadm_id':'nunique'})
                                       .rename(columns={'hadm_id':'subjects_n'})
                                       .reset_index())
        subject_segments['subjects_prop'] = subject_segments['subjects_n'] / subject_segments['subjects_n'].sum()

        # Base
        base_segments = (base_adm.groupby(['age_adm_bucket', 'gender'])
                                 .agg({'hadm_id':'nunique'})
                                 .rename(columns={'hadm_id':'base_n'})
                                 .reset_index())
        base_segments['base_prop'] = base_segments['base_n'] / base_segments['base_n'].sum()

        proportions_compare = pd.merge(subject_segments, base_segments, how='outer',
                                       left_on=['age_adm_bucket', 'gender'], right_on=['age_adm_bucket', 'gender'])

        # Compare
        proportions_compare['ratio'] = proportions_compare['base_prop'] / proportions_compare['subjects_prop']

        # Find min ratio
        lowest = proportions_compare.loc[proportions_compare['ratio'] == proportions_compare['ratio'].min()]
        total_sample_size = math.floor(lowest['base_n'] / lowest['subjects_prop'])

        proportions_compare['new_base_grp_size'] = (total_sample_size * proportions_compare['subjects_prop']).apply(np.floor)

        base_adm_sampled = pd.DataFrame(columns = base_adm.columns.tolist())

        for idx,row in proportions_compare.iterrows():
            age = row['age_adm_bucket']
            gender = row['gender']
            n = int(row['new_base_grp_size'])

            sample_df = base_adm[(base_adm['age_adm_bucket']==age) & (base_adm['gender']==gender)].sample(n=n, random_state=8)

            base_adm_sampled = base_adm_sampled.append(sample_df)

        print('Original base group size: ' + str(len(base_adm)))
        print('Sampled base group size: ' + str(len(base_adm_sampled)))
        print('Subject group size: ' + str(len(subject_adm)))

    # Combine into a single DF
    subject_adm['target'] = 1
    base_adm_sampled['target'] = 0
    df = subject_adm.append(base_adm_sampled).reset_index(drop=True)

    if show_stats_graphs:
        graph_comparisons(df = df, group_col = 'target', group_a = 1, group_b = 0)

    df['subject_id'] = df['subject_id'].astype(int)
    df['hadm_id'] = df['hadm_id'].astype(int)

    return df