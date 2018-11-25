'''

Group of functions that compares populations, creates visualisations and computes significance tests

'''

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
src_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
src_preparation_folder = os.path.join(src_folder, 'preparation')

from import_data import get_admission_data


def plot_KDE(df, group_by, group_a, group_b, feature):

    '''

    Plots a KDE for two groups of patients for a single continuous feature

    '''

    plt.figure(figsize = (7, 5))

    #Remove NaNs
    df = df[[group_by, feature]]
    df.dropna(inplace=True)

    # KDE plot of patients who survived
    sns.kdeplot(df.loc[df[group_by] == group_a, feature], label = group_a)

    # KDE plot of patients who died
    sns.kdeplot(df.loc[df[group_by] == group_b, feature], label = group_b)

    # Labeling of plot
    plt.xlabel(feature); plt.ylabel('Density'); plt.title(feature);

    plt.show()



def plot_perc_bar_chart(df, group_by, group_a, group_b, feature, value):

    '''

    Plots bar charts for discrete variables, comparing the proportion of 2 populations falling into each category

    '''

    plt.figure(figsize = (7, 5))

    # Count number of unique subjects in the subject and base group, then work out % of group totals
    t = df.groupby([group_by, feature]).agg({value: 'nunique'}).reset_index()
    t['tot'] = t.groupby(group_by).subject_id.transform('sum')
    t['perc'] = t[value] / t['tot']

    # Plot
    sns.barplot(data=t, x=feature, y="perc", hue=group_by)
    plt.xticks(rotation='vertical');
    plt.show()


def compare_groups(subjects, base):

    '''

    Funciton that takes 2 lists of admissions and compares age, gender, ethnicity and likelihood of death for the two groups

    '''

    # --- Get patient, admission and diagnosis data for these admissions
    subject_admission_data = get_admission_data(subjects)
    base_admission_data = get_admission_data(base)

    # Add flags for subject or base group
    subject_admission_data['group'] = 'subject'
    base_admission_data['group'] = 'base'

    # Concatenate into 1 DF
    df = subject_admission_data.append(base_admission_data)
    del subject_admission_data, base_admission_data

    # Keep only necessary columns
    df = df[['subject_id', 'group', 'gender', 'age_on_admission', 'age_adm_bucket', 'ethnicity', 'ethnicity_simple',\
             'diagnosis_name', 'hospital_expire_flag']]

    # KDE for age
    plot_KDE(df = df,
             group_by = 'group',
             group_a = 'subject',
             group_b = 'base',
             feature = 'age_on_admission')

    # Bar chart for age buckets
    plot_perc_bar_chart(df = df,
                        group_by = 'group',
                        group_a = 'subject',
                        group_b = 'base',
                        feature = 'age_adm_bucket',
                        value = 'subject_id')

    # Bar chart for Gender
    plot_perc_bar_chart(df = df,
                        group_by = 'group',
                        group_a = 'subject',
                        group_b = 'base',
                        feature = 'gender',
                        value = 'subject_id')

    # Bar chart for ethnicity
    plot_perc_bar_chart(df = df,
                        group_by = 'group',
                        group_a = 'subject',
                        group_b = 'base',
                        feature = 'ethnicity_simple',
                        value = 'subject_id')

    # Bar chart for hotel expiry
    plot_perc_bar_chart(df = df,
                        group_by = 'group',
                        group_a = 'subject',
                        group_b = 'base',
                        feature = 'hospital_expire_flag',
                        value = 'subject_id')
    
    return df

