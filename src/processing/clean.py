'''

Functions that contain common data cleaning operations

'''

import numpy as np
import psycopg2 as p
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths
src_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
src_preparation_folder = os.path.join(src_folder, 'preparation')

from import_data import get_table


def replace_itemid_with_label(df, exclude):

  '''

  Takes a DF with itemids as column names and replaces the column name with the label

  '''

  # As a QA step, ensure all columns not in the exclude list are type int
  all_cols = [int(c) if c not in exclude else c for c in df.columns]

  # Take all column names, and extract the matching labels for itemids
  i = tuple(set([c for c in all_cols if c not in exclude]))

  chart_names = get_table(host = 'localhost',
                        dbname = 'mimic',
                        schema = 'mimiciii',
                        table = 'd_items',
                        columns = 'label, itemid',
                        where = "itemid IN {}".format(i))

  lab_names = get_table(host = 'localhost',
                        dbname = 'mimic',
                        schema = 'mimiciii',
                        table = 'd_labitems',
                        columns = 'label, itemid',
                        where = "itemid IN {}".format(i))

  full_names = chart_names.append(lab_names)

  new_cols = [full_names.loc[full_names['itemid']==f, 'label'].values[0] if f in full_names.itemid.tolist() else f for f in all_cols]
  df.columns = new_cols

  return df


def find_populated_cols(df, target, frac):

  '''

  Takes a DF and returns only the cols that have a given % of populated values - for both the target and hospital

  '''

  # Find the unique target values
  target_values = df[target].unique().tolist()

  # Find the unique hospital values
  hospital_values = df['dbsource'].unique().tolist()

  # Create dict to hold lists of keep cols
  keep_cols_dict = {}

  # Cycle through each group (split by target) making lists of which cols contain more non missing values than frac
  for v in target_values:
      v_temp = []
      for h in hospital_values:
        t = df[(df[target]==v) & (df['dbsource']==h)]
        v_temp += [i for i,x in enumerate(t.isna().sum().values.tolist()) if x < ((1-frac)*len(t))]
      keep_cols_dict[v] = v_temp
      del v_temp

  # De-dupe the lists, leaving only column indexes that have enough non missing values in every group
  print("Original features: " + str(df.shape[1]))

  keep_cols = [i for i in range(0, df.shape[1])]
  for key, value in keep_cols_dict.items():
      keep_cols = list(set(keep_cols).intersection(value))

  print("Features kept: " + str(len(keep_cols)))

  # Keep only the columns that have enough non missing values in all groups & hospitals
  df = df.iloc[:,sorted(keep_cols)]

  return df



