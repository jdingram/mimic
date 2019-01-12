import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


def plot_KDE(df, group_by, feature):

    '''

    Plots a KDE for all distinct groups of patients for a single
    continuous feature.
    
    Parameters:
        1. df - the Pandas DataFrame containing the data to be
                visualised
        2. group_by - the column that differentiates the groups that
           should be compared (eg, 'Gender' would compare Males v Females)
        3. feature - the continuous variable who's distribution
           should be visualised

    '''

    plt.figure(figsize = (7, 5))

    #Remove NaNs
    df = df[[group_by, feature]]
    df.dropna(inplace=True)

    # KDE plots for each of the groups
    groups = list(df[group_by].unique())
    for g in groups:
        sns.kdeplot(df.loc[df[group_by] == g, feature], label = g)

    # Labeling of plot
    plt.xlabel(feature);
    plt.ylabel('Density');
    plt.title(feature);

    plt.show()
    
    del df
    plt.clf()


def plot_perc_bar_chart(df, group_by, feature, value):

    '''

    Plots bar charts for discrete variables, comparing the proportion
    of multiple populations falling into each category.
    
    Parameters:
        1. df - the input dataframe
        2. group_by - the primary column to be grouped by
        3. feature - the secondary column to be grouped by
        4. value - the column to be counted and charted
    
    Eg: For both Genders (group_by), find the unique Subject_IDs (value)
        that are in each age bucket (feature). Subject IDs are plotted as
        the % of subjects in each age bucket as a proportion of the total
        subjects for each gender

    '''

    plt.figure(figsize = (7, 5))
    
    t = (df.groupby([group_by, feature])
           .agg({value: 'nunique'})
           .reset_index()
           .rename(columns={value:'col'}))
    t['tot'] = t.groupby(group_by).col.transform('sum')
    t['perc'] = t['col'] / t['tot']

    # Plot
    sns.barplot(data=t, x=feature, y="perc", hue=group_by)
    plt.xticks(rotation='vertical');
    plt.show()
    
    del t
    plt.clf()



def graph_comparisons(df, ids, group_col,
                      plot=['age_on_admission',
                            'age_adm_bucket',
                            'gender',
                            'ethnicity_simple']):
    
    ''' wrapper for plot_perc_bar_chart and plot_KDE that
        uses the former if the data in 'plot' is discrete
        and the latter if it is continuous '''
    
    for p in plot:

        if df[p].dtype == 'O':
            # Bar chart for discrete variables
            plot_perc_bar_chart(df = df,
                                group_by = group_col,
                                feature = p,
                                value = ids)

        else:
            # KDE plot for continuous variables
            plot_KDE(df = df,
                     group_by = group_col,
                     feature = p)



def best_cv_by_run(results, cv_score):
    
    '''
    
    Plots the ongoing best cross validation score by run number of
    a hyperparameter random search. The purpose is to show how the best
    score changes by run number, to ensure the improvement curve had
    flattened by the end of the random search (indicating no benefit in
    running the random search for more iterations).
    
    Parameters:
        1. results - dataframe showing the random search results
        2. cv_score - the column that contains the cross validation scores
    
    '''
    
    # Find the ongoing best CV score by run number
    results.sort_index(ascending=True, inplace=True)
    max_score = results[cv_score].cummax(axis=0).values
    
    # Plot best CV score as a line chart, by run number
    run = list(np.arange(1,len(max_score)+1))
    plt.figure(figsize = (7, 5))
    sns.lineplot(x=run, y=max_score)
    plt.ylabel('Best CV score');
    plt.ylabel('Run');
    plt.title('Best CV score by run');
    plt.show()
    plt.clf()



def plot_single_results(results, training_score, cv_score, params_dict):
    
    '''
    
    Takes the results of a hyperparameter random search and plots a scatter
    chart showing the training and cross validation scores against each
    hyperparameter. The purpose is to see the individual effect of each
    hyperparameter on the training and cross validation scores.
    
    Parameters:
        1. results - the output of the random search, showing training and
           cross validation scores and the associated hyperparameters
        2. training_score - column containing the training score
        3. cv_score - column containing the cross validation scores
        4. params_dict - column containing the dictionary of all
           hyperparameters for the row
    
    The function will assume all columns that are not either training_score,
    cv_score or params_dict are hyperparameter columns, and will therefore be
    plotted.
    
    '''
    
    params = [param for param in results.columns
              if param not in [training_score, cv_score, params_dict]]
    
    for param in params:
    
        results.sort_values(by=param, inplace=True)

        # Extract information from the cross validation model
        train_scores = results[training_score]
        cv_scores = results[cv_score]
        param_values = list(results[param])

        # Plot the scores over the parameter
        plt.figure(figsize = (7, 5))
        plt.scatter(param_values, train_scores, label = 'train')
        plt.scatter(param_values, cv_scores, label = 'cv')
        plt.legend()
        plt.xlabel(param)
        plt.ylabel('Score')
        plt.title('Score vs %s' % param)

        plt.show()
        plt.clf()