import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


def plot_KDE(df, group_by, group_a, group_b, feature):

    '''

    Plots a KDE for two groups of patients for a single continuous feature

    '''

    plt.figure(figsize = (7, 5))

    #Remove NaNs
    df = df[[group_by, feature]]
    df.dropna(inplace=True)

    # KDE plots for both groups
    sns.kdeplot(df.loc[df[group_by] == group_a, feature], label = group_a)
    sns.kdeplot(df.loc[df[group_by] == group_b, feature], label = group_b)

    # Labeling of plot
    plt.xlabel(feature); plt.ylabel('Density'); plt.title(feature);

    plt.show()
    
    del df
    plt.clf()



def plot_perc_bar_chart(df, group_by, group_a, group_b, feature, value):

    '''

    Plots bar charts for discrete variables, comparing the proportion
    of 2 populations falling into each category

    '''

    plt.figure(figsize = (7, 5))

    # Count number of unique subjects in the subject and base group,
    # then work out % of group totals
    
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



def graph_comparisons(df, ids, group_col, group_a, group_b,
                      plot=['age_on_admission',
                            'age_adm_bucket',
                            'gender',
                            'ethnicity_simple']):

    for p in plot:

        if df[p].dtype == 'O':
            # Bar chart for discrete variables
            plot_perc_bar_chart(df = df,
                                group_by = group_col,
                                group_a = group_a,
                                group_b = group_b,
                                feature = p,
                                value = ids)

        else:
            # KDE plot for continuous variables
            plot_KDE(df = df,
                     group_by = group_col,
                     group_a = group_a,
                     group_b = group_b,
                     feature = p)



def best_cv_by_run(results, cv_score):
    results.sort_index(ascending=True, inplace=True)
    max_score = results[cv_score].cummax(axis=0)
    run = results.index + 1
    plt.figure(figsize = (7, 5))
    sns.lineplot(x=run, y=max_score)
    plt.ylabel('Best CV score'); plt.ylabel('Run');
    plt.title('Best CV score by run');
    plt.show()



def plot_single_results(results, training_score, test_score, fit_time):
    
    params = [param for param in results.columns
              if param not in [training_score, test_score, fit_time]]
    
    for param in params:
    
        results.sort_values(by=param, inplace=True)

        # Extract information from the cross validation model
        train_scores = results[training_score]
        test_scores = results[test_score]
        train_time = results[fit_time]
        param_values = list(results[param])

        # Plot the scores over the parameter
        plt.subplots(1, 2, figsize=(10, 6))
        plt.subplot(121)
        plt.plot(param_values, train_scores, label = 'train')
        plt.plot(param_values, test_scores, label = 'test')
        plt.legend()
        plt.xlabel(param)
        plt.ylabel('Score')
        plt.title('Score vs %s' % param)

        plt.subplot(122)
        plt.plot(param_values, train_time)
        plt.xlabel(param)
        plt.ylabel('Train Time (sec)')
        plt.title('Training Time vs %s' % param)

        plt.tight_layout(pad = 4)
        plt.show()


def plot_double_results(results, scores):
    
    params = [param for param in results.columns if param not in scores]
    pairs = list(combinations(params, 2))
    
    for p in pairs:
        
        print('=========')
        print('{} and {}'.format(p[0], p[1]))
        print('=========')

        fig, ax = plt.subplots(3,1,figsize=(10, 20))
        fig.subplots_adjust(wspace=3)
        
        for i, s in enumerate(scores):
            
            # Find mean score if duplicate parameters exist
            data = results.groupby([p[0], p[1]]).agg({s:'mean'}).reset_index()
            data = data.pivot(p[0], p[1], s)
            
            sns.heatmap(data, ax=ax[i])
            ax[i].set_title(s)
        
            del data
        
        plt.show()        
        plt.clf()
        
        print()