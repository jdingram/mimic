# Predicting Acute Kidney Failure using MIMIC Critical Care Database

## Background
The purpose of this project is to use Machine Learning to predict Acute Kidney Failure using chart and lab data from the MIMIC critical care database, which was collected during patient admissions to the critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.

PLEASE NOTE: This project is intended as an introductory investigation into the data and it's ability to predict diagnoses. For this version, certain investigations have not yet been carried out when producing the models. For example, the models do not control for any comorbidities suffered by the patients or the effects of any drugs. Therefore, this project is only a starting point and therefore should not be used to predict Acute Kidney Failure on real patients.

## Methodology
The data used for creating the models contains 15,575 patient admissions, 39% of whom were diagnosed with Acute Kidney Failure and 61% of whom were not. The purpose of the models is to use chart and lab data to detect whether patients are suffering from Acute Kidney Failure. The scoring metric used was AUC due to the class imbalance of the data.

For each patient admission, 39 chart and lab events were used. A single reading was used per admission, and this was chosen as the first reading per chart and lab event when the patient was admitted to the ICU. This was so that the model could (theoretically) be used to diagnose patients early in an ICU admission. 20% of the data was kept as the test set, and the remaining 80% was used for training and cross validation scoring.

Five Machine Learning models were trained: a Logistic Regression classifier, a Decision Tree, a Random Forest (all from Scikit-Learn), LightGBM and a neural network (Keras's sequential model). The trained models were then all tested on the test data and scored using AUC to assess which performed the best.

The methodology was designed to be general case - meaning that for this version the diagnosis chosen was Acute Kidney Failure, however very simple edits could be made in the notebooks to train the models for any other diagnosis in the database.

## Notebooks
There are 6 notebooks used in the project:

*0_generate_datasets* - Used to clean the raw data and produce CSVs containing patient admission data and chart/ lab data which is used for the remainder of the project.

*1_select_patients* - Used to create the training and test sets used to predict a single diagnosis (in this case Acute Kidney Failure but can easily be configured for any diagnosis in the database). All data pre-processing is completed here and the output datasets are ready for the machine learning models.

*2_baseline_models* - The data is used to train 3 classifiers from Scikit-Learn: LogisticRegression, DecisionTreeClassifier and RandomForestClassifier. The best models (using cross validation accuracy) are saved on S3.

*3_light_gbm* - A LightGBM is trained, with the best model (in terms of cross validation accuracy) saved on S3.

*4_neural_network* - Keras's Sequential Model is trained, with the best model (in terms of cross validation accuracy) saved on S3.

*5_model_testing* - All 5 trained models are then tested on the previously unseen test data so their final accuracies can be compared.

## Results
On the test set, the best model was the Random Forest, with an AUC of 0.88. From looking at the feature importances, the most important features by far are Creatinine and BUN.

## Pipeline
The packages used in this project are saved in the env.yml file. This is largely the Deep Learning AMI (Ubuntu) Version 20.0 from AWS, with the only modifications being the installation of LightGBM and upgrading Seaborn to version 0.9.0. The project was run end to end on AWS EC2 on Ubuntu machines, and all the raw data, clean data and trained models saved on AWS S3.

To reproduce the results, the raw data must be obtained directly from Physio Net. For this reason, the data is not made available in this project directory, and was instead securely saved on AWS S3. https://physionet.org/works/MIMICIIIClinicalDatabase/access.shtml

## Credits
The data used for this project was from the MIMIC Critical Care Database.
MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman L, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. Available at: http://www.nature.com/articles/sdata201635
