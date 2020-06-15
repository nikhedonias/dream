# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:10:02 2020

@author: tstat
"""
# Packages required: pandas, time, lightgbm, numpy
# This script runs the algorithm on a few features including
# fever, lymph count, and rhinorhea amongst others.
#%% Cell 1: Importing and formatting of data

import pandas as pd
import pickle
import time

#initialize timer
t1 = time.perf_counter() 

#import all csv
condition_era = pd.read_csv('condition_era.csv')
condition_occurrence = pd.read_csv('condition_occurrence.csv')
device_exposure = pd.read_csv('device_exposure.csv')
drug_era = pd.read_csv('drug_era.csv')
drug_exposure = pd.read_csv('drug_exposure.csv')
drug_era = pd.read_csv('drug_era.csv')
goldstandard = pd.read_csv('goldstandard.csv')
location = pd.read_csv('location.csv')
measurement = pd.read_csv('measurement.csv')
observation = pd.read_csv('observation.csv')
observation_period = pd.read_csv('observation_period.csv')
person = pd.read_csv('person.csv')
procedure_occurrence = pd.read_csv('procedure_occurrence.csv')
visit_occurrence = pd.read_csv('visit_occurrence.csv')
data_dictionary = pd.read_csv('data_dictionary.csv')
t2 = time.perf_counter()
print("CSV Import as Dataframe: ",t2-t1," seconds")

#create dictionaries for each type of data
dictionaries = []

observationDictionary = data_dictionary[data_dictionary.table == 'observation']

device_exposureDictionary = data_dictionary[data_dictionary.table == 'device_exposure']

condition_occurrenceDictionary = data_dictionary[data_dictionary.table == 'condition_occurrence']

drug_exposureDictionary = data_dictionary[data_dictionary.table == 'drug_exposure']

procedure_occurrenceDictionary = data_dictionary[data_dictionary.table == 'procedure_occurrence']

measurementDictionary = data_dictionary[data_dictionary.table == 'measurement']

visit_occurrenceDictionary = data_dictionary[data_dictionary.table == 'visit_occurrence']

dictionaries = [observationDictionary,device_exposureDictionary,condition_occurrenceDictionary,measurementDictionary,
                    procedure_occurrenceDictionary,visit_occurrenceDictionary,drug_exposureDictionary]
for i in range(len(dictionaries)):
     dictionaries[i] = dictionaries[i].iloc[:,:2]
     dictionaries[i] = dictionaries[i].set_index('concept_id').to_dict()['concept_name']


t3 = time.perf_counter()
print("Dictionary creation: ",t3-t2," seconds")

# append a datetime type datapoint to the end of all dataframes
condition_era = condition_era.assign(datetime = pd.to_datetime(condition_era['condition_era_end_date']))
condition_era = condition_era.assign(condition_era_start_date = pd.to_datetime(condition_era['condition_era_start_date']))
drug_era = drug_era.assign(datetime =pd.to_datetime(drug_era['drug_era_end_date']))
drug_era = drug_era.assign(drug_era_start_date =pd.to_datetime(drug_era['drug_era_start_date']))
condition_occurrence = condition_occurrence.assign(condition_start_datetime =pd.to_datetime(condition_occurrence['condition_start_datetime']))
condition_occurrence = condition_occurrence.assign(datetime =pd.to_datetime(condition_occurrence['condition_end_datetime']))
device_exposure = device_exposure.assign(device_exposure_start_datetime =pd.to_datetime(device_exposure['device_exposure_start_datetime']))
device_exposure = device_exposure.assign(datetime =pd.to_datetime(device_exposure['device_exposure_end_datetime']))
drug_exposure = drug_exposure.assign(drug_exposure_start_datetime =pd.to_datetime(drug_exposure['drug_exposure_start_datetime']))
drug_exposure = drug_exposure.assign(datetime =pd.to_datetime(drug_exposure['drug_exposure_end_datetime']))
measurement = measurement.assign(datetime =pd.to_datetime(measurement['measurement_datetime']))
procedure_occurrence = procedure_occurrence.assign(datetime =pd.to_datetime(procedure_occurrence['procedure_datetime']))
visit_occurrence = visit_occurrence.assign(visit_start_datetime =pd.to_datetime(visit_occurrence['visit_start_datetime']))
visit_occurrence = visit_occurrence.assign(datetime =pd.to_datetime(visit_occurrence['visit_end_datetime']))
observation = observation.assign(datetime =pd.to_datetime(observation['observation_datetime']))
observation_period = observation_period.assign(observation_period_start_date =pd.to_datetime(observation_period['observation_period_start_date']))
observation_period = observation_period.assign(datetime =pd.to_datetime(observation_period['observation_period_end_date']))

t4 = time.perf_counter()
print("Datetime dataframe: ",t4-t3," seconds")

# initialize person level matrix of data for use in machine learning called finalData
dataTables = [condition_era,drug_era,condition_occurrence,device_exposure,drug_exposure,measurement,procedure_occurrence,visit_occurrence,observation,observation_period]
people = list(person.person_id)
people = list(set(people))
finalData = goldstandard
finalData.index = finalData.person_id
del finalData['person_id']
t5 = time.perf_counter()
print("Data tables ready to append: ",t5-t4," seconds")

#%% Cell 2: Append conditions.
from dataFinder import dataFinder
from dataAppend import dataAppendOccurrence,dataAppendMeasurement
import lightgbm as lgb

# symptoms that can be selected for machine learning depending on the 
# feature selection results
symptoms = [254761,437663,31967,255848,321588,321689,314171,314659,432898,
            433417,3004327]

# for each symptom this code parses the DataFrames and selects patient level
# data relevant to the symptom. 
for i in range(len(symptoms)):
    outputs = dataFinder(symptoms[i],dataTables) #finds all data that references the symptom
    data = outputs[symptoms[i]][2]
    category = outputs[symptoms[i]][-1]
    label = outputs[symptoms[i]][0]
    # if data is best defined in a binary format dataAppendOccurrence is run and 
    # appends to the patient level matrix a binary column with a 1 if the patient has had the
    # condition and a zero if they haven't
    if category == 'drug_exposure' or category == 'condition_occurrence' or category == 'device_exposure' or category == 'procedure_occurrence':
        finalData = dataAppendOccurrence(label,data,finalData,time=['all'])
    # for data best represented in a non-binary, integer format dataAppendMeasurement
    # is run and appends the most recent measurement taken
    else:
        data = outputs[symptoms[i]][1]
        finalData = dataAppendMeasurement(label,data,finalData,time=['all'])
    columns = list(finalData.columns)
    # removes 'JSON special characters' which throw an error if fed into lightGBM
    for i in range(len(columns)):
        for j in '[]#/':
            columns[i] = columns[i].replace(j,'')
    finalData.columns = columns

t6 = time.perf_counter()
print("Symptoms appended: ",t6-t5," seconds")
#%% Cell 3: Feature selection using with LGBM regressor model fit
X = finalData.iloc[:,1:len(finalData.columns)]
y = finalData.iloc[:,0]

#Define model.
gbm = lgb.LGBMRegressor()
gbm.fit(X, y)
gbm.booster_.feature_importance()

#Obtain importance of each attribute
fea_imp_ = pd.DataFrame({'cols':X.columns, 'fea_imp':gbm.feature_importances_})

features = fea_imp_.loc[fea_imp_.fea_imp > 0].sort_values(by=['fea_imp'], ascending = False)
#Features gives a ranked list of features where the feature importance is greater than zero.
t7 = time.perf_counter()
print("Features ranked: ",t7-t6," seconds")
#%% Cell 4: Machine learning
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Arbitrary parameters to run gradient-boosted decision tree.
d_train = lgb.Dataset(x_train, label=y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 100
params['min_data'] = 500
params['max_depth'] = 10
clf = lgb.train(params, d_train, 100)

#Prediction
y_pred=clf.predict(x_test)
#convert into binary values
for i in range(len(y_pred)):
    if y_pred[i]>=.101:       # setting threshold to .5
       y_pred[i]=1
    else:  
       y_pred[i]=0

t8 = time.perf_counter()
print("Predicted results output: ",t8-t7," seconds")
print("Total time elapsed: ",t8-t1,"seconds")

#ROC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
score = roc_auc_score(y_test, y_pred)
lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')