# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 19:50:00 2020

@author: Dale
"""
print("Script initiated.", flush = True)
def dataAppendOccurrence(label,data,finalData,time = ['all']):
    import pandas as pd
    
    peopleDict = {i : 0 for i in list(finalData.index)}
        
    if time[0] == 'all':
        persons = list(set(list(data.person_id)))
        peopleDict2 = peopleDict
        for i in persons:
            peopleDict2[i] += 1
        adf = pd.DataFrame.from_dict(peopleDict2,orient = 'index')
        finalData[label] = adf
    elif time[0] != 'all':
        for i in range(len(time)):
            data2 = data
            start = time[i]
            if i+1 != len(time):
                end = time[i+1]
            else:
                end = 'present'
            
            if end != 'present':
                data2 = data2[(data2['datetime'] >= start)]
                data2 = data2[(data2['datetime'] < end)]
            if end == 'present':
                data2 = data2[data2.datetime >= start]
            
            persons = list(set(list(data2.person_id)))
            peopleDict = {i : 0 for i in list(finalData.index)}
            for i in persons:
                peopleDict[i] += 1
            adf = pd.DataFrame.from_dict(peopleDict,orient = 'index')
            label2 = str(label) + ' ' + end
            finalData[label2] = adf
            
    return finalData


def dataAppendMeasurement(label,data,finalData,time = ['all']):
    import pandas as pd
    import numpy as np
    
    if time[0] == 'all':
        df = pd.DataFrame(np.nan,index = finalData.index,columns = ['A'])
        for i in list(finalData.index):
            adf = data[data['person_id'] == i]
            if len(adf.index) > 0:
                row = adf[adf['datetime'] == max(adf['datetime'])]
                if len(row.index) > 1:
                    row = row.iloc[1,:]
                value = row['value_as_number'].item()
                df.loc[i,'A'] = value
        finalData[label] = df
    elif time[0] != 'all':
        for i in range(len(time)):
            data2 = data
            start = time[i]
            if i+1 != len(time):
                end = time[i+1]
            else:
                end = 'present'
            
            if end != 'present':
                data2 = data2[(data2['datetime'] >= start)]
                data2 = data2[(data2['datetime'] < end)]
            if end == 'present':
                data2 = data2[data2.datetime >= start]
            df = pd.DataFrame(np.nan,index = finalData.index,columns = ['A'])
            for i in list(finalData.index):
                
                adf = data2[data2['person_id'] == i]
                if len(adf.index) > 0:
                    row = adf[adf['datetime'] == max(adf['datetime'])]
                    if len(row.index) > 1:
                        row = row.iloc[1,:]
                    value = row['value_as_number'].item()
                    df.loc[i,'A'] = value
            label2 = str(label) + ' ' + end
            finalData[label2] = df
            
    return finalData

def dataFinder(values,tables):
    # this function finds all instances of a certain set of conditions in all 
    # relevant data tables. It takes a list of the values that the user wants 
    # to find and the data tables available to find it in.
    import pandas as pd
    
    # names tables inputted, assumes all given
    #condition_era = tables[0]
    #drug_era = tables[1]
    condition_occurrence = tables[1]
    #device_exposure = tables[3]
    #drug_exposure = tables[4]
    measurement = tables[0]
    #procedure_occurrence = tables[6]
    #visit_occurrence = tables[7]
    #observation = tables[8]
    #observation_period = tables[9]
    
    # reads in the data_dictionary and formats it
    dataDictFrame = pd.read_csv('/data/data_dictionary.csv')
    dataDictInitial = dataDictFrame.to_dict()
    
    # creates two dataframes that can be used in either direction, from name to id
    # or from id to name
    dataDict = {dataDictInitial['concept_name'][i]:[dataDictInitial['concept_name'][i],dataDictInitial['concept_id'][i],dataDictInitial['table'][i]] for i in list(dataDictInitial['concept_id'].keys())}
    dictData = {dataDictInitial['concept_id'][i]:[dataDictInitial['concept_name'][i],dataDictInitial['concept_id'][i],dataDictInitial['table'][i]] for i in list(dataDictInitial['concept_name'].keys())}
    valuesNum = []
    # gets dictionary data for each value provided whether it is an integer or
    # a string
    if type(values) == int:
        values = [values]
    for i in values:
        if type(i) == str:
            valuesNum.append(dataDict[i])
        elif type(i) == int:
            valuesNum.append(dictData[i])
    outputs = {}
    
    # puts together output. This includes 3-4 values, the label, the individual
    # occurrence data frame, the era data frame if applicable, and the category
    for i in valuesNum:
        category = i[2]
        value = i[1]
        
        if category == 'condition_occurrence':
            outputs[value] = [i[0],condition_occurrence[condition_occurrence.condition_concept_id == value], condition_era[condition_era.condition_concept_id == value],category]
        if category == 'drug_exposure':
            outputs[value] = [i[0],drug_exposure[drug_exposure.drug_concept_id == value],drug_era[drug_era.drug_concept_id == value],category]
        if category == 'observation':
            outputs[value] = [i[0],observation[observation.observation_concept_id == value],observation_period[observation_period.observation_concept_id == value],category]
        if category == 'measurement':
            outputs[value] = [i[0],measurement[measurement.measurement_concept_id == value],category]
        if category == 'device_exposure':
            outputs[value] = [i[0],device_exposure[device_exposure.device_concept_id == value],category]
        if category == 'visit_occurrence':
            outputs[value] = [i[0],visit_occurrence[visit_occurrence.visit_concept_id == value],category]
        if category == 'procedure_occurrence':
            outputs[value] = [i[0],procedure_occurrence[procedure_occurrence.procedure_concept_id == value],category]
        
    return outputs

print("Functions defined.", flush = True)

import pandas as pd
import pickle

goldstandard = pd.read_csv('/data/goldstandard.csv')
measurement = pd.read_csv('/data/measurement.csv')
condition_occurrence = pd.read_csv('/data/condition_occurrence.csv')
person = pd.read_csv('/data/person.csv')
    
measurement = measurement.assign(datetime =pd.to_datetime(measurement['measurement_datetime']))
condition_occurrence = condition_occurrence.assign(condition_start_datetime =pd.to_datetime(condition_occurrence['condition_start_datetime']))
condition_occurrence = condition_occurrence.assign(datetime =pd.to_datetime(condition_occurrence['condition_end_datetime']))

dataTables = [measurement,condition_occurrence]
people = list(person.person_id)
people = list(set(people))
finalData = goldstandard
finalData.index = finalData.person_id
del finalData['person_id']

print("All measurements collected.", flush = True)

import datetime as dt
import numpy as np
from datetime import date
import pandas as pd
print("Load measurement.csv", flush = True)
measurement = pd.read_csv('/data/measurement.csv',usecols = ['measurement_concept_id','value_as_number','person_id'])
measurement_feature = {'3020891':37.5,'3027018':100,'3012888':80,'3004249':120,
'3023314':52,'3013650':8,'3004327':4.8,'3016502':95,'3023091':21}
measurement = measurement.dropna(subset = ['measurement_concept_id'])
measurement = measurement.astype({"measurement_concept_id": int})
measurement = measurement.astype({"measurement_concept_id": str})
feature = dict()
'''
 measurement
| Feature|OMOP Code|Domain|Notes|
|-|-|-|-|
|age|-|person|>60|
|temperature|3020891|measurement|>37.5'|
|heart rate|3027018|measurement|>100n/min|
|diastolic blood pressure|3012888|measurement|>80mmHg|
|systolic blood pressure|3004249|measurement|>120mmHg|
|hematocrit|3023314|measurement|>52|
|neutrophils|3013650|measurement|>8|
|lymphocytes|3004327|measurement|>4.8|
|oxygen saturation in artery blood|3016502|measurement|<95%|
|IL-6 levels|3023091|measurement|>21pg/ml|
'''
for i in measurement_feature.keys():
    subm = measurement[measurement['measurement_concept_id'] == i]
    if i != '3016502':
        subm_pos = subm[subm['value_as_number'] > measurement_feature[i]]
        feature[i] = set(subm_pos.person_id)
    else:
        subm_pos = subm[subm['value_as_number'] < measurement_feature[i]]
        feature[i] = set(subm_pos.person_id)

'''
condition
| Feature|OMOP Code|Domain|Notes|
|-|-|-|-|
|cough|254761|condition|-|
|pain in throat|259153|condition|-|
|headache|378253|condition|-|
|fever|437663|condition|-|
'''
print("Load condition.csv", flush = True)
condition_feature = ['254761','437663','378253','259153']
condition = pd.read_csv("/data/condition_occurrence.csv",usecols = ['condition_concept_id','person_id'])
condition = condition.dropna(subset = ['condition_concept_id'])
condition = condition.astype({"condition_concept_id": int})
condition = condition.astype({"condition_concept_id": str})
for i in condition_feature:
    subm = condition[condition['condition_concept_id'] == i]
    feature[i] = set(subm.person_id)
person = pd.read_csv('/data/person.csv')
today = date.today().year
person['age'] = person['year_of_birth'].apply(lambda x: today - x )
sub = person[person['age'] > 60]
feature['age'] = set(sub.person_id)

'''generate the feature set'''
person = person.drop_duplicates(subset = ['person_id'])
person_index = dict(zip(person.person_id, range(len(person.person_id))))
feature_index = dict(zip(feature.keys(), range(len(feature.keys()))))
index_feat_matrix = np.zeros((len(person_index), len(feature_index)))
for i in feature.keys():
    index_f = feature_index[i]
    for person_id in feature[i]:
        index_p = person_index[person_id]
        index_feat_matrix[index_p,index_f] = 1
score = index_feat_matrix.sum(axis = 1)
num_feature = 14
print("Feature set is generated", flush = True)
score_temp = score/num_feature
score_temp = pd.DataFrame(score_temp,columns = ['score'])
person_id = person[['person_id']]
predictions = pd.concat([person_id,score_temp],axis = 1,ignore_index = False)
print("Symptomatic people generated.", flush = True)

people = predictions[predictions.score >= .55] #Threshold of 0.55 for pre-filtering


# symptoms that can be selected for machine learning depending on the 
# feature selection results
symptoms = [3000905,3000963,3003396,3003694,3004249,3004327,3004501,3005491,
            3006262,3006923,3007220,3008037,3008152,3010156,3011960,3012158,3012544,
            3012888,3013650,3013682,3013707,3013721,3014576,3015242,3016407,
            3016502,3016723,3018405,3018677,3019550,3019897,3019977,3020716,
            3020891,3021337,3022192,3022250,3023091,3023103,3023314,3023548,3024128,
            3024171,3024561,3024929,3025023,3025315,3025634,3027018,3027114,3027801,
            3027946,3028167,3033891,3036277,3038288,3038297,3041623,3042194,3042596,
            3044254,3044938,3045716,3046279,4196147,40765161,42870366]

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
    if category == 'condition_occurrence':
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
    
del finalData['status']
print("All measurements appended.", flush = True)

#Dropping NaN values (by order of most NaN to least NaN features)
idx = people.iloc[:,0]
finalData = finalData.loc[idx,:]
t = finalData.dropna(axis=1,how = 'all')
thresh = 0.9*len(t) #Threshold of 90% of filtered cohort is applied.
t3 = t.fillna(-10000)
sums = t3.sum(axis=0)
a = sums.idxmin()
t = t.drop(labels = a, axis = 1)
t2 = t.dropna(axis=0)

while len(t2) < thresh:
    t3 = t.fillna(-10000)
    sums = t3.sum(axis=0)
    a = sums.idxmin()
    t = t.drop(labels = a, axis = 1)
    t2 = t.dropna(axis=0)
X = t2
print("Pre-filter applied.", flush = True)

import pandas as pd 
from sklearn.cluster import SpectralClustering 
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.decomposition import PCA 

cluster = 2

# Scaling the Data 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 

# Normalizing the Data 
X_normalized = normalize(X_scaled) 
  
# Converting the numpy array into a pandas DataFrame 
X_normalized = pd.DataFrame(X_normalized) 
  
# Reducing the dimensions of the data 
pca = PCA(n_components = cluster) 
X_principal = pca.fit_transform(X_normalized) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 

# Building the clustering model 
spectral_model_nn = SpectralClustering(n_clusters = cluster, affinity ='nearest_neighbors') 
# Training the model and Storing the predicted cluster labels 
labels_nn = spectral_model_nn.fit_predict(X_principal) 

print("Nearest neighbor spectral clustering complete.", flush = True)

#To select which cluster has a higher probability of risk factors.
idx = X[labels_nn==0].index
idx1 = X[labels_nn>0].index
score = pd.DataFrame(score,columns = ['score'])
score1 = score.loc[idx,:].sum()
score2 = score.loc[idx1,:].sum()

if score1[0]>score2[0]:
    score.loc[idx,:] = score.loc[idx,:]+4
else:
    score.loc[idx1,:] = score.loc[idx1,:]+4
    
num_feature = num_feature+4
print("Feature set is generated", flush = True)
score_real = score/num_feature
score_real = pd.DataFrame(score_real,columns = ['score'])
person_id = person[['person_id']]
predictions = pd.concat([person_id,score_real],axis = 1,ignore_index = False)
predictions.to_csv('/output/predictions.csv', index = False)
print("Predictions are generated.", flush = True)
