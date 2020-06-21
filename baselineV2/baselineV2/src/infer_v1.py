# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 09:43:49 2020

@author: Dale
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:18:27 2020

@author: Dale
"""


import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.decomposition import PCA
import datetime as dt
from datetime import date
import time

print("Script initiated.",flush=True)
t1 = time.perf_counter()
person = pd.read_csv('/data/person.csv')
measurement = pd.read_csv('/data/measurement.csv',usecols = ['measurement_concept_id','value_as_number','person_id'])
condition = pd.read_csv("/data/condition_occurrence.csv",usecols = ['condition_concept_id','person_id'])
dataDict = pd.read_csv("/data/data_dictionary.csv",usecols = ['concept_id'])
print("Data loaded.")

people = list(person.person_id)
people = list(set(people))
zero_data = np.zeros(shape=len(people))
finalData = pd.DataFrame(zero_data)
finalData.columns = ['status']

print("Load measurement.csv", flush = True)
measurement_feature = {'3020891':37.5,'3027018':100,'3012888':80,'3004249':120,
'3023314':52,'3013650':8,'3004327':4.8,'3016502':95,'3023091':21}
measurement = measurement.dropna(subset = ['measurement_concept_id'])
measurement = measurement.astype({"measurement_concept_id": int})
measurement = measurement.astype({"measurement_concept_id": str})
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
        a = list(set(subm_pos.person_id))
        zeros1 = np.zeros(shape = len(people))
        zeros1[a] = 1
        finalData[i] = pd.DataFrame(zeros1)
    else:
        subm_pos = subm[subm['value_as_number'] < measurement_feature[i]]
        a = list(set(subm_pos.person_id))
        zeros1 = np.zeros(shape = len(people))
        zeros1[a] = 1
        finalData[i] = pd.DataFrame(zeros1)

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
dataDict = dataDict.astype({"concept_id":int})
dataDict = dataDict.astype({"concept_id":str})
dataDict = dataDict.values.tolist()

del finalData['status']


condition_feature = [['254761'],['437663'],['257683'],['4223659'],['196523'],['378253'],
                     ['436235'],['4185711'],['27674'],['4168213']]
condition = condition.dropna(subset = ['condition_concept_id'])
condition = condition.astype({"condition_concept_id": int})
condition = condition.astype({"condition_concept_id": str})
for i in condition_feature:
    subm = condition[condition['condition_concept_id'] == i[0]]
    zeros = np.zeros(shape = len(people))
    zeros[subm.person_id] = 1
    finalData[i[0]] = pd.DataFrame(zeros)
    
finalData = finalData.loc[:, (finalData != 0).any(axis=0)]
    
print("Load person.csv", flush = True)
today = date.today().year
person['age'] = person['year_of_birth'].apply(lambda x: today - x )
sub = person[person['age'] > 60]
zeros = np.zeros(shape = len(people))
zeros[sub.person_id] = 1
finalData['age'] = pd.DataFrame(zeros)

print("Clustering initiated", flush = True)
X = finalData

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
# Training the model and Storing the predicted cluster labels 
model_ac = AgglomerativeClustering(n_clusters=2,affinity = 'cityblock', linkage = 'complete')
print("Model generated",flush=True)
labels_ac = model_ac.fit(X_principal)
labels = labels_ac.labels_
print("labels generated",flush = True)

measure = finalData.iloc[:,0:9]
measure = measure.sum(axis=1)

idx = X[labels>0].index
idx1 = X[labels==0].index
if len(idx) < len(idx1):
    measure[idx] = measure[idx] + 4
else:
    measure[idx1] = measure[idx1] + 4
measure = measure/13
score_real = pd.DataFrame(measure,columns = ['score'])
person_id = person[['person_id']]
predictions = pd.concat([person_id,score_real],axis = 1,ignore_index = False)
predictions.to_csv('/output/predictions.csv', index = False)
print("Predictions are generated.", flush = True)
    

t2 = time.perf_counter()

print("Total Time Elapsed: ",t2-t1,flush = True)
