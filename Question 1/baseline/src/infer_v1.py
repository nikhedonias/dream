# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:32:11 2020

@author: Dale
"""


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
score = score/num_feature
score = pd.DataFrame(score,columns = ['score'])
person_id = person[['person_id']]
predictions = pd.concat([person_id,score],axis = 1,ignore_index = False)
predictions.to_csv('/output/predictions.csv', index = False)
print("Predictions are generated", flush = True)