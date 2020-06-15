# -*- coding: utf-8 -*-
"""
Created on Sun May 31 09:06:56 2020

@author: tstat
"""


def dataFinder(values,tables):
    
    import pandas as pd
    
    
    condition_era = tables[0]
    drug_era = tables[1]
    condition_occurrence = tables[2]
    device_exposure = tables[3]
    drug_exposure = tables[4]
    measurement = tables[5]
    procedure_occurrence = tables[6]
    visit_occurrence = tables[7]
    observation = tables[8]
    observation_period = tables[9]
    
    dataDictFrame = pd.read_csv('data_dictionary.csv')
    dataDictInitial = dataDictFrame.to_dict()
    dataDict = {dataDictInitial['concept_name'][i]:[dataDictInitial['concept_name'][i],dataDictInitial['concept_id'][i],dataDictInitial['table'][i]] for i in list(dataDictInitial['concept_id'].keys())}
    dictData = {dataDictInitial['concept_id'][i]:[dataDictInitial['concept_name'][i],dataDictInitial['concept_id'][i],dataDictInitial['table'][i]] for i in list(dataDictInitial['concept_name'].keys())}
    valuesNum = []
    if type(values) == int:
        values = [values]
    for i in values:
        if type(i) == str:
            valuesNum.append(dataDict[i])
        elif type(i) == int:
            valuesNum.append(dictData[i])
    outputs = {}
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
   

def inTimeFrame(df,column,time):
    
    df = df[df.column >= time]




         
    