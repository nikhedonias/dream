# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 08:59:31 2020

@author: tstat
"""


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


