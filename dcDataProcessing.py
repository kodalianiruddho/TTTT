# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 20:02:17 2020

@author: Administrator
"""

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import pickle
import joblib
#%%
df = pd.read_csv('/Users/Administrator/.spyder-py3/DC_Properties.csv')
dfNull=  df.isnull().sum()
#%%
print('Percent of missing "Price" records is %.2f%%' %((df['PRICE'].isnull().sum()/df.shape[0])*100))
#%%
#Percent of missing "Price" records is 38.21%
total = df.isnull().sum().sort_values(ascending=False)
percent_1 = df.isnull().sum()/df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([df.count(),total, percent_2], axis=1, keys=['Sum','Total', '%'])
df_NULLdataset=missing_data
#%%
dummy_dataset = df

dummy_dataset['Price_Flag'] = np.where(dummy_dataset.PRICE > 0 , 1,0)

unknown_dataset = dummy_dataset[dummy_dataset.Price_Flag != 1]

unknown_dataset.shape
dataset = dummy_dataset[dummy_dataset.Price_Flag != 0]
dataset.corr()
#%%
df=dataset
df.drop(['Unnamed: 0', "CMPLX_NUM", "LIVING_GBA" , "ASSESSMENT_SUBNBHD", "CENSUS_TRACT", 
         "CENSUS_BLOCK", "GIS_LAST_MOD_DTTM", "SALE_NUM", "USECODE", "CITY", 
         "STATE", "NATIONALGRID",'X','Y','SALEDATE'],axis=1,inplace=True)
#df.drop(['Unnamed: 0', "CMPLX_NUM", "LIVING_GBA" , "ASSESSMENT_SUBNBHD", "CENSUS_TRACT","CENSUS_BLOCK", "GIS_LAST_MOD_DTTM", "SALE_NUM","STORIES", "USECODE", "CITY","STATE", "NATIONALGRID",'X','Y','SALEDATE'],axis=1,inplace=True)
#%%
df.dropna(subset=['AYB'],inplace=True)
group_remodel= df.groupby(['EYB','AYB']).mean()['YR_RMDL']
#%%
def applyRemodel(x):
    if pd.notnull(x['YR_RMDL']):
        return x['YR_RMDL']
    else:
        return round(group_remodel.loc[x['EYB']][x['AYB']])
    
#%%
df['YR_RMDL'] = df[['YR_RMDL','EYB','AYB']].apply(applyRemodel,axis = 1)
df.dropna(subset=['YR_RMDL'],inplace=True)
#%%
df.dropna(subset=['QUADRANT'],inplace=True)

#%%
df.GBA = df.GBA.fillna(df.GBA.mean())

df.KITCHENS = df.KITCHENS.fillna(df.KITCHENS.median())

df.NUM_UNITS = df.NUM_UNITS.fillna(df.NUM_UNITS.median())

df.STYLE = df.STYLE.fillna(df.STYLE.mode()[0])

#%%
df.STORIES = df.STORIES.fillna(df.STORIES.mode()[0])
#%%
df.STRUCT = df.STRUCT.fillna(df.STRUCT.mode()[0])

df.GRADE = df.GRADE.fillna(df.GRADE.mode()[0])

df.CNDTN = df.CNDTN.fillna(df.CNDTN.mode()[0])

df.EXTWALL = df.EXTWALL.fillna(df.EXTWALL.mode()[0])

df.ROOF = df.ROOF.fillna(df.ROOF.mode()[0])

df.INTWALL = df.INTWALL.fillna(df.INTWALL.mode()[0])
#%%
df['AYB'] = df['AYB'].values.astype(int)
df['YR_RMDL'] = df['YR_RMDL'].values.astype(int)
df['NUM_UNITS'] = df['NUM_UNITS'].values.astype(int)
df['KITCHENS'] = df['KITCHENS'].values.astype(int)
df['ZIPCODE'] = df['ZIPCODE'].values.astype(int)

#%%
df.dropna(subset=['QUADRANT'],inplace=True)
df.FULLADDRESS = df.STYLE.fillna(df.STYLE.mode()[0])
temp = []
for item in df.FULLADDRESS.values:
    splt = item.split()[1:]
    sub_address = ' '.join(splt[:(len(splt)-1)])
    temp.append(sub_address)
df.FULLADDRESS = temp
df.rename(columns={'FULLADDRESS':'SUBADDRESS'},inplace=True)
#%%
index_heat_drop = list(df[df.HEAT == 'No Data'].index)
df.drop(index=index_heat_drop,inplace=True)
df = df.reset_index().drop('index',axis=1)

index_condition_drop = list(df[df.CNDTN == 'Default'].index)
df.drop(index=index_condition_drop,inplace=True)
df = df.reset_index().drop('index',axis=1)

def applyAC(x):
    for item in x.AC:
        if item != '0' :
            return(item)
        else :
            return('N')

def applyQuad(x):
    if x.QUADRANT == 'NW':
        return 'Northwest'
    elif x.QUADRANT == 'NE':
        return 'Northeast'
    elif x.QUADRANT == 'SE':
        return 'Southeast'
    elif x.QUADRANT == 'SW':
        return 'Southwest'

df['AC'] = df.apply(applyAC,axis=1)

df.QUADRANT = df.apply(applyQuad,axis=1)

#%%
df.drop(index=56600,inplace=True)
df_Describe=df.describe()
#%%
df.reset_index().drop('index',axis=1,inplace=True)
def check_outlier(data, col):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3-q1
    upper_limit = q3 + (1.5 * iqr)
    lower_limit = q1 - (1.5 * iqr)
    return data[(data[col] < lower_limit) | (data[col] > upper_limit)].index ,upper_limit,lower_limit

index_to_drop,upper,lower= check_outlier(df,'PRICE')
df.drop(index_to_drop,inplace=True)
#%%

df = df.reset_index().drop('index',axis=1)
df_DescribeOutlier=df.describe()
df.to_csv('DC_PropertiesOulier_Clean.csv')
#%%

df.GRADE= df.GRADE.map({
    'Fair Quality': 0,
    'Low Quality' : 1,
    'Average' : 2,
    'Above Average' : 3,
    'Good Quality' : 4,
    'Very Good' : 5, 
    'Exceptional-C' : 6,
    'Exceptional-B' : 7,
    'Exceptional-A' : 8,
    'Excellent' : 9,
    'Superior' : 10,
})
df.CNDTN= df.CNDTN.map({
    'Poor': 0,
    'Fair' : 1,
    'Average' : 2,
    'Good' : 3,
    'Very Good' : 4,
    'Excellent' : 5,
})
#%%
y = df['PRICE']
x2 = df.drop(['SUBADDRESS', 'ZIPCODE','LATITUDE','LONGITUDE','SQUARE','SOURCE','AYB','EYB','PRICE','Price_Flag'],axis=1)
#%%

import category_encoders as ce
x_temp = pd.get_dummies(x2.drop(['GRADE','CNDTN'],axis=1),drop_first=True)
x_temp2 = x2[['GRADE','CNDTN']]

encoder = ce.BinaryEncoder(cols=['HEAT','AC','STYLE','STRUCT','EXTWALL','ROOF','QUALIFIED',
                                 'INTWALL','ASSESSMENT_NBHD','WARD','QUADRANT'])
encoder_transformer= encoder.fit(x2.drop(['GRADE','CNDTN'],axis=1))
x_temp3 = encoder_transformer.transform(x2.drop(['GRADE','CNDTN'],axis=1)).drop(['HEAT_0','AC_0','STYLE_0','STRUCT_0',
            'EXTWALL_0','QUALIFIED_0','ROOF_0','INTWALL_0','ASSESSMENT_NBHD_0','WARD_0','QUADRANT_0'], axis=1)
#%%

x2 = pd.concat([x_temp3,x_temp2],axis=1)
#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler_transformer = scaler.fit(x2[['YR_RMDL','GBA','LANDAREA']])
x_scaled = scaler_transformer.transform(x2[['YR_RMDL','GBA','LANDAREA']])
x_scaled = pd.DataFrame(x_scaled,columns=['YR_RMDL','GBA','LANDAREA'])
x2.drop(['YR_RMDL','GBA','LANDAREA'],axis=1,inplace=True)
x2 = pd.concat([x2,x_scaled],axis=1)
x2.reset_index().drop('index',axis=1,inplace=True)
#regression(x2,y)
#%%

x_train,x_test, y_train,y_test = train_test_split(x2,y, test_size = 0.3, random_state = 0)
model_LR = LinearRegression()
model_LR.fit(x_train, y_train)
# Predicting the Test set results
y_pred = model_LR.predict(x_test)

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
test_set_rmse = (np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Mean Absolute Error:',round( metrics.mean_absolute_error(y_test, y_pred),2) )  
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred),2) )  
print('Root Mean Squared Error:',round( np.sqrt(metrics.mean_squared_error(y_test, y_pred)),2))
#%%
"""
import pickle
filename = 'finalized_model.sav'
pickle.dump(model_LR, open(filename, 'wb'))

filename = 'encoder_features.sav'
pickle.dump(encoder_transformer, open(filename,'wb'))

filename = 'scaller_features.sav'
pickle.dump(scaler_transformer, open(filename,'wb'))
#%%
pickle.dump(model_LR, open('modelDC.pkl','wb'))
"""
#%%%
from sklearn.tree import DecisionTreeRegressor

x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(x2,y,test_size=0.10,random_state=42,shuffle=True)
model =  DecisionTreeRegressor(max_depth=5,random_state=0)
model.fit(x_training_set, y_training_set)
from sklearn.metrics import mean_squared_error, r2_score
model_score = model.score(x_training_set,y_training_set)
# Have a look at R sq to give an idea of the fit ,
# Explained variance score: 1 is perfect prediction
print("coefficient of determination r2 of the prediction.:", model_score)
y_predicted = model.predict(x_test_set)

# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(y_test_set, y_predicted))
# Explained variance score: 1 is perfect prediction
print('Test Variance score: %.2f' % r2_score(y_test_set, y_predicted))























