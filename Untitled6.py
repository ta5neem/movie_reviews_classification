#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

import seaborn as sns 
import os,joblib,missingno
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression,SGDRegressor,LogisticRegression, Ridge, Lasso, ElasticNet

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
# from sklearn.features.transformers import DataFrameSelector
from sklearn_features.transformers import DataFrameSelector


# In[11]:


File_path=os.path.join(os.getcwd(),'train.csv')
df_dimond=pd.read_csv(File_path)


# In[12]:


df_dimond['size']=df_dimond['x']*df_dimond['y']*df_dimond['z']


# In[13]:


df_dimond =df_dimond.drop(['Unnamed: 0' , 'x' , 'y' ,'z'], axis=1)


# In[14]:


X=df_dimond.drop(columns='price',axis=1)
y=df_dimond['price']

X_train,X_test,y_train,y_test = train_test_split(X,y,shuffle=True ,test_size=0.15 , random_state=42)


# In[15]:


num_cols=[col for col in X_train.columns if X_train[col].dtype in ['int32', 'int64' , 'float32','float64']]
categ_cols=[col for col in X_train.columns if X_train[col].dtype not in ['int32', 'int64' , 'float32','float64']]


# In[19]:


num_pipline = Pipeline(steps =
         [    ('selector', DataFrameSelector(num_cols)), 
             ('imputer' ,SimpleImputer(strategy='median') ),
             ('scaler' ,StandardScaler())
         ])

categ_pipline = Pipeline(steps =
         [   ('selector', DataFrameSelector(categ_cols)), 
             ('imputer' ,SimpleImputer(strategy='constant',fill_value='missing') ),
             ('ohe' ,OneHotEncoder(sparse=False))
         ])

total_pipeline = FeatureUnion(transformer_list=[
                                            ('categ_pipe', categ_pipline),
                                            ('num_pipe', num_pipline)
                                           
                                               ]
                             )

X_train_final = total_pipeline.fit_transform(X_train) ## fit


# In[20]:


def preprocess_new(X_new):
    ''' This Function tries to process the new instances before predicted using Model
    Args:
    *****
        (X_new: 2D array) --> The Features in the same order
                ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 
                 'population', 'households', 'median_income', 'ocean_proximity']
        All Featutes are Numerical, except the last one is Categorical.
        
     Returns:
     *******
         Preprocessed Features ready to make inference by the Model
    '''
    return total_pipeline.transform(X_new)


# In[ ]:




