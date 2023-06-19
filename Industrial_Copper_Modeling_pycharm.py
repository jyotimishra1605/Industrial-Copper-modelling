#!/usr/bin/env python
# coding: utf-8

# In[145]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

import time
from datetime import datetime
start_time = time.time()

#Importing Libaries for modeling and evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

#Importing libraries for plotting graph
import seaborn as sns
import matplotlib.pyplot as plt


# In[146]:


def import_data(file_path):
    data = pd.read_csv(file_path)
    df = data.copy(deep=True)
    return (df, data)


# In[151]:


def correcting_column_headers(df):
    dict = {'quantity tons': 'quantity_tons', 'item type': 'item_type', 'delivery date': 'delivery_date', 'delivery date new': 'delivery_date_new'}
    df.rename(columns=dict, inplace=True)
    return df


# In[154]:


def handling_invalid_data(df):
    a=df['material_ref'].str.startswith("000000")
    df['material_ref'][a==True] = np.NaN
    
    df['quantity_tons'].replace('(\D.*)',np.nan,inplace=True, regex=True)
    df['quantity_tons'] = df['quantity_tons'].astype('float64')
    return df


# In[157]:


def droping_irrelevent_col(df):
    df = df.drop(['material_ref', 'id'], axis=1)
    return df


# In[159]:


def missing_values(df, data):
    
    cat = [ 'item_type', 'country', 'product_ref', 'status']
    num = [ 'quantity_tons', 'customer', 'application', 'thickness', 'width', 'selling_price']
    
    df['thickness'].fillna(df['thickness'].median(), inplace = True) 
    for i in cat:
        df[i].fillna(df[i].mode()[0], inplace = True) 
    for i in num:
        df[i].fillna(df[i].mean(), inplace = True)
        
    df['item_date'] = df['item_date'].replace([np.nan,None,'null'], df['item_date'].mode()[0])
    df['delivery_date'] = df['delivery_date'].replace([np.nan,None,'null'], df['delivery_date'].mode()[0])
        
    return df


# In[162]:


def invalid_dtypes(df):
    cat = [ 'item_type', 'country', 'product_ref', 'status']
    num = [ 'quantity_tons', 'customer', 'application', 'thickness', 'width', 'selling_price']

    for i in cat:
        df[i] = df[i].astype("object")
    for i in num:
        df[i] = df[i].astype("float64")
        
    df['item_date'] = df['item_date'].astype('int64')
    df['delivery_date'] = df['delivery_date'].astype('int64')
    
    for j in ['item_date', 'delivery_date']:
        for i in range(len(df[j])):
            if int(str(df[j][i])[:4]) not in range(1995,2023):
                df[j][i] = df[j].mode()[0]
            if int(str(df[j][i])[4:6]) not in range(1,13):
                df[j][i] = df[j].mode()[0]
            if int(str(df[j][i])[6:8]) not in range(1,31):
                df[j][i] = df[j].mode()[0]

    for j in ['item_date', 'delivery_date']:
        df[j] = pd.to_datetime(df[j], format='%Y%m%d')
        
    return df


# In[164]:


def feature_creation(df):
    
    #ANALYTICALLY DELIVERY TIME COULD IMPACT TRAINING REGRESSION
    df['delivery_time'] = (df['delivery_date'] - df['item_date']).dt.total_seconds() # CALCULATING DIFFERENCES IN SECONDS
    
    #ASPECT RATIO OF COPPER AND TOTAL AMOUNT IN CURRENCY
    df['aspect_ratio'] = df['width'] / df['thickness']
    df['total_amount'] = df['quantity_tons'] * df['selling_price']
    
    #DROPPING IRRELEVANT FEATURES
    df.drop(['item_date', 'delivery_date'], axis=1, inplace=True)
    
    return df


# In[167]:


def remove_outliers(df):
    
    #Finding and clipping our data based on outliers using iqr technique
    for i in df.select_dtypes(include=['int64', 'float64']):
            iqr = df[i].quantile(0.75) - df[i].quantile(0.25)
            upper_threshold = df[i].quantile(0.75) + (1.5 * iqr) # q3 + 1.5iqr
            lower_threshold = df[i].quantile(0.25) - (1.5 * iqr) # q1 - 1.5iqr
            df = df.copy()
            df[i] = df[i].clip(lower_threshold, upper_threshold)
            
    return df


# In[172]:


def save_cleaned_data(df):
    clean_data = df.copy(deep=True)
    return df


# In[173]:


class RegressorModel:
    
    def __init__(self, x_train, x_test, y_train, y_test):
        
        self.x_test = x_test
        self.x_train = x_train
        self.y_train = y_train
        self.y_test = y_test
    
    def initialize_models(self, reg_mod):

        seed = 1
        if reg_mod == 'Linear Regression':
            return LinearRegression(n_jobs = -1)
        if reg_mod == 'Lasso Regression':
            return Lasso(random_state = seed)
        if reg_mod == 'Ridge Regression':
            return Ridge(random_state = seed)
        if reg_mod == 'ElasticNet':
            return ElasticNet(random_state = seed)
        if reg_mod == 'Decision Tree Regressor':
            return DecisionTreeRegressor(random_state = seed, max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features=None)
        if reg_mod == 'Extra Trees Regressor':
            return ExtraTreesRegressor(n_estimators=100)
        if reg_mod == 'XGB Regressor':
            return XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)

    
    def cross_validate(self, model):
        neg_score = cross_val_score(model, self.x_train, self.y_train, cv = 5, n_jobs = -1, scoring = 'neg_mean_squared_error')
        score = np.round(np.sqrt(-1*neg_score), 5)
        return score.mean()
        
    def train_models(self, model):
        
        model = model.fit(self.x_train, self.y_train) # train
        y_pred = model.predict(self.x_test) # predict
        trcr = model.score(self.x_train, self.y_train)
        tscr = model.score(self.x_test, self.y_test) # evaluate (R2)
        return [trcr, tscr]
            
    def regressor_main(self, model):
        
        model_dict = []
        if model == 'All':
            models = ['Linear Regression', 'Lasso Regression', 'Ridge Regression', 'ElasticNet', 'Decision Tree Regressor', 'Extra Trees Regressor', 'XGB Regressor']
        else:
            models = [model]
        for mod in models:
            model = self.initialize_models(mod)
            rsme = self.cross_validate(model)
            r2_score = self.train_models(model)
            model_dict.append({'Model': mod, 'RSME': rsme, 'Train_Score': r2_score[0], 'Test_Score': r2_score[1]})
        return model_dict


# In[174]:


def regression_analysis(df, model='All'):
    
    #Target Encoding
    df.update(df[[ 'item_type', 'country', 'product_ref', 'status']]
          .apply(lambda s: s.map(df['selling_price'].groupby(s).mean()))
          )
    
    #Splitting Data into Train and Test
    x = df[['quantity_tons', 'customer', 'country', 'status', 'item_type',
           'application', 'thickness', 'width', 'product_ref',
           'delivery_time', 'aspect_ratio', 'total_amount']].values
    y = df[['selling_price']].values
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 101)
    
    #Standarize the dataset before fitting it into the model
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    #Build
    reg_model = RegressorModel(x_train, x_test, y_train, y_test) #intialize regressor class
    reg_score = reg_model.regressor_main(model) #Calling main function in regressor class
    reg_mod = pd.DataFrame.from_dict(reg_score)
    
    
    return reg_mod.sort_values(by='Score', ascending=False)


# In[175]:


class ClassificationModel:
    
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_test = x_test
        self.x_train = x_train
        self.y_train = y_train
        self.y_test = y_test
        
        
    def log_reg_model(self):

        log_reg = LogisticRegression() # initialise the model
        log_reg.fit(self.x_train, self.y_train) #training the data
        y_pred = log_reg.predict_proba(self.x_test) #Predicting
        roc = roc_auc_score(self.y_test, y_pred[:,1]) #Evaluation
        return roc
  

    def knn_model(self):
        
        #Finding the best value for K hyper parameter based on higest cv score
        
        khp = 0
        hcv = 0
        """
        for i in [1,2,3,4,5,6,7,8,9,10]:
            knn = KNeighborsClassifier(i) #initialising the model
            knn.fit(self.x_train,self.y_train) # training the model
            if np.mean(cross_val_score(knn, self.x_train, self.y_train, cv=10, scoring = "roc_auc")) > hcv:
                hcv = np.mean(cross_val_score(knn, self.x_train, self.y_train, cv=10, scoring = "roc_auc"))
                khp = i
            else:
                break
        """
        
        #Input the kbest K value and fit the model
        knn = KNeighborsClassifier(6)
        knn.fit(self.x_train,self.y_train)
        y_pred = knn.predict(self.x_test)
        roc = roc_auc_score(self.y_test, y_pred)
        return roc

    
    def dec_tree_model(self):
        
        """
        khp = 0
        hcv = 0
        for i in [11,12,13,14,15]:
            knn = DecisionTreeClassifier(max_depth=i) #initialising the model
            knn.fit(self.x_train,self.y_train) # training the model
            if np.mean(cross_val_score(knn, self.x_train, self.y_train, cv=10, scoring = "roc_auc")) > hcv:
                hcv = np.mean(cross_val_score(knn, self.x_train, self.y_train, cv=10, scoring = "roc_auc"))
                khp = i
                print (i, hcv)
            else:
                break
        """

        dt = DecisionTreeClassifier(max_depth=14)
        dt.fit(self.x_train, self.y_train)
        y_pred = dt.predict(self.x_test)
        roc = roc_auc_score(self.y_test, y_pred)
        return roc
    
    def ens_model(self):

        model1 = LogisticRegression(random_state=1)
        model2 = tree.DecisionTreeClassifier(max_depth=9, random_state=1)
        model3 = KNeighborsClassifier(6)
        model = VotingClassifier(estimators=[('lr', model1), ('dt', model2),('knn',model3)], voting='soft') 
        model.fit(self.x_train, self.y_train)
        model.predict(self.x_test)
        y_pred = model.predict_proba(self.x_test)
        roc = roc_auc_score(self.y_test,y_pred[:,1])
        return roc
    
    def rf_model(self):

        rf = RandomForestClassifier(max_depth=10,n_estimators=100, max_features='sqrt')
        rf.fit(self.x_train, self.y_train) 
        y_pred = rf.predict(self.x_test)
        roc = roc_auc_score(self.y_test, y_pred)
        return roc
    
    def xg_model(self):
        
        model = XGBClassifier(learning_rate=0.5,n_estimators=100,verbosity=None)
        model.fit(self.x_train, self.y_train)
        y_pred = model.predict(self.x_test)
        roc = roc_auc_score(self.y_test, y_pred)
        return roc
    
    def claasfication_main(self, model):
        
        temp_dict = {'Model':[], 'Score':[]}
            
        if model == 'KNN' or model == 'All':
            knn_score = self.knn_model()
            temp_dict['Model'].append('KNN')
            temp_dict['Score'].append(knn_score)
            
        if model == 'Logistic Regression' or model == 'All':
            log_reg_score = self.log_reg_model()
            temp_dict['Model'].append('Logestic Regression')
            temp_dict['Score'].append(log_reg_score)
        
        if model == 'Decision Tree Classifier' or model == 'All':
            dec_score = self.dec_tree_model()
            temp_dict['Model'].append('Decision Tree Classifier')
            temp_dict['Score'].append(dec_score)
            
        if model == 'Voting Classifier' or model == 'All':
            ens_score = self.ens_model()
            temp_dict['Model'].append('Voting Classifier')
            temp_dict['Score'].append(ens_score)
        
        if model == 'Random Forest Classifier' or model == 'All':
            rf_score = self.rf_model()
            temp_dict['Model'].append('Random Forest Classifier')
            temp_dict['Score'].append(rf_score)
            
        if model == 'XGB Classifier' or model == 'All':
            xg_score = self.xg_model()
            temp_dict['Model'].append('XGB Classifier')
            temp_dict['Score'].append(xg_score)

        cls_df = pd.DataFrame.from_dict(data=temp_dict)
            
        return cls_df


# In[176]:


def classification_analysis(df, model='All'):
    
    #Droping rows with status not as 'Won' or 'Lost'
    indexAge = df[ (df['status'] != 'Won') & (df['status'] != 'Lost') ].index
    df.drop(indexAge , inplace=True)
    
    #Get label encoding for status column
    df["status"] = df["status"].map({"Won":1,"Lost":0}) #encoding binary class data (run only once)
    
    #Target Encoding
    df.update(df[[ 'item_type', 'country', 'product_ref']]
          .apply(lambda s: s.map(df['status'].groupby(s).mean()))
          )
    
    #Splitting Data into Train and Test
    x = df[['quantity_tons', 'customer', 'country', 'item_type',
           'application', 'thickness', 'width', 'product_ref',
           'delivery_time', 'aspect_ratio', 'total_amount', 'selling_price']].values
    y = df[['status']].values
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 101)
    
    #Standarize the dataset before fitting it into the model
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    #Build
    clsmod = ClassificationModel(x_train, x_test, y_train, y_test) #Intialzie the class
    cls_mod = clsmod.claasfication_main(model) 
    return cls_mod.sort_values(by='Score', ascending=False)


# In[177]:


def ml_execution(df, task, model):

    if task == 'Regression':
        df = regression_analysis(df, model)
    else:
        df = classification_analysis(df, model)
        
    return df


# In[ ]:


import streamlit as st

# Define the Streamlit app
def app():
    st.set_page_config(page_title="twitter scraper", layout="wide", initial_sidebar_state="collapsed")
    
    
    # Title and header to be dispalyed
    colT1, colT2 = st.columns([3, 5])
    with colT2:
        title = st.title(' :blue[Industrial Copper Modeling]')
    
    
    #Input Fields
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Add a dropdown to select regression or classification
        task = st.selectbox('Select task', ['Regression', 'Classification'])

    with col2:
    # Add a dependent dropdown to select the model
        if task == 'Regression':
            models = ['All', 'Linear Regression', 'Lasso Regression', 'Ridge Regression', 'ElasticNet', 'Decision Tree Regressor', 'Extra Trees Regressor', 'XGB Regressor']
        else:
            models = ['All', 'Logistic Regression', 'KNN', 'Decision Tree Classifier', 'Voting Classifier', 'Random Forest Classifier', 'XGB Classifier']

        model = st.selectbox('Select model', models)
    
        
    with col3:
        # Upload the CSV file
        file = st.file_uploader('Upload a CSV file', type=['csv'])


    if file is not None:
        # Load the CSV file into a DataFrame
        df = import_data(file)
        
        df = clean_data(df)

        # Run the ML code on the DataFrame with the selected model
        if task == 'Regression':
            if model == 'All':
                results = ml_execution(df, task, 'All')
            if model == 'Linear Regression':
                results = ml_execution(df, task, 'Linear Regression')
            elif model == 'Lasso Regression':
                results = ml_execution(df, task, 'Lasso Regression')
            elif model == 'Ridge Regression':
                results = ml_execution(df, task, 'Ridge Regression')
            elif model == 'ElasticNet':
                results = ml_execution(df, task, 'ElasticNet')
            elif model == 'Decision Tree Regressor':
                results = ml_execution(df, task, 'Decision Tree Regressor')
            elif model == 'Extra Trees Regressor':
                results = ml_execution(df, task, 'Extra Trees Regressor')
            elif model == 'XGB Regressor':
                results = ml_execution(df, task, 'XGB Regressor')
        else:
            if model == 'Logistic Regression':
                results = ml_execution(df, task, 'Logistic Regression')
            elif model == 'KNN':
                results = ml_execution(df, task, 'KNN')
            elif model == 'Decision Tree Classifier':
                results = ml_execution(df, task, 'Decision Tree Classifier')
            elif model == 'Voting Classifier':
                results = ml_execution(df, task, 'Voting Classifier')
            elif model == 'Random Forest Classifier':
                results = ml_execution(df, task, 'Random Forest Classifier')
            elif model == 'XGB Classifier':
                results = ml_execution(df, task, 'XGB Classifier')

        # Display the results DataFrame
        st.write(results)

if __name__ == '__main__':
    app()


# In[ ]:




