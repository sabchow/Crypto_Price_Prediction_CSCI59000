#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 19:03:17 2022

@author: sabrinachowdhury
"""


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
solana=pd.read_csv("/Users/sabrinachowdhury/Desktop/BigData_Spring2022/Project/Data_Set/coin_Solana.csv",index_col=0,usecols=[3,5])
solana.rename(columns={"Low":"Price"},inplace = True)
#solana.iloc[0:452]
solana.head()        

import matplotlib.pyplot as plt
import seaborn as sns

#A variable for predicting 'n' days into the future 
prediction_days = 30
# another column for shifted n units up
solana['Prediction'] = solana['Price'].shift(-prediction_days)

# Create independent data set
# Convert the dataframe to a numpy array and drop prediction column
X1 = np.array(solana.drop(['Prediction'], 1))

# Remove the last n rows where n is the prediction_days
X1 = X1[:len(solana)-prediction_days]

# Create a depedent data set
# Convert the dataframe to a numpy array and drop prediction column
y1 = np.array(solana['Prediction'])
y1 = y1[: -prediction_days]

# Split data into training and testing data
from sklearn.model_selection import train_test_split
# for ada
X1_train, X1_test, y1_train, y1_test =train_test_split(X1,y1,test_size=0.2)

# Set the prediction days array equal to last 30 days from the original data
# for ada
prediction_days_array1 = (solana.drop(['Prediction'],1))[-prediction_days:]

from sklearn.svm import SVR
# Create and train the Support Vector Machine (Regression) using radial basis function
svr_rbf = SVR(kernel='rbf',verbose=True)
# for ada
svr_rbf.fit(X1_train, y1_train)

# Test the model
# for ada
svr_rbf_confidence1 = svr_rbf.score(X1_test, y1_test)
print('svr_rbf accuracy for solana',svr_rbf_confidence1 )

# Print the predicted values
# for ada
svm_prediction1 = svr_rbf.predict(X1_test)
print(svm_prediction1)
# Print the actual values
print(y1_test)

# print the model predictions for next 30 days
#for ada
svm_prediction1 = svr_rbf.predict(prediction_days_array1)
print('FOR SOLANA ',svm_prediction1)

# Predictions for each crypto_currency in a tabular format for 30 days
new_predictions = pd.DataFrame([svm_prediction1[:30], y1_test], index=['SOLANA','SOLANA_ACTUAL']) 
new_predictions = new_predictions.transpose()
new_predictions.head()

# Plotting the attained dataset
lisst = []
for i in range(1,31):
    lisst.append(i)

plt.figure(figsize=(15,10))

plt.subplot(3,2,1)
sns.lineplot(x = lisst, y = new_predictions['SOLANA'][:30], marker = 'o',color='orange' )
plt.title("SOLANA Prediction Prices vs Days")
plt.xlabel("Days")
plt.ylabel("Predicted Price")

plt.subplot(3,2,2)
new_predictions['SOLANA_ACTUAL'][:30].plot(marker='o',color='blue')
plt.title("SOLANA Actual Prices vs Days")
plt.xlabel("Days")
plt.ylabel("Actual Price")

plt.tight_layout()
plt.show()

new_list=pd.DataFrame(new_predictions['SOLANA'][:30])

new_list.to_csv("/Users/sabrinachowdhury/Desktop/BigData_Spring2022/Project/Data_Set/svm_solana_pred.csv")







