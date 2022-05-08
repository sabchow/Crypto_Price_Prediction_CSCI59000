#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 01:35:42 2022

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
        
bitcoin=pd.read_csv("/Users/sabrinachowdhury/Desktop/BigData_Spring2022/Project/Data_Set/coin_Bitcoin.csv",index_col=0,usecols=[3,5])
bitcoin.rename(columns={"Low":"Price"},inplace = True)
#bitcoin.iloc[2950:2991]
bitcoin.head()        

import matplotlib.pyplot as plt
import seaborn as sns

#A variable for predicting 'n' days into the future 
prediction_days = 730
# another column for shifted n units up
bitcoin['Prediction'] = bitcoin['Price'].shift(-prediction_days)

# Create independent data set
# Convert the dataframe to a numpy array and drop prediction column
X1 = np.array(bitcoin.drop(['Prediction'], 1))

# Remove the last n rows where n is the prediction_days
X1 = X1[:len(bitcoin)-prediction_days]

# Create a depedent data set
# Convert the dataframe to a numpy array and drop prediction column
y1 = np.array(bitcoin['Prediction'])
y1 = y1[: -prediction_days]

# Split data into training and testing data
from sklearn.model_selection import train_test_split
# for bitcoin
X1_train, X1_test, y1_train, y1_test =train_test_split(X1,y1,test_size=0.2)

# Set the prediction days array equal to last 30 days from the original data
# for bitcoin
prediction_days_array1 = (bitcoin.drop(['Prediction'],1))[-prediction_days:]

from sklearn.svm import SVR
# Create and train the Support Vector Machine (Regression) using radial basis function
#svr_rbf = SVR(kernel='poly', C=1e4, gamma=0.00001,verbose=True)
svr_rbf = SVR(kernel='rbf',verbose=True)

# for bitcoin
svr_rbf.fit(X1_train, y1_train)

# Test the model
# for bitcoin
svr_rbf_confidence1 = svr_rbf.score(X1_test, y1_test)
print('svr_rbf accuracy for bitcoin',svr_rbf_confidence1 )

# Print the predicted values
# for bitcoin
svm_prediction1 = svr_rbf.predict(X1_test)
print(svm_prediction1)
# Print the actual values
print(y1_test)

# print the model predictions for next 730 days
#for bitcoin
svm_prediction1 = svr_rbf.predict(prediction_days_array1)
print('FOR bitcoin ',svm_prediction1)

# Predictions for each crypto_currency in a tabular format for 30 days
new_predictions = pd.DataFrame([svm_prediction1[:730], y1_test], index=['bitcoin','bitcoin_ACTUAL']) 
new_predictions = new_predictions.transpose()
new_predictions.head()

# Plotting the attained dataset
lisst = []
for i in range(1,731):
    lisst.append(i)

plt.figure(figsize=(15,10))

plt.subplot(3,2,1)
sns.lineplot(x = lisst, y = new_predictions['bitcoin'][:730]*7, marker = 'o',color='orange' )
plt.title("Bitcoin Prediction Prices vs Days")
plt.xlabel("Days")
plt.ylabel("Predicted Price")

plt.subplot(3,2,2)
new_predictions['bitcoin_ACTUAL'][:730].plot(marker='o',color='blue')
plt.title("Bitcoin Actual Prices vs Days")
plt.xlabel("Days")
plt.ylabel("Actual Price")

plt.tight_layout()
plt.show()



new_list=pd.DataFrame(new_predictions['bitcoin'][:730]*7)

new_list.to_csv("/Users/sabrinachowdhury/Desktop/BigData_Spring2022/Project/Data_Set/svm_Bitcoin_pred.csv")




