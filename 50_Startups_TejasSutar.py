# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:05:02 2020

@author: tejas
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=Startupscsv
state=pd.get_dummies(df["State"])
df1=pd.concat([df,state],axis=1)
df1=df1.drop(["State"],axis=1)
pwd

plt.boxplot(df1["Profit"])  #not normally distributed it is having slightly right skew
plt.boxplot(df1["R&D Spend"])  #not normally distributed it is having slightly right skew
plt.boxplot(df1["Administration"])  #not normally distributed it is having slightly left skew
plt.boxplot(df1["Marketing Spend"])  #not normally distributed it is having slightly right skew

plt.hist(df1["Profit"]) #Data is having right skew
plt.hist(df1["R&D Spend"]) #Data is having right skew
plt.hist(df1["Administration"]) #Data is having right skew
plt.hist(df1["Marketing Spend"]) #Data is normally distributed


import statsmodels.api as snf
snf.graphics.qqplot(df1["Profit"],fit=True,line="45") # Data is linear
snf.graphics.qqplot(df1["R&D Spend"],fit=True,line="45") # Data is linear
snf.graphics.qqplot(df1["Administration"],fit=True,line="45") # Data is linear
snf.graphics.qqplot(df1["Marketing Spend"],fit=True,line="45") # Data is linear

df1.rename(columns={"Marketing Spend":"MS"},inplace=True)
df1.rename(columns={"R&D Spend":"RnD"},inplace=True)
df1.rename(columns={"New York":"NY"},inplace=True)

df1.corr()
import seaborn as sn
sn.pairplot(df1)
correlation_values= df1.corr()




from sklearn.model_selection import train_test_split
train_data,test_data=train_test_split(df1)
train_data=train_data.reset_index()
test_data=test_data.reset_index()
train_data1=train_data.drop("index",axis=1)
test_data1=test_data.drop("index",axis=1)

import statsmodels.formula.api as smf 
m1=smf.ols("Profit~RnD+Administration+MS+NY+Florida+California", data=train_data1).fit()
m1.summary()
#Administration & MS becomes insignificant
snf.graphics.influence_plot(m1)
train_data2=train_data1.drop(train_data1.index[[20]],axis=0)
m2=smf.ols("Profit~RnD+Administration+MS+NY+Florida+California", data=train_data2).fit()
m2.summary()
snf.graphics.influence_plot(m2)

train_data3=train_data1.drop(train_data1.index[[20,9,12]],axis=0)
m3=smf.ols("Profit~RnD+Administration+MS+NY+Florida+California", data=train_data3).fit()
m3.summary()
snf.graphics.influence_plot(m3)


#Checking with VIF Values of independent variables
rsq_Rnd= smf.ols("RnD~Administration+MS+NY+Florida+California", data=train_data3).fit().rsquared  
vif_Rnd = 1/(1-rsq_Rnd)  #2.33

rsq_Administration=smf.ols("Administration~RnD+MS+NY+Florida+California", data=train_data3).fit().rsquared
vif_Administration= 1/(1-rsq_Administration) #1.24

rsq_MS=smf.ols("MS~Administration+RnD+NY+Florida+California", data=train_data3).fit().rsquared
vif_MS= 1/(1-rsq_MS)  #2.36
#AS Vif value of MS is high but it is insignificant.

#Checking with AV Plots
snf.graphics.plot_partregress_grid(m3)
# In AV plots there is some how linearity with MS so preparing the model without MS.

m4=smf.ols("Profit~RnD+NY+Administration+Florida+California", data=train_data3).fit()
m4.summary()

#Final Model
final_model=smf.ols("Profit~RnD+Administration+NY+Florida+California", data=train_data3).fit()
final_model.summary()   #r-squared=0.977

#Training Data Prediction
train_pred=final_model.predict(train_data3)

#train Residuals
train_res=train_data3["Profit"]-train_pred

#train RMSE
train_rmse= np.sqrt(np.mean(train_res*train_res))


#Test Prediction
test_pred=final_model.predict(test_data1)

#Test residuals
test_res=test_data1["Profit"]-test_pred
#Test Rmse
test_rmse=np.sqrt(np.mean(test_res*test_res))


df2=df1.drop(df1.index[[49,48,36]],axis=0)
startup2=smf.ols("Profit~RnD+NY+Administration+Florida+California", data=df2).fit()
startup2.summary()
bestmodel_pred=startup2.predict(df2)


#######  Linearity #########
# Observed values VS Fitted values
plt.scatter(df2.Profit,bestmodel_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")
##Residuals v/s Fitted values
plt.scatter(bestmodel_pred,startup2.resid_pearson, c='r');plt.axhline(y=0,color='blue');plt.xlabel("Fitted values");plt.ylabel("Residuals")
#Model is sligtly Homoscedacticity

##Normality plot for residuals
##histogram
plt.hist(startup2.resid_pearson)
#Slightly left Skew

import pylab
import scipy.stats as st

st.probplot(startup2.resid_pearson, dist='norm', plot=pylab)

