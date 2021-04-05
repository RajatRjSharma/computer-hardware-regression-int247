# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 23:10:06 2021

@author: DELLRJ5999
"""

#names=["vendor name",'Model Name',"MYCT",'MMIN',"MMAX","CACH","CHMIN","CHMAX","PRP","ERP"]

import pandas as pd
import numpy as np

df=pd.read_csv("machine.csv")                      

df.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(df,height=1)
plt.show()

c=df.corr()
sns.heatmap(c)

df.info()

df.head()

df.describe()

df=df.drop(['vendor name','Model Name',"ERP"],axis=1)

target=df[["PRP"]]

data=df.drop('PRP',axis=1)

dd=data.corr()
sns.heatmap(dd)

from sklearn.preprocessing import MinMaxScaler
mn=MinMaxScaler()

data=mn.fit_transform(data)

target=mn.fit_transform(target)

pd.DataFrame(data).describe()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,RANSACRegressor
from sklearn.metrics import mean_squared_error


x_tr,x_te,y_tr,y_te=train_test_split(data,target,test_size=0.2)


lr = LinearRegression()
lr.fit(x_tr,y_tr)

mean_squared_error(lr.predict(x_tr),y_tr)

mean_squared_error(lr.predict(x_te),y_te)

Rr = RANSACRegressor()
Rr.fit(x_tr,y_tr)

mean_squared_error(Rr.predict(x_tr),y_tr)

mean_squared_error(Rr.predict(x_te),y_te)



from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



dr = DecisionTreeRegressor(max_features='sqrt')
dr.fit(x_tr,y_tr)

mean_squared_error(dr.predict(x_tr),y_tr)

mean_squared_error(dr.predict(x_te),y_te)

rf = RandomForestRegressor()
rf.fit(x_tr,y_tr.ravel())

mean_squared_error(rf.predict(x_tr),y_tr)

mean_squared_error(rf.predict(x_te),y_te)


df["New Out"]=Rr.predict(data)

df["New Out"]=mn.inverse_transform(df["New Out"])


out1=lr.predict(data)

out1=out1.reshape(-1,1)

out1=mn.inverse_transform(out1)

tar=mn.inverse_transform(target)

mean_squared_error(out1,tar)

df["Random R"]=out1

data1=df.drop(['MYCT',"PRP","New Out"],axis=1)

data1=df.drop(['MYCT',"PRP"],axis=1)

target1=df[['PRP']]

dd=data1.corr()
sns.heatmap(dd)

from sklearn.preprocessing import MinMaxScaler
mn=MinMaxScaler()

data1=mn.fit_transform(data1)

target1=mn.fit_transform(target1)


lr = LinearRegression()
lr.fit(data1,target1)

mean_squared_error(lr.predict(data1),target1)



Rr = RANSACRegressor()
Rr.fit(data1,target1)

mean_squared_error(Rr.predict(data1),target1)




from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



dr = DecisionTreeRegressor(max_features='sqrt')
dr.fit(data1,target1)

mean_squared_error(Rr.predict(data1),target1)


rf = RandomForestRegressor()
rf.fit(data1,target1.ravel())

mean_squared_error(Rr.predict(data1),target1)


out1=rf.predict(data1)

out1=out1.reshape(-1,1)

out1=mn.inverse_transform(out1)

df["Random R"]=out1

df=df.drop(['New Out1'],axis=1)

df.to_csv("Regression")

mean_squared_error(df["PRP"],df["New Out1"])

mean_squared_error(df["PRP"],df["New Out"])










