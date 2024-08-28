# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 23:44:23 2021

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




from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(2)

x_pol=poly.fit_transform(data)




x_tr,x_te,y_tr,y_te=train_test_split(x_pol,target,test_size=0.2)



lr=LinearRegression()

lr.fit(x_tr,y_tr)


mean_squared_error(lr.predict(x_tr),y_tr)

mean_squared_error(lr.predict(x_te),y_te)

df1=df.drop(["MYCT"],axis=1)

data=df1.drop(["PRP"],axis=1)


out=lr.predict(x_pol)

out=mn.inverse_transform(out)

df["Polynomial R"]=out

tar=mn.inverse_transform(target)

mean_squared_error(out,tar)





