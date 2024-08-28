# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

df=pd.read_csv("machine.csv",names=["vendor name",'Model Name',"MYCT",'MMIN',"MMAX","CACH","CHMIN","CHMAX","PRP","ERP"])                      

df.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(df,height=1)
plt.show()

c=df.corr()
sns.heatmap(c)

df1=df.iloc[:,2:8].values
target=df.iloc[:,8].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

df1sc=sc.fit_transform(df1)

pd.DataFrame(df1sc).describe()

from sklearn.model_selection import train_test_split

x_tr,x_te,y_tr,y_te=train_test_split(df1sc,target,test_size=0.2)

from sklearn.linear_model import RANSACRegressor

rr=RANSACRegressor()

rr.fit(x_tr,y_tr)

out=rr.predict(x_tr)

out1=rr.predict(x_te)

from sklearn.metrics import mean_squared_error

mean_squared_error(out,y_tr)

r2_score(out,y_tr)

r2_score(out1,y_te)

mean_squared_error(out1,y_te)


from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lr.fit(x_tr,y_tr)

out=lr.predict(x_tr)

out1=lr.predict(x_te)

from sklearn.metrics import mean_squared_error,r2_score

mean_squared_error(out,y_tr)

r2_score(out,y_tr)

r2_score(out1,y_te)


mean_squared_error(out1,y_te)













