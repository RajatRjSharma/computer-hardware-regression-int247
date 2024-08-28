# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 17:02:34 2021

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


from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(x_tr,y_tr)



mean_squared_error(mn.inverse_transform(lr.predict(x_te)),mn.inverse_transform(y_te))

mean_squared_error(mn.inverse_transform(lr.predict(x_tr)),mn.inverse_transform(y_tr))



from sklearn.svm import SVR

svc=SVR()


svc.fit(x_tr,y_tr)

mean_squared_error(mn.inverse_transform(svc.predict(x_te).reshape(-1,1)),mn.inverse_transform(y_te))

mean_squared_error(mn.inverse_transform(svc.predict(x_tr).reshape(-1,1)),mn.inverse_transform(y_tr))


from sklearn.decomposition import PCA

pca=PCA(n_components=4)

data1=pca.fit_transform(data)

x_tr,x_te,y_tr,y_te=train_test_split(data1,target,test_size=0.2)


from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(x_tr,y_tr)



mean_squared_error(mn.inverse_transform(lr.predict(x_te)),mn.inverse_transform(y_te))

mean_squared_error(mn.inverse_transform(lr.predict(x_tr)),mn.inverse_transform(y_tr))

from sklearn.svm import SVR

svc=SVR()

svc.fit(x_tr,y_tr)

mean_squared_error(mn.inverse_transform(svc.predict(x_te).reshape(-1,1)),mn.inverse_transform(y_te))

mean_squared_error(mn.inverse_transform(svc.predict(x_tr).reshape(-1,1)),mn.inverse_transform(y_tr))




from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

for i in range(1,6):
    print(i)
    print()
    pca=PCA(n_components=i)

    data1=pca.fit_transform(data)


    x_tr,x_te,y_tr,y_te=train_test_split(data1,target,test_size=0.2)


    lr=LinearRegression()

    lr.fit(x_tr,y_tr)
    
    
    print(mean_squared_error(mn.inverse_transform(lr.predict(x_te)),mn.inverse_transform(y_te)))

    print(mean_squared_error(mn.inverse_transform(lr.predict(x_tr)),mn.inverse_transform(y_tr)))

    print()
    Rr = RANSACRegressor()
    
    Rr.fit(x_tr,y_tr)
    
    
    print(mean_squared_error(mn.inverse_transform(Rr.predict(x_te)),mn.inverse_transform(y_te)))

    print(mean_squared_error(mn.inverse_transform(Rr.predict(x_tr)),mn.inverse_transform(y_tr)))

    print()
    
    dr = DecisionTreeRegressor(max_features='sqrt')
   
    dr.fit(x_tr,y_tr)
    
    
    print(mean_squared_error(mn.inverse_transform(dr.predict(x_te).reshape(-1,1)),mn.inverse_transform(y_te)))

    print(mean_squared_error(mn.inverse_transform(dr.predict(x_tr).reshape(-1,1)),mn.inverse_transform(y_tr)))

    print()
    
    rf = RandomForestRegressor()
    
    rf.fit(x_tr,y_tr.ravel())


    print(mean_squared_error(mn.inverse_transform(rf.predict(x_te).reshape(-1,1)),mn.inverse_transform(y_te)))

    print(mean_squared_error(mn.inverse_transform(rf.predict(x_tr).reshape(-1,1)),mn.inverse_transform(y_tr)))
    print()



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda=LinearDiscriminantAnalysis(n_components=None)


data_lda=lda.fit_transform(data,target)


x_tr,x_te,y_tr,y_te=train_test_split(data_lda,target,test_size=0.2)



lr=LinearRegression()

lr.fit(x_tr,y_tr)



print(mean_squared_error(mn.inverse_transform(lr.predict(x_te)),mn.inverse_transform(y_te)))

print(mean_squared_error(mn.inverse_transform(lr.predict(x_tr)),mn.inverse_transform(y_tr)))




from sklearn.decomposition import KernelPCA

kpca=KernelPCA(kernel='linear',n_components=100)
data_kpca=kpca.fit_transform(data)


data_kpca.shape


x_tra,x_tea,y_tra,y_tea=train_test_split(data_kpca,target,test_size=0.3)

lr_kp=LinearRegression()
lr_kp.fit(x_tra,y_tra)


tr_pre=lr_kp.predict(x_tra)
te_pre=lr_kp.predict(x_tea)

print("Training Accuracy: ",mean_squared_error(mn.inverse_transform(tr_pre),mn.inverse_transform(y_tra)))
print("Testing Accuracy: ",mean_squared_error(mn.inverse_transform(te_pre),mn.inverse_transform(y_tea)))











