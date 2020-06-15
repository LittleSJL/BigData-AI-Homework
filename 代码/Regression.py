# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 21:46:23 2020

@author: leptop
"""
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
normalization = MinMaxScaler()


#读取数据
train_data = pd.read_csv("D:/不可视之禁/大三下/大数据开发实训/课程安排/第五波/NBA salary/70724_149430_bundle_archive/2017-18_NBA_salary.csv")

#清洗数据
train_data['3PAr'] = train_data['3PAr'].fillna('N')
i = train_data[(train_data['3PAr']=='N')].index.tolist()
train_data = train_data.drop(i)

#特征工程+切分数据
salary_total = train_data['Salary']
scaled_salary = normalization.fit_transform(salary_total.values.reshape(-1,1))
train_total = train_data.drop(['Salary'],axis=1)
train_total = train_total.drop(['Tm'],axis=1)
train_total = train_total.drop(['NBA_Country'],axis=1)
x_train, x_test,y_train, y_test = train_test_split(train_total, scaled_salary, test_size=0.1)

#模型训练
lr_model = LinearRegression()
lr_model.fit(x_train,y_train)

#模型预测和评估
lr_pred = lr_model.predict(x_test)
print("MSE:",mean_squared_error(y_test,lr_pred))

#保存结果到csv
prediction_csv = pd.DataFrame(lr_pred,columns=['salary'])
prediction_csv.to_csv("D:/不可视之禁/大三下/大数据开发实训/课程安排/第五波/NBA salary/70724_149430_bundle_archive/prediction.csv",index = False)
