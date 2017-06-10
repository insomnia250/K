#coding=utf-8
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
import matplotlib
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
##zhfont1 = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/cjkunifonts-ukai/ukai.ttc')

data_train = pd.read_csv('../data/train.csv')
'''--PassengerId--Pclass--Name--Sex--Age--SibSp--Parch--Ticket--Fare--Cabin--Embarked'''


'''-------------缺失值填充(Age , Cabin , Embarked)-------'''
#Age用平均值填充
data_train.Age.fillna(data_train.Age.mean() , inplace = True)

#Cabin有较多缺失值 , 将原Cabin分为有值1和无值0
data_train.loc[data_train.Cabin.notnull() , 'Cabin']='yes'
data_train.loc[data_train.Cabin.isnull() , 'Cabin']='no'


#Embarked 用较多数的种类填充
data_train.loc[data_train.Embarked.isnull() , 'Embarked'] = data_train.Embarked.value_counts().index[0]

'''----------------------哑变量转化----------------------------------'''
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
'''-----------------------------Scaling-------------------------------'''
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)

