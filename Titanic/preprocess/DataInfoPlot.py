#coding=utf-8
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
import matplotlib
import matplotlib.pyplot as plt
from pylab import mpl
import seaborn as sns
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
##zhfont1 = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/cjkunifonts-ukai/ukai.ttc')

data_train = pd.read_csv('../data/train.csv')


'''--PassengerId--Pclass--Name--Sex--Age--SibSp--Parch--Ticket--Fare--Cabin--Embarked'''
##data_train.info()

'''-------------图1-----------'''
# fig = plt.figure(1)
# ##
# plt.subplot2grid((2,3) , (0 , 0))
# data_train.Survived.value_counts().plot(kind = 'bar')
# plt.title(u"获救情况(1为获救)")
# plt.ylabel(u"人数" , fontproperties='Microsoft YaHei')

# plt.subplot2grid( (2,3) , (0,1))
# data_train.Pclass.value_counts().plot(kind = 'bar')
# plt.title(u"乘客等级分布")
# plt.ylabel(u"人数")

# plt.subplot2grid((2,3),(0,2))
# plt.scatter(data_train.Survived, data_train.Age)
# plt.ylabel(u"年龄")                         # 设定纵坐标名称
# plt.grid(b=True, which='major', axis='y') 
# plt.title(u"按年龄看获救分布 (1为获救)")

# plt.subplot2grid((2,3),(1,0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u"年龄")# plots an axis lable
# plt.ylabel(u"密度") 
# plt.title(u"各等级的乘客年龄分布")
# plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best')

# plt.subplot2grid((2,3),(1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title(u"各登船口岸上船人数")
# plt.ylabel(u"人数")

'''-------------图2-----------'''

# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title(u"各乘客等级的获救情况")
# plt.xlabel(u"乘客等级") 
# plt.ylabel(u"人数") 

# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
# df=pd.DataFrame({u'男性':Survived_m, u'女性':Survived_f})
# df.plot(kind='bar', stacked=True)
# plt.title(u"按性别看获救情况")
# plt.xlabel(u"是否获救") 
# plt.ylabel(u"人数")
# plt.show()

'''----------------------------------------------------'''
data_train.Age.fillna(data_train.Age.mean() , inplace = True)
data_train['Age_discrete'] = (data_train.Age/10).astype(int)

# data_train['Age_flag'] = np.nan

# data_train.loc[data_train['Age']<=10 , 'Age_flag'] = '10'
# data_train.loc[(data_train['Age']>10) & (data_train['Age']<=20) , 'Age_flag'] = '10_20'
# data_train.loc[(data_train['Age']>20) & (data_train['Age']<=30) , 'Age_flag'] = '20_30'
# data_train.loc[(data_train['Age']>30) & (data_train['Age']<=40) , 'Age_flag'] = '30_40'
# data_train.loc[(data_train['Age']>40) & (data_train['Age']<=50) , 'Age_flag'] = '40_50'
# data_train.loc[(data_train['Age']>50) & (data_train['Age']<=60) , 'Age_flag'] = '50_60'
# data_train.loc[data_train['Age']>60 , 'Age_flag'] = '60'


Survived_0 = data_train.Age_discrete[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Age_discrete[data_train.Survived == 1].value_counts()

df = pd.DataFrame({u'获救':Survived_1 , u'未获救':Survived_0})
df.plot(kind = 'bar' , stacked = True)
plt.xlabel(u'年龄')
plt.ylabel(u'人数')

plt.show()