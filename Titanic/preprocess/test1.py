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

'''-------------------------训练---------------------------------'''
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

X = train_np[:,1:]
y = train_np[:,0]
from sklearn import cross_validation
from sklearn import linear_model

clf = linear_model.LogisticRegression(C=1.0 , penalty = 'l2')
print cross_validation.cross_val_score(clf, X, y, cv=5)

'''------------------------30%作为测试集------------------------------------'''
split_train, split_cv = cross_validation.train_test_split(train_np,test_size=0.3,random_state=0)
clf = linear_model.LogisticRegression(penalty = 'l2' , C=1.0)
clf.fit(split_train[:,1:], split_train[:,0])
print clf.score(split_cv[:,1:] , split_cv[:,0])

clf.fit(X, y)

'''--------------------------预测集处理----------------------------------------'''
data_test = pd.read_csv('../data/test.csv')

#填充缺失值Age Fare ，Cabin用yes和no填充
data_test.Age.fillna(data_test.Age.mean() , inplace = True)
data_test.Fare.fillna(data_test.Fare.mean() , inplace = True)
data_test.loc[data_test['Cabin'].notnull() , 'Cabin'] = 'yes'
data_test.loc[data_test['Cabin'].isnull() , 'Cabin'] = 'no'

#dummies转化
dummy_Pclass = pd.get_dummies(data_test['Pclass'] , prefix='Pclass')
dummy_Sex = pd.get_dummies(data_test['Sex'] , prefix='Sex')
dummy_Cabin = pd.get_dummies(data_test['Cabin'] , prefix='Cabin')
dummy_Embarked = pd.get_dummies(data_test['Embarked'] , prefix='Embarked')

df2 = pd.concat([data_test , dummy_Cabin, dummy_Embarked, dummy_Sex, dummy_Pclass] , axis=1)
df2.drop(['Pclass' , 'Cabin' , 'Sex' , 'Embarked'] , axis = 1 , inplace = True)

#scaling
df2['Age_scaled'] = scaler.fit_transform(df2['Age'], age_scale_param)
df2['Fare_scaled'] = scaler.fit_transform(df2['Fare'], age_scale_param)

test_set_df = df2.filter(regex = 'Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
test_set_np = test_set_df.as_matrix()

'''-----------------------预测-----------------------'''
Prediction = DataFrame({'Survived':clf.predict(test_set_np).astype(int)})


Result = pd.concat([data_test['PassengerId'] , Prediction] , axis = 1)


Result.to_csv('../LogisticRegression_prediction.csv' , index = False)