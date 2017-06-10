#coding=utf-8
'''--------------在1的基础上将年龄细分-------------'''
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
import matplotlib
import matplotlib.pyplot as plt
from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


data_train = pd.read_csv('../../data/train.csv')
'''--PassengerId--Pclass--Name--Sex--Age--SibSp--Parch--Ticket--Fare--Cabin--Embarked'''


'''-------------缺失值填充(Age , Cabin , Embarked)-------'''
#Age用平均值填充
data_train.Age.fillna(data_train.Age.mean() , inplace = True)
data_train['Age_discrete'] = (data_train.Age/10).astype(int)


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

dummies_DscrtAge = pd.get_dummies(data_train['Age_discrete'] , prefix='DscrtAge')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass , dummies_DscrtAge], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked' , 'Age'], axis=1, inplace=True)
'''-----------------------------Scaling-------------------------------'''
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()

fare_scale_param = scaler.fit(df['Fare'].reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].reshape(-1,1), fare_scale_param)

'''-------------------------训练---------------------------------'''

train_df = df.filter(regex='Survived|DscrtAge_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

X = train_np[:,1:]
y = train_np[:,0]
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier 

clf = linear_model.LogisticRegression(C=1.0 , penalty = 'l2')
# clf = RandomForestClassifier(n_estimators=300 , max_depth=4 , max_features=19)

print cross_validation.cross_val_score(clf, X, y, cv=5)

'''-----------------------------bagging 集成-----------------------------------'''
from sklearn.ensemble import BaggingRegressor
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=1)
bagging_clf.fit(X, y)

# '''------------------------30%作为测试集------------------------------------'''
split_train, split_cv = cross_validation.train_test_split(train_np,test_size=0.3,random_state=0)
bagging_clf.fit(split_train[:,1:], split_train[:,0])

print bagging_clf.score(split_train[:,1:], split_train[:,0])
print bagging_clf.score(split_cv[:,1:] , split_cv[:,0])

# clf.fit(X, y)

# W_vec = DataFrame({'features':train_df.columns[1:],'coef':clf.coef_[0]})
# print W_vec



'''--------------------------预测集处理----------------------------------------'''
data_test = pd.read_csv('../../data/test.csv')

#填充缺失值Age Fare ，Cabin用yes和no填充
data_test.Age.fillna(data_test.Age.mean() , inplace = True)
data_test['Age_discrete'] = (data_test.Age/10).astype(int)

data_test.Fare.fillna(data_test.Fare.mean() , inplace = True)
data_test.loc[data_test['Cabin'].notnull() , 'Cabin'] = 'yes'
data_test.loc[data_test['Cabin'].isnull() , 'Cabin'] = 'no'

#dummies转化
dummy_Pclass = pd.get_dummies(data_test['Pclass'] , prefix='Pclass')
dummy_Sex = pd.get_dummies(data_test['Sex'] , prefix='Sex')
dummy_Cabin = pd.get_dummies(data_test['Cabin'] , prefix='Cabin')
dummy_Embarked = pd.get_dummies(data_test['Embarked'] , prefix='Embarked')
dummy_DscrtAge = pd.get_dummies(data_test['Age_discrete'] , prefix = 'DscrtAge')

df2 = pd.concat([data_test , dummy_Cabin, dummy_Embarked, dummy_Sex, dummy_Pclass , dummy_DscrtAge] , axis=1)
df2.drop(['Pclass' , 'Cabin' , 'Sex' , 'Embarked' , 'Age'] , axis = 1 , inplace = True)
df2['DscrtAge_8'] = 0.0
#scaling

df2['Fare_scaled'] = scaler.fit_transform(df2['Fare'], fare_scale_param)

test_set_df = df2.filter(regex = 'DscrtAge.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
test_set_np = test_set_df.as_matrix()

'''-----------------------预测-----------------------'''
Prediction = DataFrame({'Survived':clf.predict(test_set_np).astype(int)})


Result = pd.concat([data_test['PassengerId'] , Prediction] , axis = 1)

print 'total survived' , Result.Survived.sum()
# Result.to_csv('../../RF_prediction.csv' , index = False)