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
'''--------------------------------------------------------------------------------------------------------'''
'''----------------------缺失值填充----------------------------------'''
def set_missing_ages(df):
    from sklearn.ensemble import RandomForestRegressor
# 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df


data_train = pd.read_csv('../data/train.csv')
data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

'''------------------------哑变量转化----------------------------------'''
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
'''-------------------------------Scaling--------------------------------------'''
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)

print df['Fare_scaled'].max()
'''----------------------------训练模型---------------------------------'''
##from sklearn import linear_model
##
### 用正则取出我们要的属性值
##train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
##train_np = train_df.as_matrix()
##
### y即Survival结果
##y = train_np[:, 0]
##
### X即特征属性值
##X = train_np[:, 1:]
##
### fit到RandomForestRegressor之中
##clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
##clf.fit(X, y)
##
##pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})
##'''-------------------------Cross Validation----------------------------------'''
##from sklearn import cross_validation
##
## #简单看看打分情况
##clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
##all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
##X = all_data.as_matrix()[:,1:]
##y = all_data.as_matrix()[:,0]
##print cross_validation.cross_val_score(clf, X, y, cv=5)
##
##'''-------------测试'''
### 分割数据，按照 训练数据:cv数据 = 7:3的比例
##split_train, split_cv = cross_validation.train_test_split(df,test_size=0.3,random_state=0)
##train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
### 生成模型
##clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
##clf.fit(train_df.as_matrix()[:,1:], train_df.as_matrix()[:,0])
##
### 对cross validation数据进行预测
##cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
##predictions = clf.predict(cv_df.as_matrix()[:,1:])
##
##print clf.score(cv_df.as_matrix()[:,1:] , cv_df.as_matrix()[:,0])
####origin_data_train = pd.read_csv("../data/Train.csv")
####bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
##
##
