#coding=utf-8
'''--------------在将年龄细分 , 提取姓名前缀 如Miss作为特征-------------'''
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
import matplotlib
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


data_train = pd.read_csv('../data/train.csv')
'''--PassengerId--Pclass--Name--Sex--Age--SibSp--Parch--Ticket--Fare--Cabin--Embarked'''


'''-------------缺失值填充(Age , Cabin , Embarked)-------'''
#Age用平均值填充
data_train.Age.fillna(data_train.Age.mean() , inplace = True)
data_train['Age_discrete'] = (data_train.Age/20).astype(int)


#Cabin有较多缺失值 , 将原Cabin分为有值1和无值0
data_train.loc[data_train.Cabin.notnull() , 'Cabin']='yes'
data_train.loc[data_train.Cabin.isnull() , 'Cabin']='no'


#Embarked 用较多数的种类填充
data_train.loc[data_train.Embarked.isnull() , 'Embarked'] = data_train.Embarked.value_counts().index[0]

#提取Name中的前缀 ，作为NameTitle列
data_train['NameTitle'] = data_train['Name'].str.split(',').str[-1].str.split('.').str[0].str.strip(' ')
# data_train.loc[(data_train.NameTitle!='Mr')&(data_train.NameTitle!='Miss')&(data_train.NameTitle!='Mrs')\
# 	&(data_train.NameTitle!='Master')&(data_train.NameTitle!='Dr')&(data_train.NameTitle!='Rev') , 'NameTitle'] = 'Orthers'

data_train.loc[(data_train.NameTitle!='Mr')\
	&(data_train.NameTitle!='Master')&(data_train.NameTitle!='Rev') , 'NameTitle'] = 'Orthers'



data_train['TicketType'] = data_train['Ticket'].str.split(' ').str[0].str[:1]
# data_train.loc[(data_train.TicketType!='3')&(data_train.TicketType!='2')&(data_train.TicketType!='1')\
# 	&(data_train.TicketType!='S')&(data_train.TicketType!='C')&(data_train.TicketType!='A')&(data_train.TicketType!='P')\
# 	&(data_train.TicketType!='W'), 'TicketType'] = 'Orthers'

data_train.loc[(data_train.TicketType!='3')&(data_train.TicketType!='2')&(data_train.TicketType!='1'), 'TicketType'] = 'Orthers'






# data_train['Is_single'] = 0
# data_train[(data_train['SibSp']==0) & (data_train['Parch']==0)]['Is_single'] = 1
# survived_0 = data_train[data_train['Survived']== 0]['Is_single'].value_counts()
# survived_1 = data_train[data_train['Survived']== 1]['Is_single'].value_counts()

# df = DataFrame({'Survived':survived_1,'Dead':survived_0})
# df.plot(kind = 'bar' , stacked = True)
# plt.show()
'''------------------------画图-----------------------------------------'''
# data_train['TicketType'].str[0]
# survived_0 = data_train[data_train['Survived']== 0]['TicketType'].value_counts()
# survived_1 = data_train[data_train['Survived']== 1]['TicketType'].value_counts()

# # df = DataFrame({'Survived':survived_1,'Dead':survived_0})
# # df.plot(kind = 'bar' , stacked = True)
# # plt.show()


'''----------------------哑变量转化----------------------------------'''
dummies_TicketType = pd.get_dummies(data_train['TicketType'], prefix='TicketType')

dummies_NameTitle = pd.get_dummies(data_train['NameTitle'], prefix = 'NameTitle')

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

dummies_DscrtAge = pd.get_dummies(data_train['Age_discrete'] , prefix='DscrtAge')

df = pd.concat([data_train,dummies_TicketType , dummies_NameTitle , dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass , dummies_DscrtAge], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked' , 'Age'], axis=1, inplace=True)
'''-----------------------------Scaling-------------------------------'''
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()

fare_scale_param = scaler.fit(df['Fare'].reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].reshape(-1,1), fare_scale_param)

'''-------------------------训练---------------------------------'''

train_df = df.filter(regex='Survived|NameTitle_.*|DscrtAge_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|TicketType_.*')
train_np = train_df.as_matrix()

X = train_np[:,1:]
y = train_np[:,0]
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC

clf = SVC(C = 5.0 ,  kernel='rbf')
# clf = RandomForestClassifier(n_estimators=300 , max_depth=4 , max_features=18)
print cross_validation.cross_val_score(clf, X, y, cv=5)

'''------------------------30%作为测试集------------------------------------'''
split_train, split_cv = cross_validation.train_test_split(train_np,test_size=0.3,random_state=0)
clf.fit(split_train[:,1:], split_train[:,0])

print clf.score(split_train[:,1:], split_train[:,0])
print clf.score(split_cv[:,1:] , split_cv[:,0])

clf.fit(X, y)

'''------------------------learning curve------------------------------------'''
# from sklearn.learning_curve import learning_curve

# # 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
#                         train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
#     '''
#     画出data在某模型上的learning curve.
#     参数解释
#     ----------
#     estimator : 你用的分类器。
#     title : 表格的标题。
#     X : 输入的feature，numpy类型
#     y : 输入的target vector
#     ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
#     cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
#     n_jobs : 并行的的任务数(默认1)
#     '''
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)

#     if plot:
#         plt.figure()
#         plt.title(title)
#         if ylim is not None:
#             plt.ylim(*ylim)
#         plt.xlabel(u"训练样本数")
#         plt.ylabel(u"得分")
#         # plt.gca().invert_yaxis()
#         plt.grid()

#         plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
#                          alpha=0.1, color="b")
#         plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
#                          alpha=0.1, color="r")
#         plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
#         plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

#         plt.legend(loc="best")

#         plt.draw()
#         plt.show()
#         plt.gca().invert_yaxis()

#     midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
#     diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
#     return midpoint, diff

# plot_learning_curve(clf, u"学习曲线", X, y)




# # '''--------------------------预测集处理----------------------------------------'''
data_test = pd.read_csv('../data/test.csv')

#填充缺失值Age Fare ，Cabin用yes和no填充
data_test.Age.fillna(data_test.Age.mean() , inplace = True)
data_test['Age_discrete'] = (data_test.Age/20).astype(int)

data_test.Fare.fillna(data_test.Fare.mean() , inplace = True)
data_test.loc[data_test['Cabin'].notnull() , 'Cabin'] = 'yes'
data_test.loc[data_test['Cabin'].isnull() , 'Cabin'] = 'no'

data_test['NameTitle'] = data_test['Name'].str.split(',').str[-1].str.split('.').str[0].str.strip(' ')
# data_test.loc[(data_test.NameTitle!='Mr')&(data_test.NameTitle!='Miss')&(data_test.NameTitle!='Mrs')\
# 	&(data_test.NameTitle!='Master')&(data_test.NameTitle!='Dr')&(data_test.NameTitle!='Rev') , 'NameTitle'] = 'Orthers'
data_test.loc[(data_test.NameTitle!='Mr')\
	&(data_test.NameTitle!='Master')&(data_test.NameTitle!='Rev') , 'NameTitle'] = 'Orthers'

data_test['TicketType'] = data_test['Ticket'].str.split(' ').str[0].str[:1]
# data_test.loc[(data_test.TicketType!='3')&(data_test.TicketType!='2')&(data_test.TicketType!='1')\
# 	&(data_test.TicketType!='S'), 'TicketType'] = 'Orthers'
data_test.loc[(data_test.TicketType!='3')&(data_test.TicketType!='2')&(data_test.TicketType!='1'), 'TicketType'] = 'Orthers'





# #dummies转化
dummy_TicketType = pd.get_dummies(data_test['TicketType'] , prefix='TicketType')
dummy_NameTitle = pd.get_dummies(data_test['NameTitle'], prefix = 'NameTitle')
dummy_Pclass = pd.get_dummies(data_test['Pclass'] , prefix='Pclass')
dummy_Sex = pd.get_dummies(data_test['Sex'] , prefix='Sex')
dummy_Cabin = pd.get_dummies(data_test['Cabin'] , prefix='Cabin')
dummy_Embarked = pd.get_dummies(data_test['Embarked'] , prefix='Embarked')
dummy_DscrtAge = pd.get_dummies(data_test['Age_discrete'] , prefix = 'DscrtAge')

df2 = pd.concat([data_test , dummy_TicketType, dummy_NameTitle , dummy_Cabin, dummy_Embarked, dummy_Sex, dummy_Pclass , dummy_DscrtAge] , axis=1)
df2.drop(['Pclass' , 'Cabin' , 'Sex' , 'Embarked' , 'Age'] , axis = 1 , inplace = True)
df2['DscrtAge_4'] = 0.0

print df.info()
print df2.info()

df2['Fare_scaled'] = scaler.fit_transform(df2['Fare'], fare_scale_param)

test_set_df = df2.filter(regex = 'NameTitle_.*|DscrtAge.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|TicketType_')
test_set_np = test_set_df.as_matrix()

# '''-----------------------预测-----------------------'''
Prediction = DataFrame({'Survived':clf.predict(test_set_np).astype(int)})


Result = pd.concat([data_test['PassengerId'] , Prediction] , axis = 1)

print 'total survived' , Result.Survived.sum()
# Result.to_csv('../SVM_TicketType.csv' , index = False)

