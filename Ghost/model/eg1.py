#coding=utf-8

# https://www.kaggle.com/artgor/ghouls-goblins-and-ghosts-boo/eda-and-models-score-0-74291/comments

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('whitegrid')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn import svm

'''
Data exploration
'''
train = pd.read_csv('../raw data/train.csv')
test = pd.read_csv('../raw data/test.csv')

print train.describe(include='all')

# 每种连续变量 在 不同类中的均值
plt.subplot(1,4,1)
train.groupby('type').mean()['rotting_flesh'].plot(kind='bar',figsize=(7,4), color='r')
plt.subplot(1,4,2)
train.groupby('type').mean()['bone_length'].plot(kind='bar',figsize=(7,4), color='g')
plt.subplot(1,4,3)
train.groupby('type').mean()['hair_length'].plot(kind='bar',figsize=(7,4), color='y')
plt.subplot(1,4,4)
train.groupby('type').mean()['has_soul'].plot(kind='bar',figsize=(7,4), color='teal')
plt.show()
# # 每种颜色在不同类中的分布
# sns.factorplot("type", col="color", col_wrap=4, data=train, kind="count", size=2.4, aspect=.8)

# fig, ax = plt.subplots(2, 2, figsize = (16, 12))
# sns.pointplot(x="color", y="rotting_flesh", hue="type", data=train, ax = ax[0, 0])
# sns.pointplot(x="color", y="bone_length", hue="type", data=train, ax = ax[0, 1])
# sns.pointplot(x="color", y="hair_length", hue="type", data=train, ax = ax[1, 0])
# sns.pointplot(x="color", y="has_soul", hue="type", data=train, ax = ax[1, 1])

# # pairplot
# sns.pairplot(train, hue='type')
# plt.show()

# '''
# This pairplot shows that data is distributed normally. 
# And while most pairs are widely scattered (in relationship to the type), some of them show clusters:
#  hair_length and has_soul, hair_length and bone_length. 
# I decided to create new variables with multiplication of these columns and it worked great!
# '''

# '''
# Data preparation
# '''
# train['hair_soul'] = train['hair_length'] * train['has_soul']
# train['hair_bone'] = train['hair_length'] * train['bone_length']
# test['hair_soul'] = test['hair_length'] * test['has_soul']
# test['hair_bone'] = test['hair_length'] * test['bone_length']
# train['hair_soul_bone'] = train['hair_length'] * train['has_soul'] * train['bone_length']
# test['hair_soul_bone'] = test['hair_length'] * test['has_soul'] * test['bone_length']

# #test_id will be used later, so save it
# test_id = test['id']
# train.drop(['id'], axis=1, inplace=True)
# test.drop(['id'], axis=1, inplace=True)

# #Deal with 'color' column
# col = 'color'
# dummies = pd.get_dummies(train[col], drop_first=False)
# dummies = dummies.add_prefix("{}#".format(col))
# train.drop(col, axis=1, inplace=True)
# train = train.join(dummies)
# dummies = pd.get_dummies(test[col], drop_first=False)
# dummies = dummies.add_prefix("{}#".format(col))
# test.drop(col, axis=1, inplace=True)
# test = test.join(dummies)

# X_train = train.drop('type', axis=1)
# le = LabelEncoder()
# Y_train = le.fit_transform(train.type.values)
# X_test = test

# clf = RandomForestClassifier(n_estimators=200)
# clf = clf.fit(X_train, Y_train)
# indices = np.argsort(clf.feature_importances_)[::-1]

# # Print the feature ranking
# print('Feature ranking:')

# for f in range(X_train.shape[1]):
#     print('%d. feature %d %s (%f)' % (f + 1, indices[f], X_train.columns[indices[f]],
#                                       clf.feature_importances_[indices[f]]))

# '''
# Graphs and model show that color has little impact, so I won't use it. 
# In fact I tried using it, but the result got worse. And three features, which I created, seem to be important!
# '''
# best_features=X_train.columns[indices[0:7]]
# X = X_train[best_features]
# Xt = X_test[best_features]

# #Splitting data for validation
# Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y_train, test_size=0.20, random_state=36)


# '''
# Tune the model. 
# '''
# svc = svm.SVC(kernel='linear')
# svc.fit(Xtrain, ytrain)
# y_val_s = svc.predict(Xtest)
# print("Validation accuracy: ", sum(le.inverse_transform(y_val_s)
#                                    == le.inverse_transform(ytest))/1.0/len(ytest))


# # #The last model is logistic regression
# # logreg = LogisticRegression()

# # parameter_grid = {'solver' : ['newton-cg', 'lbfgs'],
# #                   'multi_class' : ['ovr', 'multinomial'],
# #                   'C' : [0.005, 0.01, 1, 10, 100, 1000],
# #                   'tol': [0.0001, 0.001, 0.005]
# #                  }

# # grid_search = GridSearchCV(logreg, param_grid=parameter_grid, cv=StratifiedKFold(5))
# # grid_search.fit(Xtrain, ytrain)
# # print('Best score: {}'.format(grid_search.best_score_))
# # print('Best parameters: {}'.format(grid_search.best_params_))

# log_reg = LogisticRegression(C = 1, tol = 0.0001, solver='newton-cg', multi_class='multinomial')
# log_reg.fit(Xtrain, ytrain)
# y_val_l = log_reg.predict_proba(Xtest)
# print("Validation accuracy: ", sum(pd.DataFrame(y_val_l, columns=le.classes_).idxmax(axis=1).values
#                                    == le.inverse_transform(ytest))/len(ytest))



# svc = svm.SVC(kernel='linear')
# svc.fit(X, Y_train)
# svc_pred = svc.predict(Xt)

# clf = RandomForestClassifier(n_estimators=20, n_jobs=-1, criterion = 'gini', max_features = 'sqrt',
#                              min_samples_split=2, min_weight_fraction_leaf=0.0,
#                              max_leaf_nodes=40, max_depth=100)

# calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
# calibrated_clf.fit(X, Y_train)
# for_pred = calibrated_clf.predict_proba(Xt)

# log_reg.fit(X, Y_train)
# log_pred = log_reg.predict_proba(Xt)

# #I decided to try adding xgboost.
# params = {"objective": "multi:softprob", "num_class": 3, 'eta': 0.01, 'min_child_weight' : 10, 'max_depth': 5}
# param = list(params.items())
# gbm = xgb.train(params, xgb.DMatrix(X, Y_train), 300)
# x_pred = gbm.predict(xgb.DMatrix(Xt))

# #Predicted values
# s = le.inverse_transform(svc_pred)
# l = pd.DataFrame(log_pred, columns=le.classes_).idxmax(axis=1).values
# f = pd.DataFrame(for_pred, columns=le.classes_).idxmax(axis=1).values
# x = pd.DataFrame(x_pred, columns=le.classes_).idxmax(axis=1).values
# #Average of models, which give probability predictions.
# q = pd.DataFrame(((log_pred + for_pred + x_pred)/3), columns=le.classes_).idxmax(axis=1).values  


# submission = pd.DataFrame({'id':test_id, 'type':l})
# submission.to_csv('GGG_submission.csv', index=False)                                