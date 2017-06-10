#coding=utf-8

import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
# import seaborn as sns
# import matplotlib

# import matplotlib.pyplot as plt
# from scipy.stats import skew
# from scipy.stats.stats import pearsonr
# from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

def print_ft_impts(featureColumns,bst):
	# 打印特征重要性
	FeatureImportance = DataFrame(bst.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)
	print FeatureImportance
	list1 = []
	for fNum in range(len(featureColumns)):
		list1.append('f'+str(fNum))
	FeatureDetail = DataFrame({'feature':list1,'FeatureDetail':featureColumns})
	# print pd.merge(FeatureImportance,FeatureDetail,on='feature',how = 'left')

def plot_ft_impts(featureColumns,bst):
	# plot特征重要性
	FeatureImportance = DataFrame(bst.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)
	list1 = []
	for fNum in range(len(featureColumns)):
		list1.append('f'+str(fNum))
	FeatureDetail = DataFrame({'feature':list1,'FeatureDetail':featureColumns})
	df = pd.merge(FeatureImportance,FeatureDetail,on='feature',how = 'left')
	s_ft = df['importance']
	s_ft.index = df['FeatureDetail']
	s_ft.plot(kind = "barh")

def tuning_para_for_xgb(feature_train,label_train):
	import xgboost as xgb
	from sklearn.grid_search import GridSearchCV   #Perforing grid search
# for tuning parameters
	parameters_for_testing = {
	   'colsample_bytree':[1],
	   'gamma':[0,0.03,0.1,0.3],
	   'min_child_weight':[1.5,6,10],
	   'learning_rate':[0.1,0.07,0.01],
	   'max_depth':[3,5],
	   'n_estimators':[20,35,50,100],
	   'reg_alpha':[1e-5, 1e-2,  0.75],
	   'reg_lambda':[1e-5, 1e-2, 0.45],
	   'subsample':[1]  
	}

	                    
	xgb_model = xgb.XGBRegressor(learning_rate =0.1, n_estimators=100, max_depth=5,
	    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=1,  seed=27)

	gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, iid=False, verbose=0,cv=4)
	gsearch1.fit(feature_train,label_train)
	print (gsearch1.grid_scores_)
	print('best params')
	print (gsearch1.best_params_)
	print('best score')
	print (gsearch1.best_score_)