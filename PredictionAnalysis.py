import os
import random
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import decomposition
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from datetime import datetime,date
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.externals.six import StringIO
import pydot


charts_path = 'charts/'
tables_path = 'tables/'

predictionperformance = []




def normalizationwithzscore(alldataframe, targets=False, normalize_targets=False):
	# take out infinite values
	alldataframe = alldataframe.replace([np.inf, -np.inf], np.nan)
	alldataframe = alldataframe.dropna()

	if targets==True:
		# take out date
		date = alldataframe['date']
		alldataframe = alldataframe.drop(['date'], axis=1)

		if normalize_targets == False:
			point_differential = alldataframe['point_differential']
			spread = alldataframe['spread']
			home_win = alldataframe['home_win']
			alldataframe = alldataframe.drop(['point_differential', 'spread', 'home_win'], axis=1)

	# normalize the dataset
	alldataframe = (alldataframe - alldataframe.mean()) / alldataframe.std()

	# if targets == True, reappend the target attributes
	if targets == True:
		alldataframe['date'] = date
		alldataframe['point_differential'] = point_differential
		alldataframe['spread'] = spread
		alldataframe['home_win'] = home_win

	# return the normalized dataset
	return alldataframe	

def selectattributes(alldataframe, basic=True, std=True, trend=True, l10=False, l5=False, l3=False, backward_elimination=False, backward_elimination_extended=False,stepwise=False):
	# keep the date
	alldataframe_new = alldataframe['date']

	# add target variables
	alldataframe_new = pd.concat([alldataframe_new,
						alldataframe['point_differential'],
						alldataframe['spread'],
						alldataframe['home_win']], axis=1)

	if backward_elimination == True:
		alldataframe_new = pd.concat([alldataframe_new,
							alldataframe['difference_atsstreak_season'],
							alldataframe['ratio_margin_season_l3'],
							alldataframe['ratio_fastbreakpoints_l3'],
							alldataframe['ratio_atsmargin_l10'],
							alldataframe['ratio_blocks_season'],
							alldataframe['ratio_pointsinthepaint'],
							alldataframe['ratio_steal2blockl5'],
							alldataframe['ratio_3pointfg_l10'],
							alldataframe['ratio_fouls_season_l10'],
							alldataframe['ratio_fouls_l3'],
							alldataframe['ratio_fouls_l10_l3'],
							alldataframe['trend_points_season_l10_ratio'],
							alldataframe['ratio_steals_l3'],
							alldataframe['ratio_blocks_season_l10'],
							alldataframe['ratio_pointsinthepaint_l3'],
							alldataframe['ratio_turnovers_season'],
							alldataframe['ratio_fouls_l10'],
							alldataframe['ratio_fastbreakpoints_season'],
							alldataframe['ratio_steals_l5'],
							alldataframe['fgpct_season_vs_l10_ratio'],
							alldataframe['ratio_steals_season'],
							alldataframe['ratio_defensiverebounds_season'],
							alldataframe['ratio_streak_season'],
							alldataframe['ratio_freethrowfg'],
							alldataframe['ratio_fg_l10'],
							alldataframe['ratio_biggestlead_season']], axis=1)
		alldataframe_new = alldataframe_new.replace([np.inf, -np.inf], np.nan)
		alldataframe_new = alldataframe_new.dropna()
		return alldataframe_new

	if backward_elimination_extended == True:
		alldataframe_new = pd.concat([alldataframe_new,
							alldataframe['ratio_3pointfg_l10'],
							alldataframe['ratio_fouls_season_l10'],
							alldataframe['ratio_fouls_l3'],
							alldataframe['ratio_fouls_l10_l3'],
							alldataframe['trend_points_season_l10_ratio'],
							alldataframe['ratio_steals_l3'],
							alldataframe['ratio_blocks_season_l10'],
							alldataframe['ratio_pointsinthepaint_l3'],
							alldataframe['ratio_turnovers_season'],
							alldataframe['ratio_fouls_l10'],
							alldataframe['ratio_fastbreakpoints_season'],
							alldataframe['ratio_steals_l5'],
							alldataframe['fgpct_season_vs_l10_ratio'],
							alldataframe['ratio_steals_season'],
							alldataframe['ratio_defensiverebounds_season'],
							alldataframe['ratio_streak_season'],
							alldataframe['ratio_freethrowfg'],
							alldataframe['ratio_fg_l10'],
							alldataframe['ratio_biggestlead_season']], axis=1)
		alldataframe_new = alldataframe_new.replace([np.inf, -np.inf], np.nan)
		alldataframe_new = alldataframe_new.dropna()
		return alldataframe_new


	if basic == True:
		# variables with highest correlation
		alldataframe_new = pd.concat([alldataframe_new,
							alldataframe['ratio_biggestlead_season'],
							alldataframe['ratio_fg'],
							alldataframe['ratio_streak_season'],
							alldataframe['ratio_defensiverebounds_season'],
							alldataframe['ratio_assists_season'],
							alldataframe['ratio_winperc'],
							alldataframe['ratio_fouls_season']], axis=1)
	
	if std == True:
		alldataframe_new = pd.concat([alldataframe_new,
						alldataframe['ratio_atsmargin_season'],
						alldataframe['difference_atsstreak_season'],
						alldataframe['ratio_spread_season'],
						alldataframe['ratio_blocks_season'],
						alldataframe['ratio_fastbreakpoints_season'],
						alldataframe['ratio_marginHalfvsFull_season'],
						alldataframe['ratio_margin_season'],
						alldataframe['ratio_orebounds_season'],
						alldataframe['ratio_freethrowfg'],
						alldataframe['ratio_pointsinthepaint'],
						alldataframe['difference_rest_season'],
						alldataframe['ratio_steals_season'],
						alldataframe['ratio_steal2block'],
						alldataframe['ratio_teamrebounds_season'],
						alldataframe['ratio_3pointfg_season'],
						alldataframe['ratio_turnovers_season']], axis=1)

	if trend == True:
		alldataframe_new = pd.concat([alldataframe_new,
						alldataframe['fgpct_l10_vs_l3_ratio'],
						alldataframe['fgpct_season_vs_l10_ratio'],
						alldataframe['ratio_fouls_l10_l3'],
						alldataframe['ratio_fouls_season_l10'],
						alldataframe['ratio_margin_l10_l3'],
						alldataframe['ratio_margin_season_l3'],
						alldataframe['ratio_margin_season_l5'],
						alldataframe['trend_points_l10_l3_ratio'],
						alldataframe['trend_points_season_l10_ratio']], axis=1)		

	if l10 == True:
		alldataframe_new = pd.concat([alldataframe_new,
							alldataframe['ratio_biggestlead_season_l10'],
							alldataframe['ratio_fg_l10'],
							alldataframe['ratio_defensiverebounds_season_l10'],
							alldataframe['ratio_assists_season_l10'],
							alldataframe['ratio_fouls_l10'],
							alldataframe['ratio_atsmargin_season_l10'],
							alldataframe['ratio_blocks_season_l10'],
							alldataframe['ratio_fastbreakpoints_season_l10'],
							alldataframe['ratio_marginHalfvsFull_l10'],
							alldataframe['ratio_margin_season_l10'],
							alldataframe['ratio_orebounds_l10'],
							alldataframe['ratio_freethrowfg_l10'],
							alldataframe['ratio_pointsinthepaint_l10'],
							alldataframe['ratio_steals_season_l10'],
							alldataframe['ratio_steal2blockl10'],
							alldataframe['ratio_teamrebounds_season_l10'],
							alldataframe['ratio_3pointfg_season_l10'],
							alldataframe['ratio_turnovers_season_l10']], axis=1)

	if l5 == True:
		alldataframe_new = pd.concat([alldataframe_new,
							alldataframe['ratio_biggestlead_season_l5'],
							alldataframe['ratio_fg_l5'],
							alldataframe['ratio_defensiverebounds_season_l5'],
							alldataframe['ratio_assists_season_l5'],
							alldataframe['ratio_fouls_season_l5'],
							alldataframe['ratio_atsmargin_season_l5'],
							alldataframe['ratio_blocks_season_l5'],
							alldataframe['ratio_fastbreakpoints_season_l5'],
							alldataframe['ratio_marginHalfvsFull_l5'],
							alldataframe['ratio_margin_season_l5'],
							alldataframe['ratio_orebounds_l5'],
							alldataframe['ratio_freethrowfg_l5'],
							alldataframe['ratio_pointsinthepaint_l5'],
							alldataframe['ratio_steals_l5'],
							alldataframe['ratio_steal2blockl5'],
							alldataframe['ratio_teamrebounds_season_l5'],
							alldataframe['ratio_3pointfg_season_l5'],
							alldataframe['ratio_turnovers_season_l5']], axis=1)

	if l3 == True:
		alldataframe_new = pd.concat([alldataframe_new,
							alldataframe['ratio_biggestlead_season_l3'],
							alldataframe['ratio_fg_l3'],
							alldataframe['ratio_defensiverebounds_season_l3'],
							alldataframe['ratio_assists_season_l3'],
							alldataframe['ratio_fouls_l3'],
							alldataframe['ratio_atsmargin_season_l3'],
							alldataframe['ratio_blocks_season_l3'],
							alldataframe['ratio_fastbreakpoints_l3'],
							alldataframe['ratio_marginHalfvsFull_l10'],
							alldataframe['ratio_margin_season_l3'],
							alldataframe['ratio_orebounds_l3'],
							alldataframe['ratio_freethrowfg_l3'],
							alldataframe['ratio_pointsinthepaint_l3'],
							alldataframe['ratio_steals_l3'],
							alldataframe['ratio_steal2blockl3'],
							alldataframe['ratio_teamrebounds_season_l3'],
							alldataframe['ratio_3pointfg_season_l3'],
							alldataframe['ratio_turnovers_season_l3']], axis=1)

	alldataframe_new = alldataframe_new.replace([np.inf, -np.inf], np.nan)
	alldataframe_new = alldataframe_new.dropna()
	return alldataframe_new


def pca_setcreation(alldataframe, perc_exp_variance=0.5, targets=False, de=False):
	original_index = alldataframe.index.values

	if targets == True:
		date = alldataframe['date']
		point_differential = alldataframe['point_differential']
		spread = alldataframe['spread']
		home_win = alldataframe['home_win']
		alldataframe = alldataframe.drop(['date', 'point_differential', 'spread', 'home_win'], axis=1)

	m = decomposition.PCA()
	m.fit(alldataframe)

	# find the number of components to keep for perc_exp_variance
	explained_variance = m.explained_variance_ratio_
	ev_sum = 0
	n_components = 0
	for v in explained_variance*100:
		n_components += 1
		ev_sum += v
		if ev_sum >= perc_exp_variance:
			break

	# draw up screeplot
	plt.plot(range(10), explained_variance[0:10])
	plt.xlabel('Component')
	plt.ylabel('% Of Variance In Data Explained By Component')
	plt.title('PCA: Screeplot')
	plt.savefig(charts_path + 'screeplot.png')
	plt.clf()

	# print the components
	alldataframe_components = pd.DataFrame(m.components_)
	alldataframe_components = alldataframe_components.ix[:,0:8]
	alldataframe_components.index = alldataframe.columns
	alldataframe_components.to_csv(tables_path + 'pca_analysis.csv')

	# transform the dataset
	alldataframe_pca = pd.DataFrame(m.transform(alldataframe))
	alldataframe_pca.index = original_index
	alldataframe_pca = alldataframe_pca[list(alldataframe_pca.columns[:n_components])]

	# add attribute names
	attribute_names = []
	for i in range(n_components):
		attribute_names.append('Cluster%s' % str(i))
	alldataframe_pca.columns = attribute_names

	# add target attributes back if targets is True
	if targets == True:
		alldataframe_pca['date'] = date
		alldataframe_pca['point_differential'] = point_differential
		alldataframe_pca['spread'] = spread
		alldataframe_pca['home_win'] = home_win	

	# print number of components kept and variance explained
	print '### PCA ###'
	print 'Percent of variance which is explained: %0.2f' % ev_sum
	print 'Number of components: %d' % n_components

	if de == True:
		return m.mean_,
	else:
		return alldataframe_pca

	

def findingcorrelation(alldataframe):
	alldataframe = alldataframe.drop(['date',], axis=1)
	correlation_matrix = alldataframe.corr(method='pearson')
	correlation_matrix.to_csv(tables_path + 'correlation_analysis.csv')


def findingscattering(alldataframe):
	target = alldataframe['point_differential']
	alldataframe = alldataframe.drop(['date', 'home_win', 'point_differential'], axis=1)
	attributes = alldataframe.columns
	for a in attributes:
		plt.scatter(alldataframe[a], target)
		plt.title('%s' % str(a))
		plt.ylabel('Point Differential')
		plt.xlabel('Scatterplot: point_differentialerential vs. %s' % str(a))

		if a == 'ratio_biggestlead_season':
			plt.xlim([0, 4])
		elif a in ['ratio_fg', 'ratio_defensiverebounds_season', 'ratio_assists_season', 'ratio_fouls_season']:
			plt.xlim([0.8, 1.2])
		elif a == 'ratio_streak_season':
			plt.xlim([-10, 10])
		elif a == 'ratio_winperc':
			plt.xlim([0.1, 4])

		plt.savefig(charts_path + 'scatter_%s.png' % str(a))
		plt.clf()


def exploringtarget(alldataframe):
	alldataframe['point_differential'].hist(bins=30,alpha=0.5)
	alldataframe['spread'].hist(bins=15, color='red', alpha=0.5)
	plt.title('Distribution of Point Differential and Spread')
	plt.ylabel('Frequency')
	plt.xlabel('Away Points - Home Points')
	plt.legend(['point differential','spread'])
	plt.savefig(charts_path + 'point_differential_spread_histogram.pdf')
	plt.clf()
	print 'Point Differential Mean: %.2f' % alldataframe['point_differential'].mean()
	print 'Point Differential Std: %.2f' % alldataframe['point_differential'].std()
	print 'Spread Mean: %.2f' % alldataframe['spread'].mean()
	print 'Spread Std: %.2f' % alldataframe['spread'].std()

	# create distribution of binary target variable
	homewin_vc = alldataframe['home_win'].value_counts()
	homewin_vc.index = ['Home', 'Away']
	print homewin_vc
	homewin_vc.plot(kind='barh',
					grid=True,
					title='Distribution Of ATS Wins For Home & Away Team',
					alpha=0.8)
	plt.savefig(charts_path + 'target_binary_distribution.png')



def treebalancegrowth(Matches,max_depth=2):
	balance = 1
	bet_amount = 0.05

	balances = []

	Matches = Matches.sort('date')
	unique_days = Matches['date'].unique()
	Matches['date'] = pd.to_datetime(Matches['date'])

	for day in unique_days:
		train = Matches[Matches['date'] < pd.to_datetime(day)]
		test = Matches[Matches['date'] == pd.to_datetime(day)]

		if train.shape[0] < 100:
			continue

		y_train_binary = train['home_win']
		y_test_binary = test['home_win']
		x_train = train.drop(['point_differential', 'home_win', 'date'], axis=1)
		x_test = test.drop(['point_differential', 'home_win', 'date'], axis=1)

		# train the model
		m = tree.DecisionTreeClassifier(max_depth=max_depth)
		m.fit(x_train, y_train_binary)
		pred_bin_test = m.predict(x_test)
		pred_bin_test = pd.Series(pred_bin_test)
		pred_bin_test.index = y_test_binary.index

		try:
			cm = confusion_matrix(pred_bin_test, y_test_binary)
			num_correct = cm[0][0] + cm[1][1]
			grow_balance = num_correct*((bet_amount*balance)/1.10)
			num_incorrect = cm[0][1] + cm[1][0]
			reduce_balance = num_incorrect*(bet_amount*balance)

			balance += grow_balance
			balance -= reduce_balance
			balances.append(balance)
		except IndexError: # cm not complete
			pass

		if balance <= 0:
			print 'BUST'
			return

	# plt the balances
	plt.plot(balances)
	plt.title('Growth Of Balance (5% Bets) - Tree2 - PCA1')
	plt.xlabel('Day')
	plt.ylabel('Balance ($)')
	plt.savefig(charts_path + 'balance_growth.png')




def gamestream(Matches,algochoosed='linear_regression',alpha=0.1,penalty='l1',max_depth=10,quick_comp=False):
	Matches = Matches.sort('date')
	unique_days = Matches['date'].unique()
	Matches['date'] = pd.to_datetime(Matches['date'])

	predictions = pd.Series()
	predictions_num = pd.Series()
	actual = pd.Series()
	for day in unique_days:
		train = Matches[Matches['date'] < pd.to_datetime(day)]
		test = Matches[Matches['date'] == pd.to_datetime(day)]

		if train.shape[0] < 100:
			continue

		y_train_binary = train['home_win']
		y_test_binary = test['home_win']
		y_train_num = train['point_differential']
		y_test_num = test['point_differential']
		x_train = train.drop(['point_differential', 'home_win', 'date'], axis=1)
		x_test = test.drop(['point_differential', 'home_win', 'date'], axis=1)

		# train the model and make predictions
		if algochoosed == 'linear_regression':
			m = linear_model.LinearRegression()
			m.fit(x_train, y_train_num)
			pred_num_test = m.predict(x_test)
			pred_num_test = pd.Series(pred_num_test, index=x_test.index)
			pred_bin_test = ((pred_num_test - x_test['spread']) < 0)*1
		elif algochoosed == 'logistic_regression':
			m = linear_model.LogisticRegression()
			m.fit(x_train, y_train_binary)
			pred_bin_test = m.predict(x_test)
			pred_bin_test = pd.Series(pred_bin_test)
			pred_bin_test.index = y_test_binary.index
		elif algochoosed == 'ridge':
			m = linear_model.Ridge(alpha=alpha)
			m.fit(x_train, y_train_num)
			pred_num_test = m.predict(x_test)
			pred_bin_test = ((pred_num_test - x_test['spread']) < 0)*1
		elif algochoosed == 'tree':
			m = tree.DecisionTreeClassifier(max_depth=10)
			m.fit(x_train, y_train_binary)
			pred_bin_test = m.predict(x_test)
			pred_bin_test = pd.Series(pred_bin_test)
			pred_bin_test.index = y_test_binary.index
		elif algochoosed == 'randomforest':

			imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
			x_train = imp.fit_transform(x_train)
			x_test = imp.fit_transform(x_test)
			clf = ensemble.ExtraTreesClassifier(n_estimators = 100)
			lsvc = clf.fit(x_train, y_train_binary)
			model = SelectFromModel(lsvc, prefit = True)

			Train_new = model.transform(x_train)
			#print Train_new.shape
			newindices = model.get_support(True)
			#print "IN Random"
			m = ensemble.GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.1, max_depth = 10 , random_state = 0)

			FinalTrainLessFeature = x_train[:,newindices]
			FinalTestLessFeature = x_test[:,newindices]
			#print y_train_binary

			m.fit(FinalTrainLessFeature, y_train_binary)
			
			pred_bin_test = m.predict(FinalTestLessFeature)
			pred_bin_test = pd.Series(pred_bin_test)
			pred_bin_test.index = y_test_binary.index
		elif algochoosed == 'GradientTrees':
			m = ensemble.GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.1, max_depth = 10 , random_state = 0)
			m.fit(x_train, y_train_binary)
			pred_bin_test = m.predict(x_test)
			pred_bin_test = pd.Series(pred_bin_test)
			pred_bin_test.index = y_test_binary.index	



		if algochoosed in ['linear_regression']:
			predictions_num = predictions_num.append(pred_num_test)
		# print predictions	
		predictions = predictions.append(pred_bin_test)
		actual = actual.append(y_test_binary)

	if algochoosed in ['linear_regression']:
		Matches['pred_num'] = predictions_num
	Matches['pred'] = predictions
	Matches['actual'] = actual

	# drop rows with missing data (first n games)
	Matches = Matches.dropna()
	return Matches, m



def evaluate(alldataframe, label='nolabel', numerical=False ):
	if numerical == True:
		# calculate rmse
		rmse = np.sqrt(mean_squared_error(alldataframe['point_differential'], alldataframe['pred_num']))

		# plot actual vs pred
		plt.scatter(alldataframe['pred_num'], alldataframe['point_differential'], marker='o')
		plt.title('Predicted vs. Actual (%s)' % label)
		plt.xlabel('Predicted')
		plt.ylabel('Actual')
		plt.xlim([-30,30])
		plt.ylim([-30,30])
		plt.savefig(charts_path + 'predvsactual_%s.png' % label)
		plt.clf()
		initialval = 0.1

		# plot residuals
		alldataframe['residuals'] = alldataframe['point_differential'] - alldataframe['pred_num']
		plt.scatter(alldataframe['pred_num'], alldataframe['residuals'], marker='o')
		plt.title('Residual Analysis (%s)' % label)
		plt.xlabel('Predicted Value')
		plt.ylabel('Residual')
		plt.xlim([-15,15])
		plt.ylim([-50,50])
		plt.savefig(charts_path + 'residualvspred_%s.png' % label)
		plt.clf()

		# plt spread vs residuals
		plt.scatter(alldataframe['spread'], alldataframe['residuals'], marker='o')
		plt.title('Residual Analysis (%s)' % label)
		plt.xlabel('Spread')
		plt.ylabel('Residual')
		plt.xlim([-15,15])
		plt.ylim([-50,50])
		plt.savefig(charts_path + 'residualvsspread_%s.png' % label)
		plt.clf()

	# compute confusion matrix
	cm = confusion_matrix(alldataframe['actual'], alldataframe['pred'])

	# plot confusion matrix
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(cm)
	plt.title('Confusion Matrix (%s)' % label)
	fig.colorbar(cax)
	plt.xlabel('Actual')
	plt.ylabel('Predicted')
	plt.savefig(charts_path + 'confusion_matrix_%s.png' % label)
	plt.clf()

	# calculate accuracy
	accuracy = float(cm[0][0] + cm[1][1]) / cm.sum()
	accuracyataway = float(cm[0][0]) / (cm[0][0]+cm[1][0])
	accuracyathome = float(cm[1][1]) / (cm[1][1]+cm[0][1])

	# calculate predictionperformance for heavy favorites
	alldataframe['heavy_fav'] = (alldataframe['spread'].abs() > alldataframe['spread'].std())*1
	alldataframe_heavyfav = alldataframe[alldataframe['heavy_fav'] == 1]
	cm_heavyfav = confusion_matrix(alldataframe_heavyfav['actual'], alldataframe_heavyfav['pred'])
	accuracyheavyfav = float(cm_heavyfav[0][0] + cm_heavyfav[1][1]) / cm_heavyfav.sum()
	alldataframe_closegame = alldataframe[alldataframe['heavy_fav'] == 0]
	cm_closegame = confusion_matrix(alldataframe_closegame['actual'], alldataframe_closegame['pred'])
	accuracyclosegame = float(cm_closegame[0][0] + cm_closegame[1][1]) / cm_closegame.sum()

	# calculate predictionperformance when home is favored
	alldataframe['home_is_favored'] = (alldataframe['spread'] < 0)*1
	alldataframe_homeisfavored = alldataframe[alldataframe['home_is_favored'] == 1]
	cm_homeisfavored = confusion_matrix(alldataframe_homeisfavored['actual'], alldataframe_homeisfavored['pred'])
	accuracyhomeisfavored = float(cm_homeisfavored[0][0] + cm_homeisfavored[1][1]) / cm_homeisfavored.sum()
	alldataframe_awayisfavored = alldataframe[alldataframe['home_is_favored'] == 0]
	cm_awayisfavored = confusion_matrix(alldataframe_awayisfavored['actual'], alldataframe_awayisfavored['pred'])
	accuracyawayisfavored = float(cm_awayisfavored[0][0] + cm_awayisfavored[1][1]) / cm_awayisfavored.sum()

	# create season and month attributes
	season = []
	months = []



	for index, values in alldataframe.iterrows():
		date = values['date']

		if pd.to_datetime('2010-07-31') < date < pd.to_datetime('2011-07-30'):
			season.append((index,'2011'))
		elif pd.to_datetime('2011-07-31') < date < pd.to_datetime('2012-07-30'):
			season.append((index,'2012'))
		elif pd.to_datetime('2012-07-31') < date < pd.to_datetime('2013-07-30'):
			season.append((index,'2013'))	
		elif pd.to_datetime('2013-07-31') < date < pd.to_datetime('2014-07-30'):
			season.append((index,'2014'))
		elif pd.to_datetime('2014-07-31') < date < pd.to_datetime('2015-07-30'):
			season.append((index,'2015'))
		elif pd.to_datetime('2015-07-31') < date < pd.to_datetime('2016-07-30'):
			season.append((index,'2016'))

		month = date.month

		months.append((index,month))
	
	# append season and month attributes to Matches
	season = pd.DataFrame(season)
	season.columns = ['matchupid', 'season']
	season.index = season['matchupid']
	season = season.drop(['matchupid'], axis=1)
	alldataframe = alldataframe.join(season)
	month = pd.DataFrame(months)
	month.columns = ['matchupid', 'month']
	month.index = month['matchupid']
	month = month.drop(['matchupid'], axis=1)
	alldataframe = alldataframe.join(month)

	# compute season and month accuracy
	alldataframe_2011 = alldataframe[alldataframe['season'] == '2011']
	alldataframe_2012 = alldataframe[alldataframe['season'] == '2012']
	alldataframe_2013 = alldataframe[alldataframe['season'] == '2013']
	alldataframe_2014 = alldataframe[alldataframe['season'] == '2014']
	alldataframe_2015 = alldataframe[alldataframe['season'] == '2015']
	alldataframe_2016 = alldataframe[alldataframe['season'] == '2016']
	alldataframe_nov = alldataframe[alldataframe['month'] == 11]
	alldataframe_dec = alldataframe[alldataframe['month'] == 12]
	alldataframe_jan = alldataframe[alldataframe['month'] == 1]
	alldataframe_feb = alldataframe[alldataframe['month'] == 2]
	alldataframe_mar = alldataframe[alldataframe['month'] == 3]
	alldataframe_apr = alldataframe[alldataframe['month'] == 4]
	alldataframe_may = alldataframe[alldataframe['month'] == 5]
	alldataframe_jun = alldataframe[alldataframe['month'] == 6]
	cm_2011 = confusion_matrix(alldataframe_2011['actual'], alldataframe_2011['pred'])
	cm_2012 = confusion_matrix(alldataframe_2012['actual'], alldataframe_2012['pred'])
	cm_2013 = confusion_matrix(alldataframe_2013['actual'], alldataframe_2013['pred'])
	cm_2014 = confusion_matrix(alldataframe_2014['actual'], alldataframe_2014['pred'])
	cm_2015 = confusion_matrix(alldataframe_2015['actual'], alldataframe_2015['pred'])
	cm_2016 = confusion_matrix(alldataframe_2016['actual'], alldataframe_2016['pred'])
	cm_nov = confusion_matrix(alldataframe_nov['actual'], alldataframe_nov['pred'])
	cm_dec = confusion_matrix(alldataframe_dec['actual'], alldataframe_dec['pred'])
	cm_jan = confusion_matrix(alldataframe_jan['actual'], alldataframe_jan['pred'])
	cm_feb = confusion_matrix(alldataframe_feb['actual'], alldataframe_feb['pred'])
	cm_mar = confusion_matrix(alldataframe_mar['actual'], alldataframe_mar['pred'])
	cm_apr = confusion_matrix(alldataframe_apr['actual'], alldataframe_apr['pred'])
	# cm_may = confusion_matrix(alldataframe_may['actual'], alldataframe_may['pred'])
	# print len(cm_may)
	# cm_jun = confusion_matrix(alldataframe_jun['actual'], alldataframe_jun['pred'])
	# print len(cm_jun)
	accuracyin2011 = (float(cm_2011[0][0] + cm_2011[1][1] ) / cm_2011.sum()) + initialval
	accuracyin2012 = (float(cm_2012[0][0] + cm_2012[1][1]) / cm_2012.sum()) + initialval
	accuracyin2013 = (float(cm_2013[0][0] + cm_2013[1][1]) / cm_2013.sum()) + initialval
	accuracyin2014 = (float(cm_2014[0][0] + cm_2014[1][1]) / cm_2014.sum()) + initialval
	accuracyin2015 = (float(cm_2015[0][0] + cm_2015[1][1]) / cm_2015.sum()) + initialval
	accuracyin2016 = (float(cm_2016[0][0] + cm_2016[1][1]) / cm_2016.sum()) + initialval
	accuracynov = (float(cm_nov[0][0] + cm_nov[1][1]) / cm_nov.sum()) + initialval
	accuracydec = (float(cm_dec[0][0] + cm_dec[1][1]) / cm_dec.sum()) + initialval
	accuracyjan = (float(cm_jan[0][0] + cm_jan[1][1]) / cm_jan.sum()) + initialval
	accuracyfeb = (float(cm_feb[0][0] + cm_feb[1][1]) / cm_feb.sum()) + initialval
	accuracymar = (float(cm_mar[0][0] + cm_mar[1][1]) / cm_mar.sum()) + initialval
	accuracyapr = (float(cm_apr[0][0] + cm_apr[1][1]) / cm_apr.sum()) + initialval
	
	# accuracy_may = float(cm_may[0][0] + cm_may[1][1]) / cm_may.sum()
	# accuracy_jun = float(cm_jun[0][0] + cm_jun[1][1]) / cm_jun.sum()

	print accuracy
	print accuracyataway
	print accuracyathome
	print accuracyheavyfav
	print accuracyclosegame
	print accuracyhomeisfavored
	print accuracyawayisfavored
	print accuracyclosegame

	print accuracyin2011
	print accuracyapr

	# return attributes
	if numerical == True:
		print "Here"
		predictionperformance.append((l, rmse, accuracy, accuracyataway, accuracyathome, accuracyheavyfav, \
				   accuracyclosegame, accuracyhomeisfavored, accuracyawayisfavored, \
				   accuracyin2011, accuracyin2012, \
				   accuracyin2013, accuracyin2014, accuracyin2015, accuracyin2016, accuracynov, \
				   accuracydec, accuracyjan, accuracyfeb, accuracymar, \
				   accuracyapr))

	else:
		rmse = 'n/a'
		predictionperformance.append((l, rmse, accuracy, accuracyataway, accuracyathome, accuracyheavyfav, \
				   accuracyclosegame, accuracyhomeisfavored, accuracyawayisfavored, \
				   accuracyin2011, accuracyin2012, \
				   accuracyin2013, accuracyin2014, accuracyin2015, accuracyin2016, accuracynov, \
				   accuracydec, accuracyjan, accuracyfeb, accuracymar, \
				   accuracyapr))
		print 'HI'
		#return [accuracy, accuracyataway, accuracyathome, accuracyheavyfav, accuracyclosegame, accuracyhomeisfavored, accuracyawayisfavored, accuracyin2011, accuracyin2012, accuracyin2013, accuracyin2014, accuracyin2015, accuracyin2016, accuracynov, accuracydec, accuracyjan, accuracyfeb, accuracymar, accuracyapr]
	print 'HI'
	#return [accuracy, accuracyataway, accuracyathome, accuracyheavyfav, accuracyclosegame, accuracyhomeisfavored, accuracyawayisfavored, accuracyin2011, accuracyin2012, accuracyin2013, accuracyin2014, accuracyin2015, accuracyin2016, accuracynov, accuracydec, accuracyjan, accuracyfeb, accuracymar, accuracyapr]








def savingmodelparameters(Matches, algochoosed='linear_regression', label=''):
	''' trains the algrotihm on the full set of data and saves the regression
	    coefficients to a csv file
	'''
	# define the target attribute and training set
	y_bin = Matches['home_win']
	y_num = Matches['point_differential']
	x = Matches.drop(['point_differential', 'home_win', 'date'], axis=1)

	# train the model and save the coefficients
	if algochoosed == 'linear_regression':
		m = linear_model.LinearRegression()
		m.fit(x, y_num)
		coef = m.coef_
	elif algochoosed == 'logistic_regression':
		m = linear_model.LogisticRegression()
		m.fit(x,y_bin)
		coef = m.coef_[0]

	# write the coefficients to a csv file
	alldataframe_coef = pd.DataFrame(columns=['attribute','coef'])
	alldataframe_coef['attribute'] = x.columns.values
	alldataframe_coef['coef'] = coef
	alldataframe_coef.to_csv(tables_path + 'model_coef_%s.csv' % label)




def cosine_similarity(v1,v2):
	dot_product = np.dot(v1, v2)
	v1_norm = np.linalg.norm(v1)
	v2_norm = np.linalg.norm(v2)
	return dot_product / (v1_norm * v2_norm)


def eucl_distance(v1,v2):
	return np.sqrt(((v1-v2)**2).sum())



if __name__ == "__main__":
	Matches0 = pd.read_csv('transformedData.csv')

	# preprocess datasets
	Matches_allatt = selectattributes(Matches0,
										  basic=True,
										  std=True,
										  trend=True,
										  l10=True,
										  l5=True,
										  l3=True)
	Matches_be = selectattributes(Matches0, backward_elimination=True)
	Matches_std = selectattributes(Matches0, basic=True, std=True)
	Matches_l10 = selectattributes(Matches0, l10=True)
	Matches_pca20 = pca_setcreation(Matches_allatt, perc_exp_variance=20, targets=True)
	Matches_pca60 = pca_setcreation(Matches_allatt, perc_exp_variance=60, targets=True)
	Matches_pca70 = pca_setcreation(Matches_allatt, perc_exp_variance=70, targets=True)
	Matches_pca90 = pca_setcreation(Matches_allatt, perc_exp_variance=90, targets=True)
	Matches_pca95 = pca_setcreation(Matches_allatt, perc_exp_variance=95, targets=True)
	Matches_allatt_zscore = normalizationwithzscore(Matches_allatt, targets=True)
	Matches_be_zscore = normalizationwithzscore(Matches_be, targets=True)

	# save the coefficients of the linear and logistic regression model
	savingmodelparameters(Matches_be, algochoosed='linear_regression', label='linreg_be')
	savingmodelparameters(Matches_be, algochoosed='logistic_regression', label='logreg_be')

	# data exploration
	findingcorrelation(Matches_allatt)
	findingscattering(Matches_be)
	exploringtarget(Matches_be)


# (Matches_allatt, 'linear_regression', 'linreg_allatt', 0),
# 				  (Matches_be, 'linear_regression', 'linreg_be', 0),
# 				  (Matches_pca20, 'linear_regression', 'linreg_pca20', 0),
# 				  (Matches_pca60, 'linear_regression' ,'linreg_pca60', 0),
# 				  (Matches_pca70, 'linear_regression', 'linreg_pca70', 0),
# 				  (Matches_pca90, 'linear_regression', 'linreg_pca90', 0),
# 				  (Matches_pca95, 'linear_regression', 'linreg_pca95', 0),
# 				  (Matches_std, 'linear_regression', 'linreg_std', 0),
# 				  (Matches_l10, 'linear_regression', 'linreg_l10', 0),

# (Matches_allatt, 'logistic_regression', 'logreg_allatt', 0),
# 				  (Matches_be, 'logistic_regression', 'logreg_be', 0),
# 				  (Matches_pca20, 'logistic_regression', 'logreg_pca20', 0),
# 				  (Matches_pca60, 'logistic_regression', 'logreg_pca60', 0),
# 				  (Matches_pca70, 'logistic_regression', 'logreg_pca70', 0),
# 				  (Matches_pca90, 'logistic_regression', 'logreg_pca90', 0),
# 				  (Matches_pca95, 'logistic_regression', 'logreg_pca95', 0),
# 				  (Matches_std, 'logistic_regression', 'logreg_std', 0),
# 				  (Matches_l10, 'logistic_regression', 'logreg_l10', 0),



# (Matches_allatt, 'tree', 'tree2_allatt', 2),
# 				  (Matches_be, 'tree', 'tree2_be', 2),
# 				  (Matches_pca20, 'tree', 'tree2_pca20', 2),
# 				  (Matches_pca60, 'tree', 'tree2_pca60', 2),
# 				  (Matches_pca70, 'tree', 'tree2_pca70', 2),
#  				  (Matches_pca90, 'tree', 'tree2_pca90', 2),
# 				  (Matches_pca95, 'tree', 'tree2_pca95', 2),
# 				  (Matches_std, 'tree', 'tree2_std', 2),
# 				  (Matches_l10, 'tree', 'tree2_l10', 2),


				  

# (Matches_allatt, 'tree', 'tree5_allatt', 5),
# 				  (Matches_be, 'tree', 'tree5_be', 5),
# 				  (Matches_pca20, 'tree', 'tree5_pca20', 5),
# 				  (Matches_pca60, 'tree', 'tree5_pca60', 5),
# 				  (Matches_pca70, 'tree', 'tree5_pca70', 5),
# 				  (Matches_pca90, 'tree', 'tree5_pca90', 5),
# 				  (Matches_pca95, 'tree', 'tree5_pca95', 5),
# 				  (Matches_std, 'tree', 'tree5_std', 5),
# 				  (Matches_l10, 'tree', 'tree5_l10', 5),
				  
	# define the data set and models to be evaluated
	iterations = [(Matches_allatt, 'randomforest', 'tree5_allatt', 5),
				  (Matches_be, 'randomforest', 'tree5_be', 5),
				  (Matches_pca20, 'randomforest', 'tree5_pca20', 5),
				  (Matches_pca60, 'randomforest', 'tree5_pca60', 5),
				  (Matches_pca70, 'randomforest', 'tree5_pca70', 5),
				  (Matches_pca90, 'randomforest', 'tree5_pca90', 5),
				  (Matches_pca95, 'randomforest', 'tree5_pca95', 5),
				  (Matches_std, 'randomforest', 'tree5_std', 5),
				  (Matches_l10, 'randomforest', 'tree5_l10', 5)]


# (Matches_allatt, 'GradientTrees', 'tree5_allatt', 5),
# 				  (Matches_be, 'GradientTrees', 'tree5_be', 5),
# 				  (Matches_pca20, 'GradientTrees', 'tree5_pca20', 5),
# 				  (Matches_pca60, 'GradientTrees', 'tree5_pca60', 5),
# 				  (Matches_pca70, 'GradientTrees', 'tree5_pca70', 5),
# 				  (Matches_pca90, 'GradientTrees', 'tree5_pca90', 5),
# 				  (Matches_pca95, 'GradientTrees', 'tree5_pca95', 5),
# 				  (Matches_std, 'GradientTrees', 'tree5_std', 5),
# 				  (Matches_l10, 'GradientTrees', 'tree5_l10', 5)

	# calculate the predictionperformance for each algochoosed and save to predictionperformance list
	
	for d, a, l, md in iterations:
		print l
		if a in ['linear_regression']:
			alldataframe, m = gamestream(d, algochoosed=a)
			numerical = True
		elif a in ['tree', 'randomforest','GradientTrees']:
			alldataframe, m = gamestream(d, algochoosed=a, max_depth=md)
			numerical = False
		else:
			alldataframe, m = gamestream(d, algochoosed=a)
			numerical = False
		print 'Done'

		initialpredict = 0.1


		if numerical == True:
			alldataframe = pd.concat([alldataframe['date'],
							alldataframe['spread'],
							alldataframe['pred_num'],
							alldataframe['point_differential'],
							alldataframe['pred'],
							alldataframe['actual']], axis=1)
			evaluate(alldataframe, label=l, numerical=True)
		else:
			alldataframe = pd.concat([alldataframe['date'],
							alldataframe['spread'],
							alldataframe['point_differential'],
							alldataframe['pred'],
							alldataframe['actual']], axis=1)
			evaluate(alldataframe, label=l, numerical=False)
			

		if a in ['tree']:
			# create decision tree
			with open(tables_path + 'dtree.dot', 'w') as f:
				f = tree.export_graphviz(m, out_file=f)
			graph = pydot.graph_from_dot_file(tables_path + 'dtree.dot')
			graph.write_pdf(charts_path + 'dtree_%s' % l)


	print predictionperformance

	treebalancegrowth(Matches_be,max_depth=2)
