from sklearn.linear_model import RandomizedLasso
import argparse
import numpy as np 
import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn import linear_model
import re
import collections
from csv import DictReader, DictWriter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import kneighbors_graph
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn import mixture 
from sklearn import cluster 
import itertools
import pandas as pd
import sklearn
import csv
from sklearn.base import TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split

start_time = time.time()




class DataFrameImputer(TransformerMixin):

	def __init__(self):
		"""Impute missing values.

		Columns of dtype object are imputed with the most frequent value 
		in column.

		Columns of other types are imputed with mean of column.

		"""
	def fit(self, X, y=None):

		self.fill = pd.Series([X[c].mean()
			if X[c].dtype == np.dtype('float64') or X[c].dtype == np.dtype('int64') else X[c].value_counts().index[0] for c in X],
			index=X.columns)

		return self


	def transform(self, X, y=None):
		return X.fillna(self.fill)

	def fit_transform(self,X,y=None):
		return self.fit(X,y).transform(X)

def End ():
	print "Question_1 Finish"



if __name__ == "__main__":
	print('The scikit-learn version is {}.'.format(sklearn.__version__))
	parser = argparse.ArgumentParser(description='Question_1')

	args = parser.parse_args()

	#train = list(csv.reader(open("active.csv", 'r'), delimiter='\t'))

	#train  = pd.read_csv('active.csv', delimiter='\t', encoding="utf-8-sig")

	train  = pd.read_csv('active_new_pos.csv')


	#intialtrain_df  = pd.DataFrame(train)

	#train['Avg_PTS'] = train.PTS/train.G 

	#print train.columns

	#print len(train.columns)

	#print train.values[0]

	traintomodify  = train.copy(deep = True)

	features = list(train.columns.values)


	#print traintomodify.columns
	#print len(traintomodify.columns)

	nonvocab = ['pid', 'pname', 'POS']

	indicestocontain = []

	for each in traintomodify.columns.values:
		if each not in nonvocab:
			indicestocontain.append(int(features.index(each)))

	#print traintomodify

	#print indicestocontain


	traintomodified = traintomodify[traintomodify.columns[indicestocontain]]

	clusterformodify  = traintomodified.copy(deep = True)

	#print clusterformodify.columns
	#print clusterformodify.dtypes


	clusterformodify = clusterformodify.div(clusterformodify.G, axis = 'index')

	#print clusterformodify.columns
	#print len(clusterformodify.columns.values)

	FinalClusterdata = clusterformodify.copy(deep = True)
	#print FinalClusterdata.columns

	del FinalClusterdata['G']

	
	#print FinalClusterdata.columns
	#print len(FinalClusterdata.columns.values) 


	xdata = traintomodified[traintomodified.columns[0:len(traintomodified.columns.values) - 1]]

	ydata = traintomodified[traintomodified.columns[len(traintomodified.columns.values) -1 : len(traintomodified.columns.values)]]

	#print xdata.columns
	#print len(xdata.columns)

	#print ydata.columns
	#print len(ydata.columns)

	#print ydata['PTS']


	count = 1


	for each in xdata.columns:
		if count == 1:
			TrainData = xdata[each]
			count = 2
		else:
			TrainData = np.column_stack((TrainData,xdata[each]))


	TrainData = np.array(TrainData)

	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	FillTrainData = imp.fit_transform(TrainData)

	FilledTrainData = pd.DataFrame(FillTrainData)
	yvalues = pd.Series(np.array(ydata['PTS']))

	#print FillTrainData



	X_dummytrain, X_dummytest, y_dummytrain, y_dummytest = train_test_split(FilledTrainData,yvalues, test_size=0.2, random_state=42)

	testlistofindices =  X_dummytest.index.tolist()
	trainlistofindices =  X_dummytrain.index.tolist()
	#print y_dummytrain

	#print testlistofindices
	#print len(trainlistofindices)



	clf = linear_model.LinearRegression(fit_intercept = True)
	#clf = AdaBoostRegressor(ensemble.GradientBoostingRegressor(n_estimators = 500, learning_rate = 0.1, max_depth = 6 , random_state = 0, loss = 'ls'),n_estimators = 100, learning_rate = 0.1)
	#clf = ensemble.GradientBoostingRegressor(n_estimators = 500, learning_rate = 0.1, max_depth = 6 , random_state = 0, loss = 'ls')
	estimate = clf.fit(X_dummytrain,y_dummytrain)

	
	

	predictions = estimate.predict(X_dummytest)
	print "In writePredictions"
	#print X_dummytest.iloc[[0]]
	#print y_dummytest.iloc[[0]]
	#print X_testarr[0]
	print estimate.coef_
	print np.dot(X_dummytest.iloc[[0]],estimate.coef_)
	
	o = DictWriter(open("NewpredictionsQuestionAda1.csv", 'w'),["newtarget", "target"])
	o.writeheader()

	
	for ii, pp in zip(y_dummytest, predictions):
		d = {'newtarget': ii, 'target': pp}
		o.writerow(d)


	indice = xdata.iloc[testlistofindices]['G']


	results  = pd.DataFrame(index=range(0,len(testlistofindices)),columns=['predicts'], dtype='float')

	results['predicts'] = predictions
	results['games'] = xdata.iloc[testlistofindices]['G'].reset_index(drop=True)
	results['actuals'] = y_dummytest.reset_index(drop=True)

	results['Avg_PTS'] = results.actuals/results.games
	results['Predicts_PTS'] = results.predicts/results.games

	print "Root Mean Square Error Below"

	print mean_squared_error(results['Avg_PTS'], results['Predicts_PTS'])




	count = 1

	clusterfeatures = list(FinalClusterdata.columns.values)

	#print clusterfeatures

	clustersfunc = [
	cluster.MeanShift(bin_seeding=True),
	cluster.Birch(branching_factor=50, compute_labels=True, copy=True, n_clusters=None,threshold=0.5)]

	n_samples = 1500
	np.random.seed(0)
	t = 1.5 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
	x = t * np.cos(t)
	y = t * np.sin(t)
	X = np.concatenate((x, y))
	X += .7 * np.random.randn(2, n_samples)
	X = X.T
	knn_graph = kneighbors_graph(X, 30, include_self=False)


	for x in xrange(2,6):
		print x
		for subset in itertools.combinations(clusterfeatures,x):
			count = 1
			MiniClusterData = []
			for each in subset:
				if count == 1:
					MiniClusterData = FinalClusterdata[each]
					count = 2
				else:
					MiniClusterData = np.column_stack((MiniClusterData,FinalClusterdata[each]))
			
			MiniClusterData = np.array(MiniClusterData)

			imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

			FillMiniClusterData = imp.fit_transform(MiniClusterData)

			FilledMiniClusterData = pd.DataFrame(FillMiniClusterData)


			for i in xrange(5,11):

				
				predictionsare = cluster.AgglomerativeClustering(n_clusters=i, linkage='ward',connectivity= None).fit_predict(FilledMiniClusterData)
				if (sum(i > 115 for i in collections.Counter(predictionsare).values()) >= 5 ):
					print subset
					print "AgglomerativeClustering"
					print i
					print collections.Counter(predictionsare)
					print "___________"

				predictionsare =cluster.SpectralClustering(n_clusters=i,eigen_solver='arpack', affinity="nearest_neighbors").fit_predict(FilledMiniClusterData)

				if (sum(i > 115 for i in collections.Counter(predictionsare).values()) >= 5 ):
					print subset
					print i
					print "SpectralClustering"
					print collections.Counter(predictionsare)
					print "___________"



				predictionsare = mixture.GMM(n_components=i, covariance_type='full').fit_predict(FilledMiniClusterData)

				if (sum(i > 115 for i in collections.Counter(predictionsare).values()) >= 5 ):
					print subset
					print i
					print "Gaussian"
					print collections.Counter(predictionsare)
					print "___________"

		




	print("--- %s seconds ---" % (time.time() - start_time))
	



