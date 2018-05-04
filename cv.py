
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn 
from tqdm import tqdm
from logging import getLogger
from sklearn.model_selection import train_test_split

# from sklearn.datasets import load_boston
# from sklearn.datasets import load_breast_cancer

# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

logger = getLogger(__name__)

# def read_csv(path):
#     logger.debug('enter')
#     df = pd.read_csv(path)
#     """
#     for col in tqdm(df.columns.values):
#         if 'cat' in col:
#             logger.info('categorical: {}'.format(col))
#             tmp = pd.get_dummies(df[col], col)
#             for col2 in tmp.columns.values:
#                 df[col2] = tmp[col2].values
#             df.drop(col, axis=1, inplace=True)
#     """
#     logger.debug('exit')
#     return df

# def load_train_data():
#     logger.debug('enter')
#     df = read_csv(TRAIN_DATA)
#     logger.debug('exit')
#     return df


if __name__ == '__main__':

	from sklearn.datasets import load_breast_cancer
	cancer = load_breast_cancer()
	X_train, X_test, y_train, y_test = train_test_split(
	    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
	logreg = LogisticRegression().fit(X_train, y_train)
	print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
	print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

	## change parameter C
	# logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
	# print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
	# print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

	# logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
	# print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
	# print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

	# plt.plot(logreg.coef_.T, 'o', label="C=1")
	# plt.plot(logreg100.coef_.T, '^', label="C=100")
	# plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
	# plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
	# xlims = plt.xlim()
	# plt.hlines(0, xlims[0], xlims[1])
	# plt.xlim(xlims)
	# plt.ylim(-5, 5)
	# plt.xlabel("Feature")
	# plt.ylabel("Coefficient magnitude")
	# plt.legend()
	# plt.show()

	## cv
	scores  = cross_val_score(logreg, cancer.data, cancer.target)
	print("cross_val_score: {}".format(scores))

	scores  = cross_val_score(logreg, cancer.data, cancer.target, cv=5)
	print("cross_val_score: {}".format(scores))
	print("cross_val_score_mean: {:.2f}".format(scores.mean() ))

	print('grid search')
	cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
	all_params = {'C': [10**i for i in range(-1, 2)],
					'fit_intercept': [True, False],
					'penalty': ['l2', 'l1'],
					'penalty': ['l1'],
					'random_state': [0]}

	for params in tqdm(list(ParameterGrid(all_params))):

		#clf = LogisticRegression(**params)
		#clf.fit(X_train, y_train)
		clf = xgb.sklearn.XGBClassifier(**params)
            
		# break	#1/5 end(dont run other cv)
		scores  = cross_val_score(clf, cancer.data, cancer.target, cv=5)
		print("cross_val_score: {}".format(scores))
		print("cross_val_score_mean: {:.2f}".format(scores.mean() ))
