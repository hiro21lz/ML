
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn 
from tqdm import tqdm
from logging import getLogger
from sklearn.model_selection import train_test_split

# from sklearn.datasets import load_boston
# from sklearn.datasets import load_breast_cancer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

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

	# simple data
	print('<linear regression>')
	X, y = mglearn.datasets.make_wave(n_samples=60)
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

	lr = LinearRegression().fit(X_train, y_train)
	print("lr.coef_: {}".format(lr.coef_))
	print("lr.intercept_: {}".format(lr.intercept_))

	print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
	print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))


	# more complicated data
	X, y = mglearn.datasets.load_extended_boston()
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
	lr = LinearRegression().fit(X_train, y_train)

	print("lr.coef_: {}".format(lr.coef_))
	print("lr.intercept_: {}".format(lr.intercept_))

	print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
	print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

	# Ridge regression
	print('\n<Ridge regression>')
	ridge = Ridge().fit(X_train, y_train)
	print('ridge default')
	print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
	print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

	ridge10 = Ridge(alpha=10).fit(X_train, y_train)
	print('\nridge alpha10')
	print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
	print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

	ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
	print('\nridge alpha1')
	print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
	print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))
	
	"""
	plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
	plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
	plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

	plt.plot(lr.coef_, 'o', label="LinearRegression")
	plt.xlabel("Coefficient index")
	plt.ylabel("Coefficient magnitude")
	xlims = plt.xlim()
	plt.hlines(0, xlims[0], xlims[1])
	plt.xlim(xlims)
	plt.ylim(-25, 25)
	plt.legend()
	# plt.show()
	"""

	# datasize vs score
	# mglearn.plots.plot_ridge_n_samples()
	# plt.show()

	# Lasso
	print('\n<Lasso regression>')
	lasso = Lasso().fit(X_train, y_train)
	print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
	print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
	print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

	# we increase the default setting of "max_iter",
	# otherwise the model would warn us that we should increase max_iter.
	lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
	print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
	print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
	print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))

	lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
	print("Training set score: {:.2f}".format(lasso00001.score(X_train, y_train)))
	print("Test set score: {:.2f}".format(lasso00001.score(X_test, y_test)))
	print("Number of features used: {}".format(np.sum(lasso00001.coef_ != 0)))

	plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
	plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
	plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")

	plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
	plt.legend(ncol=2, loc=(0, 1.05))
	plt.ylim(-25, 25)
	plt.xlabel("Coefficient index")
	plt.ylabel("Coefficient magnitude")
	plt.show()

