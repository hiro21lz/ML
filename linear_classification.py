
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
# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

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
	plt.close()

	# 2 feature classification
	X, y = mglearn.datasets.make_forge()

	fig, axes = plt.subplots(1, 2, figsize=(10, 3))
	"""
	for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
	    clf = model.fit(X, y)
	    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
	                                    ax=ax, alpha=.7)
	    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
	    ax.set_title("{}".format(clf.__class__.__name__))
	    ax.set_xlabel("Feature 0")
	    ax.set_ylabel("Feature 1")
	axes[0].legend()
	# plt.show()
	"""
	"""
	mglearn.plots.plot_linear_svc_regularization()
	# plt.show()
	"""
	# 
	from sklearn.datasets import load_breast_cancer
	cancer = load_breast_cancer()
	X_train, X_test, y_train, y_test = train_test_split(
	    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
	logreg = LogisticRegression().fit(X_train, y_train)
	print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
	print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

	logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
	print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
	print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

	logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
	print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
	print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))


	plt.plot(logreg.coef_.T, 'o', label="C=1")
	plt.plot(logreg100.coef_.T, '^', label="C=100")
	plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
	plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
	xlims = plt.xlim()
	plt.hlines(0, xlims[0], xlims[1])
	plt.xlim(xlims)
	plt.ylim(-5, 5)
	plt.xlabel("Feature")
	plt.ylabel("Coefficient magnitude")
	plt.legend()
	plt.show()