
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn 
from tqdm import tqdm
from logging import getLogger

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer

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
    # print(load_train_data().head())
    # print(load_test_data().head())
    # print(load_submit_data().head())


	boston = load_boston()
	print("Data shape: {}".format(boston.data.shape))
	X, y = mglearn.datasets.load_extended_boston()
	print("X.shape: {}".format(X.shape))


	# k-Neighbors Classification
	# mglearn.plots.plot_knn_classification(n_neighbors=1)
	# # plt.show()


	X, y = mglearn.datasets.make_forge()
	# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
	# clf = KNeighborsClassifier(n_neighbors=3)

	# clf.fit(X_train, y_train)
	# print("Test set predictions: {}".format(clf.predict(X_test)))
	# print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

	# # Analyzing KNeighborsClassifier
	# fig, axes = plt.subplots(1, 3, figsize=(10, 3))

	# for n_neighbors, ax in zip([1, 3, 9], axes):
	#     # the fit method returns the object self, so we can instantiate
	#     # and fit in one line:
	#     clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
	#     mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
	#     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
	#     ax.set_title("{} neighbor(s)".format(n_neighbors))
	#     ax.set_xlabel("feature 0")
	#     ax.set_ylabel("feature 1")
	# axes[0].legend(loc=3)
	# plt.show()


	# n_neighbors vs Accuracy 
	cancer = load_breast_cancer()
	X_train, X_test, y_train, y_test = train_test_split(
	    cancer.data, cancer.target, stratify=cancer.target, random_state=66)

	training_accuracy = []
	test_accuracy = []
	# try n_neighbors from 1 to 10
	neighbors_settings = range(1, 11)

	for n_neighbors in neighbors_settings:
	    # build the model
	    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
	    clf.fit(X_train, y_train)
	    # record training set accuracy
	    training_accuracy.append(clf.score(X_train, y_train))
	    # record generalization accuracy
	    test_accuracy.append(clf.score(X_test, y_test))
	    
	plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
	plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("n_neighbors")
	plt.legend()
	plt.show()