
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn 
# from tqdm import tqdm
from logging import getLogger
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons

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
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)


if __name__ == '__main__':

	from sklearn.datasets import load_breast_cancer
	cancer = load_breast_cancer()

	from sklearn.ensemble import GradientBoostingClassifier

	X_train, X_test, y_train, y_test = train_test_split(
	    cancer.data, cancer.target, random_state=0)

	gbrt = GradientBoostingClassifier(random_state=0)
	gbrt.fit(X_train, y_train)

	print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
	print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

	gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
	gbrt.fit(X_train, y_train)

	print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
	print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

	gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
	gbrt.fit(X_train, y_train)

	forest = RandomForestClassifier(n_estimators=100, random_state=0)
	forest.fit(X_train, y_train)


	fig, axes = plt.subplots(1, 2, figsize=(15, 10))
	plt.subplot(1,2,1)
	plt.title('randomforest')
	plot_feature_importances_cancer(forest)
	plt.subplot(1,2,2)
	plt.title('gradient boosting')
	plot_feature_importances_cancer(gbrt)
	plt.show()

	# random forest is more robustness