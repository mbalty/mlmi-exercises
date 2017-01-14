
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cross_validation import train_test_split
import csv
import numpy
import argparse
import itertools
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix

from sklearn import linear_model, svm

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = numpy.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 2.0

	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')


def read_csv(file_name):

	features = []
	classes = []

	with open(file_name, 'r') as f:
		reader = csv.reader(f, delimiter=';')

		for row in reader:
			try:
				n = [float(x.strip()) for x in row[0].split(',') if x]
			except ValueError as e:
				continue

			#print (n)
			if n:
				a = n[0:(len(n) - 1)]
				c = (n[len(n) - 1])

				features.append(a)
				classes.append(c)





	return features, classes





	#return feature_matrix, class_vector

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--max_depth", type = int, default = 5)
	parser.add_argument("-t", "--nr_trees", type = int, default = 20)
	args = parser.parse_args()


	max_depth = args.max_depth
	nr_trees = args.nr_trees

	X, y = read_csv('TuberculosisData.csv')

	classes = ['1', '2']



	X = numpy.array(X)
	y = numpy.array(y)

	print(len(y))

	kf = KFold(n_splits=5, shuffle=True)
	kf.get_n_splits(X)

	i = 0

	for train_index, test_index in kf.split(X):

		i = i + 1
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		rf = RandomForestClassifier(max_depth = 4, n_estimators=20)

		y_pred_rf = rf.fit(X_train, y_train).predict(X_test)

		cnf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
		figure1 = plt.figure()
		plot_confusion_matrix(cnf_matrix_rf, classes, normalize=False, title=('Random Forest Confusion matrix' + str(i)))
		figure1.savefig(('Random Forest Confusion matrix' + str(i)))

		svm_object =  svm.SVC(kernel= 'rbf', gamma=10)
		y_pred_svm = svm_object.fit(X_train, y_train).predict(X_test)

		cnf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
		figure2 = plt.figure()
		plot_confusion_matrix(cnf_matrix_svm, classes, normalize=False, title=('SVM Confusion matrix' + str(i)))

		figure2.savefig(('SVM Confusion matrix' + str(i)))

		#logreg = linear_model.LogisticRegression(C=1e5)
		logreg = linear_model.LogisticRegression()
		y_pred_log = logreg.fit(X_train, y_train).predict(X_test)

		cnf_matrix_log = confusion_matrix(y_test, y_pred_log)
		figure3 = plt.figure()
		plot_confusion_matrix(cnf_matrix_log, classes, normalize=False, title=('Logistic Regression Confusion matrix' + str(i)))

		figure3.savefig(('Logistic Regression Confusion matrix' + str(i)))








