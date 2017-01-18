
from sklearn.cross_validation import train_test_split
import numpy
import csv
import argparse
import scipy as sp

from sklearn import linear_model, datasets
from sklearn.metrics import log_loss


def column(matrix, i):
	return [row[i] for row in matrix]

def read_csv(file_name):

	features = []
	classes = []
	names = []

	with open(file_name, 'r') as f:
		reader = csv.reader(f, delimiter=';')


		i = True

		for row in reader:

			n = []

			if i:
				i = False
				names = row[0].split(',')

			for x in row[0].split(','):
				if x:
					if x == '"Absent"':
						n.append(0.0)
					elif x == '"Present"':
						n.append(1.0)
					else:
						try:
							n.append(float(x))
						except ValueError as e:
							continue
			
			if n:
				a = n[0:(len(n) - 1)]
				c = (n[len(n) - 1])

				features.append(a)
				classes.append(c)


	names.pop()
	return features, classes, names

def likehood_ratio_test(log_likelihood_full, log_likelihood_reduced, df):
	D = -2.0 * (log_likelihood_reduced - log_likelihood_full)
	p = sp.special.gammainc(df/2, D/2)
	return p

"""
def likehood(X, classifier):
	result = classifier.predict_log_proba(X)
	return result.mean()
"""

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--max_depth", type = int, default = 5)
	parser.add_argument("-t", "--nr_trees", type = int, default = 20)
	args = parser.parse_args()


	max_depth = args.max_depth
	nr_trees = args.nr_trees

	X, y, features = read_csv('SAheart.csv')

	print(features)

	X = numpy.array(X)
	y = numpy.array(y)

	logreg_null = linear_model.LogisticRegression(C=1e6)
	logreg_null.fit(numpy.zeros(y.size).reshape(-1,1), y)

	logreg = []

	for i in range(X.shape[1]):
		logreg_sf = linear_model.LogisticRegression(C=1e6)
		logreg.append(logreg_sf.fit(X[:,i].reshape(-1,1), y))


	logreg_full = linear_model.LogisticRegression(C=1e6)
	logreg_full = logreg_full.fit(X, y)
	

	

	LLF_null = log_loss(y, logreg_null.predict_proba(numpy.zeros(y.size).reshape(-1,1)) )
	LLF_full = log_loss(y, logreg_full.predict_proba(X))
	

	print('p-value Null Model:' + str(likehood_ratio_test(-LLF_full,-LLF_null,X.shape[1])))


	list_p = []
	index_features =[]



	for i in range(len(logreg)):
		LLF_1 = log_loss(y, logreg[i].predict_proba(X[:,i].reshape(-1,1)))
		p = likehood_ratio_test(-LLF_full,-LLF_1,X.shape[1] - X[:,i].reshape(-1,1).shape[1])
		list_p.append(p)
		print('p-value for feature:' + features[i] + ': ' +str(p))

	min_p_index = numpy.argmin(list_p)

	print('Best feature is ' + features[min_p_index] + ' p-value: ' + str(min(list_p)))


	index_features = [min_p_index]


	while True:
		index_combinations = []
		list_p = []

		for i in range(len(features)):
			current_index = list(index_features)

			if i not in index_features:
				current_index.append(i)
				index_combinations.append(current_index)
				cls_reg = linear_model.LogisticRegression(C=1e6)
				cls_reg.fit(X[:,current_index].reshape(-1,len(current_index)),y)
				LLF = log_loss(y, cls_reg.predict_proba(X[:,current_index].reshape(-1,len(current_index))))
				p = likehood_ratio_test(-LLF_full,-LLF,X.shape[1] - len(current_index))
				list_p.append(p)

		if not list_p:
			break

		min_p = min(list_p)

		if (min_p > 0.05):
			break

		min_p_index = numpy.argmin(list_p)

		if (len(index_features) == len(features)):
			break

		
		index_features = index_combinations[min_p_index]

		print(index_features)

	print([features[i] for i in index_features])
	