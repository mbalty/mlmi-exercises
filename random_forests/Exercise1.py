
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cross_validation import train_test_split
import numpy
import csv
import argparse

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def column(matrix, i):
	return [row[i] for row in matrix]

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



def train(X,y,range_interval,parameter = 'Depth'):

	X_train, X_test, y_train, y_test = train_test_split(X, y)



	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, 0.02), numpy.arange(y_min, y_max, 0.02))

	figure1 = plt.figure(figsize=(27, 9))

	cm_bright = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])

	i = 0

	for d in range_interval:

		i = i + 1

		if parameter is 'Depth':
			rf = RandomForestClassifier(max_depth = d, n_estimators=20)
		else:
			rf = RandomForestClassifier(max_depth = 5, n_estimators=d)

		rf.fit(X_train, y_train)

		score = rf.score(X_test, y_test)

		Z = rf.predict( numpy.c_[xx.ravel(), yy.ravel()] )

		# Put the result into a color plot
		Z = Z.reshape(xx.shape)

		ax = plt.subplot(3, 3, i)
		ax.contourf(xx, yy, Z, cmap=cm_bright, alpha=.8)

		# Plot also the training points
		ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
		# and testing points
		ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)

		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		ax.set_xticks(())
		ax.set_yticks(())
		ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
		ax.set_title(parameter + ' ' + str(d))


		





	#return feature_matrix, class_vector

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("-d", "--max_depth", type = int, default = 5)
	parser.add_argument("-t", "--nr_trees", type = int, default = 20)
	args = parser.parse_args()


	max_depth = args.max_depth
	nr_trees = args.nr_trees

	features_spiral, classes_spiral = read_csv('SpiralData.csv')
	features_twist, classes_twist = read_csv('TwistData.csv')

	#X = features_spiral + features_twist
	#y = classes_spiral + classes_twist

	X = features_twist
	y = classes_twist

	#X = features_spiral
	#y = classes_spiral



	X = numpy.array(X)
	y = numpy.array(y)

	# just plot the dataset first

	train(X,y,range(10,55,5),'Tree')
	train(X,y,range(2,9),'Depth')









	plt.tight_layout()
	plt.show()