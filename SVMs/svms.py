from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve

#svm fit
# kernel= ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
def svm_fit(X, Y, C=1,kernel='linear'):
    svmClassifier = svm.SVC(C=C,kernel=kernel)
    svmClassifier.fit(X, Y)
    return svmClassifier


# svm predict
class ClasifEval:
    def __init__(self, pred=None, acc=0, recall=0, precision=0, fpr=0, sup=None):
        self.pred = pred
        self.accuracy = acc
        self.recall = recall
        self.precision = precision
        self.fpr = fpr
        self.support_vectors = sup

    def __repr__(self):
        return ''.join([
            'accuracy: ', str(self.accuracy), '\n',
            'precision: ', str(self.precision), '\n',
            'recall: ', str(self.recall), '\n',
            'FPR: ', str(self.fpr), '\n'
        ]
        )


def svm_predict(svmClassifier, Xtest, Ytest):
    Ypred = svmClassifier.predict(Xtest)
    c_eval = ClasifEval()
    c_eval.pred = Ypred
    c_eval.accuracy = accuracy_score(Ytest, Ypred)
    c_eval.precision = precision_score(Ytest, Ypred)
    c_eval.recall = recall_score(Ytest, Ypred)
    c_eval.fpr, _, _ = roc_curve(Ytest - 1, Ypred - 1)
    c_eval.support_vectors = svmClassifier.support_vectors_
    return c_eval