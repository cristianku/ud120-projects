#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

#########################################################

#
# from sklearn.svm import SVC
# clf = SVC(kernel="linear")
#
#
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
#
# from sklearn.metrics import accuracy_score
# acc = accuracy_score (pred, labels_test)
# print (acc)
# ##############
# from sklearn.svm import SVC
# clf = SVC(kernel="rbf", C=10)
#
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
#
# from sklearn.metrics import accuracy_score
# acc = accuracy_score (pred, labels_test)
# print ( acc)
#
# ##############
# from sklearn.svm import SVC
# clf = SVC(kernel="rbf", C=100)
#
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
#
# from sklearn.metrics import accuracy_score
# acc = accuracy_score (pred, labels_test)
# print ( acc)
# ##############
# from sklearn.svm import SVC
# clf = SVC(kernel="rbf", C=1000)
#
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
#
# from sklearn.metrics import accuracy_score
# acc = accuracy_score (pred, labels_test)
# print ( acc)
#
# ##############
# from sklearn.svm import SVC
# clf = SVC(kernel="rbf", C=1000)
#
# clf.fit(features_train, labels_train)
# pred = clf.predict(features_test)
#
# from sklearn.metrics import accuracy_score
# acc = accuracy_score (pred, labels_test)
# print ( acc)
#
#
# ##############
#
# features_train, features_test, labels_train, labels_test = preprocess()


from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=10000)

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score (pred, labels_test)
print ( acc)

# 0 or 1, corresponding to Sara and Chris respectively

print(pred[10])
print(pred[26])

print(pred[10])




