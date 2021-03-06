import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

# fit = train ( learns the pattern )

# X = features
# Y = labels

clf.fit(X, Y)

GaussianNB(priors=None)

print(clf.predict([[10.5, 18]]))

