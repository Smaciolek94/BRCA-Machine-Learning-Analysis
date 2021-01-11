import numpy as np
import pandas as pd
import sklearn.model_selection
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import VotingClassifier
#have to import every function
import statistics as stat

x = pd.read_csv("C:\\Users\\smaci\\Desktop\\Python tests\\brcax.csv")
y = pd.read_csv("C:\\Users\\smaci\\Desktop\\Python tests\\brcay.csv")
y = y['x']

names = x.columns.tolist()
for i in range(1,31):
    plt.hist(x.iloc[i],bins=20)
    plt.title(names[i])
    plt.show()

datatot = pd.DataFrame.join(x,y)
pltB = x[datatot['x']=="B"]
pltM = x[datatot['x']=="M"]
for i in range(1,31):
    plt.hist(pltM.iloc[i],bins=20,color="red")
    plt.hist(pltB.iloc[i],bins=20,color="blue")
    plt.title(names[i])
    plt.show()
    
cor = x.corr()
cor.to_csv('C:\\Users\\smaci\\Desktop\\Python tests\\corr.csv')
    
xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(x, y, test_size=0.20)

glmmodel = LogisticRegression().fit(xtrain,ytrain)
#there must be at least empty parenthesse here
glmpred = glmmodel.predict(xtest)
glmacc = stat.mean(glmpred == ytest)

ldamodel = LinearDiscriminantAnalysis().fit(xtrain,ytrain)
ldapred = ldamodel.predict(xtest)
ldaacc = stat.mean(ldapred == ytest)

qdamodel = QuadraticDiscriminantAnalysis().fit(xtrain,ytrain)
qdapred = ldamodel.predict(xtest)
qdaacc = stat.mean(qdapred == ytest)

#knnmodel = NearestNeighbors(n_neighbors=5).fit(xtrain,ytrain)
knnmodel = KNeighborsClassifier(n_neighbors=5).fit(xtrain,ytrain)
knnpred = knnmodel.predict(xtest)
knnacc = stat.mean(knnpred == ytest)

nbmodel = GaussianNB().fit(xtrain,ytrain)
nbpred = nbmodel.predict(xtest)
nbacc = stat.mean(nbpred == ytest)

rfmodel = RandomForestClassifier().fit(xtrain,ytrain)
rfpred = rfmodel.predict(xtest)
rfacc = stat.mean(rfpred == ytest)

svmmodel = svm.SVC().fit(xtrain,ytrain)
svmpred = svmmodel.predict(xtest)
svmacc = stat.mean(svmpred == ytest)

ensemble = VotingClassifier(estimators=[('glm',LogisticRegression()),
    ('lda',LinearDiscriminantAnalysis()),('qda',QuadraticDiscriminantAnalysis()),
    ('knn',KNeighborsClassifier()),('nb',GaussianNB()),('rf',RandomForestClassifier()),
    ('svm',svm.SVC())])
ensmodel = ensemble.fit(xtrain,ytrain)
enspred = ensemble.predict(xtest)
ensacc = stat.mean(enspred == ytest)

acclabs = ["GLM","LDA","QDA","KNN","NB","RF","SVM","ENS"]
acc = [glmacc, ldaacc, qdaacc, knnacc, nbacc, rfacc, svmacc, ensacc]
out = pd.DataFrame(acclabs,acc)
print(out)