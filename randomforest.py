from sklearn.linear_model import LogisticRegression
from sklearn import tree,ensemble,metrics
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import random

p = 202500
data = np.ndarray(shape = (300,p),dtype=float)
#read in data
with open('processed.csv') as f:
    f.readline()
    for i,line in enumerate(f):
        data[i] = [float(x) for x in line.split(',')[1:]]
#split train/test 7:3
y = np.append(np.zeros(150),np.ones(150))
random.seed(6)
trainidx = random.sample(range(300),int(300*0.7))
testidx = list(set(range(300)) - set(trainidx))
trainX = data[trainidx]
trainY = y[trainidx]
testX = data[testidx]
testY = y[testidx]

#tune random forest tree depth
max_depths = list(range(1,9))
depths_train_accu = []
depths_test_accu = []
depths_forests = []
for depth in max_depths:
    #print(depth)
    rf = ensemble.RandomForestClassifier(max_depth=depth, random_state=6, n_estimators=100 ,max_features = 1. ,n_jobs = 6)
    rf.fit(trainX,trainY)
    depths_forests.append(rf)
    depths_train_accu.append(accuracy_score(trainY,rf.predict(trainX)))
    depths_test_accu.append(accuracy_score(testY,rf.predict(testX)))
    #print(depths_test_accu)
    
#plot the results
from matplotlib.legend_handler import HandlerLine2D
plt.figure(figsize=(12,8))
line1, = plt.plot(max_depths, depths_train_accu, 'b', label="Training accuracy")
line2, = plt.plot(max_depths, depths_test_accu, 'r', label="Testing accuracy")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy')
plt.xlabel('Tree depth')
plt.grid()
plt.savefig('depth_vs_accu2.jpg',dpi = 300)
plt.show()

#tune random forest mtry
mtrys = np.linspace(0,1,6)[1:]
mtrys_train_accu = []
mtrys_test_accu = []
mtrys_forests = []
for m in mtrys:
    #print(m)
    rf = ensemble.RandomForestClassifier(max_depth=4, random_state=6, n_estimators=100 ,max_features = m, n_jobs = 6)
    rf.fit(trainX,trainY)
    mtrys_forests.append(rf)
    mtrys_train_accu.append(accuracy_score(trainY,rf.predict(trainX)))
    mtrys_test_accu.append(accuracy_score(testY,rf.predict(testX)))
    #print(mtrys_test_accu)

#plot the result
plt.figure(figsize=(12,8))
line1, = plt.plot(mtrys, mtrys_train_accu, 'b', label="Training accuracy")
line2, = plt.plot(mtrys, mtrys_test_accu, 'r', label="Testing accuracy")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy')
plt.xlabel('% of features to consider')
plt.grid()
plt.savefig('mtry_vs_accu.jpg',dpi = 300)
plt.show()

#performance of the final model
bestrf = mtry_forests[-1]
testres = bestrf.predict(testX)
print(classification_report(testY,testres))
