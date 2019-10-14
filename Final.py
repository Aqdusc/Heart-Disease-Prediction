# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:59:22 2019

@author: aqdus
"""
#import sys



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataset = pd.read_csv("rep_data.csv")

#Importing the test result
testset=pd.read_csv('Predict.csv')
testset=testset.convert_objects(convert_numeric=True)
x_test1=testset.iloc[:,0:13].values
y_test1=testset.iloc[:,13].values

'''
#Importing command line argument
pred=np.array([[int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7]),int(sys.argv[8]),int(sys.argv[9]),float(sys.argv[10]),int(sys.argv[11]),int(sys.argv[12]),int(sys.argv[13])]])
x_test1=np.vstack([x_test1,pred])
'''

#converting all string values to nan
dataset = dataset.convert_objects(convert_numeric=True)
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy ="mean", axis = 0)
imputer = imputer.fit(x[:,0:13])   
x[:, 0:13] = imputer.transform(x[:, 0:13])
imputer=imputer.fit(x_test1[:,0:13])
x_test1[:,0:13]=imputer.transform(x_test1[:,0:13])   

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x = sc_X.fit_transform(x)
x_test1=sc_X.fit_transform(x_test1)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size=0.16,random_state=20)

mean=[0,0,0,0,0,0,0,0,0,0,0,0,0]
for j in range(13):
    for i in range(len(x_train)):
        if(y_train[i]==1):
            mean[j]=mean[j]+x_train[i][j]
for i in range(13):
    mean[i]=mean[i]/len(x_train)
print(mean)    
    
# Applying PCA to testdata.
from sklearn.decomposition import PCA
pca = PCA(n_components =2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_
x_test1=pca.transform(x_test1)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state =20)
classifier.fit(x_train, y_train)

'''
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state =0)
classifier.fit(x_train, y_train)
'''

#Predicting the Test_Set
y_predict = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predict)

#Predicting the Test_data from Predict.csv
y_predict1 = classifier.predict(x_test1)
from sklearn.metrics import confusion_matrix
#f=open('Result.txt' , 'w')
if(int(y_predict1[-1])==1):
    print('You Have A Heart Disease')
    #f.write('You Have A Heart Disease')
elif(int(y_predict1[-1])==0):
    print('You Do Not Have A Heart Disease')
    #f.write('You Do Not Have A Heart Disease')
#f.close()


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'black'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()



# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'black'))(i), label = j)
plt.title('Logistic (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
