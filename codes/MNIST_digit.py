#!/usr/bin/python 

import math, time 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns 
from matplotlib.colors import Normalize




# import from sklearn 
#from sklearn import datasets,  metrics
#from sklearn.datasets import fetch_mldata
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
# test set and smaller test set creation 




start = time.time()





# train set and smaller train set creation



#MNIST_train_df = pd.read_csv('mnist_train.csv')
#print MNIST_df.shape
#MNIST_train_small = MNIST_train_df.iloc[0:12000]
#MNIST_train_small.to_csv('mnist_train_small.csv')
MNIST_train_small_df = pd.read_csv('mnist_train_small.csv', sep=',', index_col=0)
print MNIST_train_small_df.head(3)
print MNIST_train_small_df.shape

#MNIST_train_small_label = MNIST_train_small_df[['label']]
# test the label selection bias i.e. count of numbers are distributed well or not
#print MNIST_train_small_label.shape
#print type(MNIST_train_small_label)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ check the counts of training data (smaller set)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++

fig = sns.countplot(MNIST_train_small_df['label'])
plt.xlabel("Digits", fontsize=15)
plt.ylabel('Digit Counts', fontsize=15)
plt.show(fig)# looks kinda okay 

# or we can just print 

print MNIST_train_small_df['label'].value_counts()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ separate train and test data set from the training data set
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# first separate label and pixel columns

X_tr = MNIST_train_small_df.iloc[:,1:] # iloc ensures X_tr will be a dataframe
y_tr = MNIST_train_small_df.iloc[:, 0]


# check

#print X_tr.head(2)
#print y_tr.head(2)

X_train, X_test, y_train, y_test = train_test_split(X_tr,y_tr,test_size=0.2, random_state=30, stratify=y_tr)

#print type(y_test)
#print type(y_test.values)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ use standard scaler 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# as the pixels are ranging from 0 to 255, we standardize using Std Scalar

steps = [('scaler', StandardScaler()), ('SVM', SVC(kernel='poly'))]
parameters = {'SVM__C':[0.001, 0.1, 100, 10e5], 'SVM__gamma':[10,1,0.1,0.01]}
pipeline = Pipeline(steps) # define 


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ use grid search cv
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)
grid.fit(X_train, y_train)

print "score = %3.2f" %(grid.score(X_test, y_test))


print "best parameters from train data: ", grid.best_params_



# so far polynomial does excellent; score (95%) with 1632 seconds compared to rbf kernel score (78%) and ~3800 seconds
# sigmoid kernel does worst, with score 62%, terrible !!!
# with 12000 columns now the score is 96% wth polynomial kernel


# call the predict method of GridSearchCV

y_pred = grid.predict(X_test)


print type(y_pred)
print len(y_pred)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print y_pred[100:105]
print y_test[100:105]

# 100 out of 100 correct
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#+++++++++++++++++++++++++++++++++++++++++++++++++++
#+ Plot the digits and compare with prediction 
#+++++++++++++++++++++++++++++++++++++++++++++++++++

for i in (np.random.randint(0,270,6)):
	two_d = (np.reshape(X_test.values[i], (28, 28)) * 255).astype(np.uint8)
	plt.title('predicted label: {0}'. format(y_pred[i]))
	plt.imshow(two_d, interpolation='nearest', cmap='gray')
	plt.show()
print "!!!!!!!!!!! uhooooo goood job suvoooooooo !!!!!!!"



print "confusion matrix: \n ", confusion_matrix(y_test, y_pred)




# so far everything was checked on training data set and now we want to use the test data set, the complete different file 
# Repeat the process above and just use the same values obtained for c and gamma from grid search

#MNIST_df = pd.read_csv('mnist_test.csv')
#print MNIST_df.head(4)
#print MNIST_df.shape
#MNIST_test_small = MNIST_df.iloc[0:5000]
#print MNIST_test_small.head()
#MNIST_test_small.to_csv('mnist_test_small.csv')
MNIST_test_small_df = pd.read_csv('mnist_test_small.csv', sep=',', index_col=0)
#print MNIST_test_small_df.head(3)
print type(MNIST_test_small_df.columns)
#mnist_column_names_list = MNIST_test_small_df.columns.values.tolist() # convert the columns name into a list 



print MNIST_test_small_df.shape
X_small_test = MNIST_test_small_df.iloc[:,1:]
Y_small_test = MNIST_test_small_df.iloc[:,0]

# convert the test dataframe into a numpy array

#test_small_arr = MNIST_test_small_df.values bad idea as there are 785 columns so we gotta kick the label column out
test_small_arr = MNIST_test_small_df.iloc[:,1:].values



#divide the test data set into test and training sample

X_test_train, X_test_test, y_test_train, y_test_test = train_test_split(X_small_test,Y_small_test,test_size=0.2, random_state=30, stratify=Y_small_test)



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ use standard scaler 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# as the pixels are ranging from 0 to 255, we standardize using Std Scalar

steps1 = [('scaler', StandardScaler()), ('SVM', SVC(kernel='poly'))]
parameters1 = {'SVM__C':[grid.best_params_['SVM__C']], 'SVM__gamma':[grid.best_params_['SVM__gamma']]} # we use the same value of best fit parameters obtained from training samples 
pipeline1 = Pipeline(steps1) # define 



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ use grid search cv
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#
print "performing fit on the test data set"
print "|||||||||||||||||||||||||||||||||||||||"
print "C and gama are determined from train data set"
print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
grid1 = GridSearchCV(pipeline1, param_grid=parameters1, cv=5)
grid1.fit(X_test_train, y_test_train)

print "score on the test data set= %3.2f" %(grid1.score(X_test_test, y_test_test))
print "best parameters from train data: ", grid1.best_params_

print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

y_test_pred = grid1.predict(X_test_test)

#+++++++++++++++++++++++++++++++++++++++++++++++++++
#+ Plot the digits and compare with prediction 
#+++++++++++++++++++++++++++++++++++++++++++++++++++

for im in (np.random.randint(0,270,6)):
	two_d = (np.reshape(X_test_test.values[im], (28, 28)) * 255).astype(np.uint8)
	plt.title('predicted label: {0}'. format(y_test_pred[im]))
	plt.imshow(two_d, interpolation='nearest', cmap='gray')
	plt.show()
print "!!!!!!!!!!! uhooooo goood job suvoooooooo!  2nd time !!!!!!!"



print "confusion matrix: \n ", confusion_matrix(y_test_test, y_test_pred)





#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ finally check with on handwritten data
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#a.to_csv('MNIST_test_small_include_hand_new1.csv', index=False)

new_file_df_hand = pd.read_csv('MNIST_test_small_include_hand_new1.csv')
#print type(new_file_df_hand)
print new_file_df_hand.tail(10)


print "including hand written data we have 8 extra columns", new_file_df_hand.shape

X_hand = new_file_df_hand.iloc[:,1:]
Y_hand = new_file_df_hand.iloc[:,0]


hand_arr = new_file_df_hand.iloc[:,1:].values




#divide the test data set into test and training sample

#X_hand_train, X_hand_test, y_hand_train, y_hand_test = train_test_split(X_hand,Y_hand,test_size=0.2, random_state=30, stratify=Y_hand)



X_hand_train = new_file_df_hand.iloc[0:3500, 1:]
X_hand_test  = new_file_df_hand.iloc[3500:5011, 1:]
y_hand_test = new_file_df_hand.iloc[3500:5011, 0]
y_hand_train = new_file_df_hand.iloc[0:3500, 0]



#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ use standard scaler 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# as the pixels are ranging from 0 to 255, we standardize using Std Scalar

steps2 = [('scaler', StandardScaler()), ('SVM', SVC(kernel='poly'))]
parameters2 = {'SVM__C':[grid.best_params_['SVM__C']], 'SVM__gamma':[grid.best_params_['SVM__gamma']]} # we use the same value of best fit parameters obtained from training samples 
pipeline2 = Pipeline(steps2) # define pipeline object






#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ use grid search cv
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#
print "performing fit on the test data set"
print "|||||||||||||||||||||||||||||||||||||||"
print "C and gama are determined from train data set"
print "++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
grid2 = GridSearchCV(pipeline2, param_grid=parameters2, cv=5)
grid2.fit(X_hand_train, y_hand_train)

print "score on the test data set include hand-written data = %3.2f" %(grid2.score(X_hand_test, y_hand_test))
print "best parameters from train data: should be same as before: ", grid2.best_params_

print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

y_hand_pred = grid2.predict(X_hand_test)

print type(y_hand_pred), len(y_hand_pred)
print "predicted list: ", y_hand_pred[1496:1511]
print "real handwritten list", y_hand_test[1496:1511]

#+++++++++++++++++++++++++++++++++++++++++++++++++++
#+ Plot the digits and compare with prediction 
#+++++++++++++++++++++++++++++++++++++++++++++++++++

#for ik in (np.random.randint(0,280,4)):
#	three_d = (np.reshape(X_hand_test.values[ik], (28, 28)) * 255).astype(np.uint8)
#	plt.title('predicted label: {0}'. format(y_hand_pred[ik]))
#	plt.imshow(three_d, interpolation='nearest', cmap='gray')
#	plt.show()

for ik in range(1496, 1511, 1):
	three_d = (np.reshape(X_hand_test.values[ik], (28, 28)) * 255).astype(np.uint8)
	plt.title('predicted label: {0}'. format(y_hand_pred[ik]))
	plt.imshow(three_d, interpolation='nearest', cmap='gray')
	plt.show()


print "!!!!!!!!!!! uhooooo goood job suvoooooooo!  3rd time !!!!!!!"


#print "+++++++++++++++++++++++++++++++++++++++++++++++"
#print "now just check the final numbers i.e. handwritten data"
#print "+++++++++++++++++++++++++++++++++++++++++++++++"




#for ig in range(4956, 5003,1):
#	three_d = (np.reshape(X_hand_test.values[ig], (28, 28)) * 255).astype(np.uint8)
#	plt.title('predicted label: {0}'. format(y_hand_pred[ig]))
#	plt.imshow(three_d, interpolation='nearest', cmap='gray')
#	plt.show()
#print "!!!!!!!!!!! uhooooo goood job suvoooooooo!  3rd time !!!!!!!"




endt = time.time()

print "total time taken = %3.3f"%(endt-start)

