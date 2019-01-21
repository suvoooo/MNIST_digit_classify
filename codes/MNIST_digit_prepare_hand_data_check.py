#!/usr/bin/python 

import math, time 
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns 
from matplotlib.colors import Normalize






start = time.time()





#MNIST_test_small_df = pd.read_csv('mnist_test_small.csv', sep=',', index_col=0) # start with mnist_test_small.csv for concatenation after the first number move to 
#MNIST_test_small_include_hand_new1.csv ; should be fixed soon # SB17012019

#print MNIST_test_small_df.head(3)

MNIST_test_small_df = pd.read_csv('MNIST_test_small_include_hand_new1.csv', sep=',')
print MNIST_test_small_df.tail(3)




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+ finally check with on handwritten data
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#hand_df_read  = pd.read_csv('hand_check.csv', sep=',',index_col=0)
hand_df_read = pd.read_csv('./mldata/sampleImages/hand_check.csv')
#hand_df_read.columns = MNIST_test_small_df.columns
#print hand_df_read.head(2)


a=hand_df_read.drop(columns=['Unnamed: 0']) # first check the shape and then see where is the problem.  

#print type(a)
print a.head(2)


new_hand_df = pd.concat([MNIST_test_small_df, a])
##print new_hand_df.tail(3)
new_hand_df.to_csv('MNIST_test_small_include_hand_new1.csv', index=False)
print new_hand_df.tail(10)

#a.to_csv('MNIST_test_small_include_hand_new1.csv', index=False)

#new_file_df_hand = pd.read_csv('MNIST_test_small_include_hand_new1.csv')
#print type(new_file_df_hand)
#print new_hand_df.tail(10)


print "including hand written data we have 8 extra columns", new_hand_df.shape



endt = time.time()

print "total time taken = %3.3f"%(endt-start)




