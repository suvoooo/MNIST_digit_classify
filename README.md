# MNIST_digit_classify
## MNIST_digit_classify with SVM polynomial kernel
This small project deals with MNIST hand-written digit classification problem. The train and test data are obtained   
from [Kaggle](https://www.kaggle.com/oddrationale/mnist-in-csv) and it contains 784 pixel values in columns and label 
for each digit. 
Using only 12000 training samples (out of 60000), 96% score is obtained and on the test data the score is 93%. Best parameters for SVM are determined using GridSearchCV on the training data-set and the obtained parameters (C = 0.001, $\gamma$ = 10). These parameters are used for test data. Polynomial kernel takes the least time and gives the best result following my codes.   
Finally using [Mypaint](http://mypaint.org/), I have created images from 0 to 9. Using ImageMagick converter, 
the images were resized to 28X28 pixels and they were analyzed using the algorithm optimized for MNIST data. 
70% of the samples were correctly classified i.e. 7 numbers were correctly identified among 10 (0-9) numbers. 
A detailed explanation is given in [Towards Data Science](https://towardsdatascience.com/support-vector-machine-mnist-digit-classification-with-python-including-my-hand-written-digits-83d6eca7004a). 
