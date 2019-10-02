import numpy as np

def perceptron_train(X,Y):
    return 0

def perceptron_test(X_test, Y_test, w, b):
    return 0

X = np.array( [[0,1], [1,0], [5,4], [1,1], [3,3], [2,4], [1,6]] )
Y = np.array( [[1], [1], [0], [1], [0], [0], [0]] )
W = perceptron_train(X,Y)
#test_acc = perceptron_test(X,Y,W[0],W[1])