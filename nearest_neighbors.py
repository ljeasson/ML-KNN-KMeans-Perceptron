import numpy as np

def KNN_test(X_train, Y_train, X_test, Y_test, K):
    return 0

def choose_K(X_train,Y_train,X_val,Y_val):
    return 0

X_train = np.array( [[1,1], [2,1], [0,10], [10,10], [5,5], [3,10], [9,4], [6,2], [2,2], [8,7]] )
Y_train = np.array( [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1] )

acc_K1 = KNN_test(X_train, Y_train, X_train, Y_train, 1)
acc_K2 = KNN_test(X_train, Y_train, X_train, Y_train, 3)
acc_K3 = KNN_test(X_train, Y_train, X_train, Y_train, 5)

'''
test1 = (1,1,1)
test2 = (2,1,-1)
test3 = [0,10,1]
test4 = [10,10,−1]
test5 = [5,5,1]
test6 = [3,10,−1]
test7 = [9,4,1]
test8 = [6,2,−1]
test9 = [2,2,1]
test10 = [8,7,−1]
'''