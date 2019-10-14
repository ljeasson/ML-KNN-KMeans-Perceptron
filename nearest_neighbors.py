import numpy as np
import math

def euclidean_distance(data1, data2, length):
    distance = 0
    for i in range(length):
        distance += pow((data1[i] - data2[i]), 2)
    return math.sqrt(distance)

def KNN_test(X_train, Y_train, X_test, Y_test, K):
    # Initialize list of distances
    distances = []
    # Get length of training data
    length = X_train.shape[0]
    
    # Iterate through points in training data (TODO)
    for i in range(length):
        dist = euclidean_distance(X_train[i], X_test[i], length)
        distances.append(dist)

    # Sort distances
    distances.sort()
    print(distances)

    # Find K nearest neighbors
    neighbors = []
    for i in range(K):
        neighbors.append(distances[i])

    print(neighbors)   

    return 0

def choose_K(X_train,Y_train,X_val,Y_val):
    return 0


X_train = np.array( [[1,1], [2,1], [0,10], [10,10], [5,5], [3,10], [9,4], [6,2], [2,2], [8,7]] )
Y_train = np.array( [[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]] )
acc_K1 = KNN_test(X_train, Y_train, X_train, Y_train, 1)
acc_K2 = KNN_test(X_train, Y_train, X_train, Y_train, 3)
acc_K3 = KNN_test(X_train, Y_train, X_train, Y_train, 5)

X_train = np.array( [[1,5], [2,6], [2,7], [3,7], [3,8], [4,8], [5,1], [5,9], [6,2], [7,2], [7,3], [8,3], [8,4], [9,5]] )
Y_train = np.array( [[-1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [1]] )
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