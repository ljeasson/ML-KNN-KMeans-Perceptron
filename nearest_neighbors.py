import numpy as np
import math

def euclidean_distance(data1, data2, length):
    distance = 0
    for i in range(length):
        distance += pow((data1[i] - data2[i]), 2)
    return math.sqrt(distance)

def KNN_test(X_train, Y_train, X_test, Y_test, K):
    # Get length of training and test data
    test_length = X_test.shape[0]
    train_length = X_train.shape[0]

    # Get number of features
    num_features = X_test.shape[1]

    # Score for calculation of accuracy
    score = 0

    # Iterate through all test data
    for i in range(test_length):

        # Initialize list of distances
        distances = []

        # Iterate throuhg all training data
        for j in range(train_length):
            #print(X_test[0],X_train[j]) #Y_train[j]
            dist = euclidean_distance(X_test[i], X_train[j], num_features)
            distances.append([dist, Y_train[j][0]])

        # Sort distances
        distances.sort()

        # Find K nearest neighbors
        neighbors = []
        for i in range(K):
            neighbors.append(distances[i])
        #print(neighbors)

        # Voting
        positive1 = 0
        negative1 = 0
        determined_label = 0
        for i in range(len(neighbors)):
            if neighbors[i][1] == -1:
                negative1 += 1
            else:
                positive1 += 1
        
        # Determine label
        #print("Number of Positives:",positive1)
        #print("Number of Negatives:",negative1)
        if positive1 > negative1:
            determined_label = 1
        else:
            determined_label = -1
    
        # Update score if determined and test label
        # are equal
        #print("Determined Label:",determined_label)
        #print("Test Label:",Y_test[0][0])
        if determined_label == Y_test[0][0]:
            score += 1

    # Calculate accuracy of test data
    accuracy = score / train_length
    #print("Accuracy:",accuracy)
    return accuracy


def choose_K(X_train,Y_train,X_val,Y_val):
    return 0


X_train = np.array( [[1,5], [2,6], [2,7], [3,7], [3,8], [4,8], [5,1], [5,9], [6,2], [7,2], [7,3], [8,3], [8,4], [9,5]] )
Y_train = np.array( [[-1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [1]] )

X_test = np.array( [[1,1], [2,1], [0,10], [10,10], [5,5], [3,10], [9,4], [6,2], [2,2], [8,7]] )
Y_test = np.array( [[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]] )

acc_K1 = KNN_test(X_train, Y_train, X_test, Y_test, 1)
acc_K2 = KNN_test(X_train, Y_train, X_test, Y_test, 3)
acc_K3 = KNN_test(X_train, Y_train, X_test, Y_test, 5)
print("Accuracy 1:",acc_K1)
print("Accuracy 2:",acc_K2)
print("Accuracy 3:",acc_K3)