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
    return accuracy


def choose_K(X_train,Y_train,X_val,Y_val):
    # Initialize K and max K
    K = 1
    max_k = len(X_train)
    # Dictionary of (k, accuracy) pairs
    accuracies = {}

    # Iterate from K=1 to K=(number of samples)
    while K < max_k:
        # Calculate accuracy and add to dictionary
        current_acc = KNN_test(X_train, Y_train, X_val, Y_val, K)
        accuracies[K] = current_acc
        K += 1

    # Get K value with maximum accuracy
    best_acc = max(accuracies.values())
    best_K = [k for k, v in accuracies.items() if v == best_acc][0]

    return best_K
