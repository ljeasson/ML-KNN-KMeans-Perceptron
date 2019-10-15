import numpy as np
import matplotlib.pyplot as plt

def calculate_activation(features, weights, bias):
    a = 0
    for i in range(len(features)):
        a += features[i] * weights[i]
    a += bias
    return a

def update_weight(weight, x, y):
    return weight + (x * y)

def update_bias(bias, y):
    return bias + y

def epoch(X,Y,num_samples, weights, bias, redo_epoch):
    return

def perceptron_train(X,Y):
    # Number of samples and features
    num_samples = X.shape[0]
    num_features = X.shape[1]
    
    # Initialize weights to 1
    # and bias to 0
    weights = np.zeros(num_features, dtype=int)
    bias = 0
    
    # Redo another epoch indicator
    redo_epoch = False

    # Calculate weights and bias
    #epoch(X,Y,num_samples, weights, bias, redo_epoch)

    for sample in range(num_samples):
        # Calculate activation and set current label
        activation = calculate_activation(X[sample], weights, bias)
        current_label = Y[sample][0]
        
        # If y * a <= 0, update weights and bias
        if activation * current_label <= 0:
            # Update weights
            for w in range(len(weights)):
                weights[w] = update_weight(weights[w], X[sample][w], Y[sample])
            
            # Update bias
            bias = update_bias(bias, Y[sample])
            
            # Redo another epoch
            redo_epoch = True
        
    while (redo_epoch):
        # Disable redo indicator
        redo_epoch = False

        # Calculate weights and bias
        for sample in range(num_samples):
            # Calculate activation and set current label
            activation = calculate_activation(X[sample], weights, bias)
            current_label = Y[sample][0]
            
            # If y * a <= 0, update weights and bias
            if activation * current_label <= 0:
                # Update weights
                for w in range(len(weights)):
                    weights[w] = update_weight(weights[w], X[sample][w], Y[sample])

                # Update bias
                bias = update_bias(bias, Y[sample])
                
                # Redo another epoch
                redo_epoch = True
            
    # Output weights and bias
    return (weights, bias)


def perceptron_test(X_test, Y_test, w, b):
    score = 0
    num_features_and_weights = X_test.shape[1]
    num_samples = Y_test.shape[0]
    redo_epoch = False
    max_iter = 2

    '''
    for i in range(num_samples):
        activation = 0
        
        for j in range(num_features_and_weights):
            activation += w[j]*X_test[i][j]
        activation += b[0]

        if activation == Y_test[i][0]:
            score += 1
    '''
    for i in range(num_samples):
        activation = 0
        
        for j in range(num_features_and_weights):
            activation += w[j]*X_test[i][j]
        activation += b[0]

        if activation == Y_test[i][0]:
            score += 1
        else:
            for j in range(num_features_and_weights):
                w[j] = update_weight(w[j], X_test[i][j], Y_test[i][0])
            b[0] = update_bias(b[0], Y_test[i][0])
            redo_epoch = True
    '''
    while (redo_epoch and max_iter > 0):
        for i in range(num_samples):
            activation = 0
        
            for j in range(num_features_and_weights):
                activation += w[j]*X_test[i][j]
            activation += b[0]

            if activation == Y_test[i][0]:
                score += 1
            else:
                for j in range(num_features_and_weights):
                    w[j] = update_weight(w[j], X_test[i][j], Y_test[i][0])
                b[0] = update_bias(b[0], Y_test[i][0])
                redo_epoch = True
        max_iter -= 1
    '''

    accuracy = score/num_samples
    return accuracy