import numpy as np

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

def perceptron_train(X,Y):
    # Number of samples and features
    num_samples = X.shape[0]
    num_features = X.shape[1]
    
    # Initialize weights to 1
    # and bias to 0
    weights = np.ones(num_features, dtype=int)
    bias = 0
    
    # Calculate weights and bias
    for sample in range(num_samples):
        # Calculate activation and set current label
        activation = calculate_activation(X[sample], weights, bias)
        current_label = Y[sample]
        
        # If y * a <= 0, update weights and bias
        if activation * current_label <= 0:
            for w in range(len(weights)):
                weights[w] = update_weight(weights[w], X[sample][w], Y[sample])

            bias = update_bias(bias, Y[sample])

    # Output weights and bias
    print(weights," ",bias)
    return (weights, bias)

def perceptron_test(X_test, Y_test, w, b):
    return 0


#X = np.array( [[0,1], [1,0], [5,4], [1,1], [3,3], [2,4], [1,6]] )
#Y = np.array( [[1], [1], [0], [1], [0], [0], [0]] )

X = np.array( [[-2,1], [1,1], [1.5,-0.5], [-2,-1], [-1,-1.5], [2,2]] )
Y = np.array( [[1], [1], [1], [-1], [-1], [-1]] )

W = perceptron_train(X,Y)
test_acc = perceptron_test(X,Y,W[0],W[1])