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
    num_samples = X.shape[0]
    num_features = X.shape[1]
    print("Num Samples:  ",num_samples)
    print("Num Features: ",num_features)

    weights = np.ones(num_features, dtype=int)
    print("Weights: ",weights)
    bias = 0
    print("Bias: ",bias,"\n")

    for sample in range(num_samples):
        activation = calculate_activation(X[sample], weights, bias)
        print("Activation: ",activation)
        current_label = Y[sample]
        print("Current Label: ",current_label)
        
        if activation * current_label <= 0:
            # Modify with better test data
            for w in range(len(weights)):
                print("Old Weight: ",weights[w])
                weights[w] = update_weight(weights[w], X[sample][w], Y[sample])
                print("New Weight: ",weights[w])

            print("Old Bias: ",bias)    
            bias = update_bias(bias, Y[sample])
            print("New Bias: ",bias) 
        
        print()

    print(weights," ",bias)
    return (weights, bias)

def perceptron_test(X_test, Y_test, w, b):
    return 0


X = np.array( [[0,1], [1,0], [5,4], [1,1], [3,3], [2,4], [1,6]] )
Y = np.array( [[1], [1], [0], [1], [0], [0], [0]] )

#X = np.array( [[-2,1], [1,1], [1.5,-0.5], [-2,-1], [-1,-1.5], [2,2]] )
#Y = np.array( [[1], [1], [1], [-1], [-1], [-1]] )

W = perceptron_train(X,Y)
test_acc = perceptron_test(X,Y,W[0],W[1])