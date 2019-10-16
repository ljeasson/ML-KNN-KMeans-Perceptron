import numpy as np
import matplotlib.pyplot as plt
import perceptron as p
import nearest_neighbors as knn

# KNN TESTING
X_train = np.array( [[1,5], [2,6], [2,7], [3,7], [3,8], [4,8], [5,1], [5,9], [6,2], [7,2], [7,3], [8,3], [8,4], [9,5]] )
Y_train = np.array( [[-1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [1]] )

X_test = np.array( [[1,1], [2,1], [0,10], [10,10], [5,5], [3,10], [9,4], [6,2], [2,2], [8,7]] )
Y_test = np.array( [[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]] )

acc_K1 = knn.KNN_test(X_train, Y_train, X_test, Y_test, 1)
acc_K2 = knn.KNN_test(X_train, Y_train, X_test, Y_test, 3)
acc_K3 = knn.KNN_test(X_train, Y_train, X_test, Y_test, 5)
print("Accuracy 1:",acc_K1)
print("Accuracy 2:",acc_K2)
print("Accuracy 3:",acc_K3)
print()

best_K = knn.choose_K(X_train, Y_train, X_test, Y_test)
print("Best K:",best_K)
print()

'''
# PERCEPTRON TESTING

# Hand-Tested Data
X = np.array( [[1,1], [1,-1], [-1,1], [-1,-1]] )
Y = np.array( [[1], [-1], [-1], [-1]] )
W = p.perceptron_train(X,Y)
print("Hand-Tested Data    W1: ",W[0][0],"  W2: ",W[0][1],"  b:",W[1][0])
test_acc = p.perceptron_test(X,Y,W[0],W[1])
print("Accurancy:",test_acc,"\n")

# Percepton Test Data 
X = np.array( [[0,1], [1,0], [5,4], [1,1], [3,3], [2,4], [1,6]] )
Y = np.array( [[1], [1], [-1], [1], [-1], [-1], [-1]] )
W = p.perceptron_train(X,Y)
print("Preceptron Test Data 1    W1: ",W[0][0],"  W2: ",W[0][1],"  b:",W[1][0])
test_acc = p.perceptron_test(X,Y,W[0],W[1])
print("Accurancy:",test_acc,"\n")

# Perceptron Test Data - Writeup
X = np.array( [[-2,1], [1,1], [1.5,-0.5], [-2,-1], [-1,-1.5], [2,-2]] )
Y = np.array( [[1], [1], [1], [-1], [-1], [-1]] )
W = p.perceptron_train(X,Y)
print("Preceptron Test Data 2    W1: ",W[0][0],"  W2: ",W[0][1],"  b:",W[1][0])
test_acc = p.perceptron_test(X,Y,W[0],W[1])
print("Accurancy:",test_acc,"\n")


# Graph weight vector
w1 = W[0][0]
w2 = W[0][1]
b  = W[1][0]

slope = -(b/w2)/(b/w1)  
intercept = -b/w2

x = np.linspace(-3, 3, 10)
y = slope*x + intercept

plt.title('Graph of Decision Boundary')
plt.plot(x, y, '-r', label='')

plt.show(block=False)
input('press <ENTER> to continue')
'''