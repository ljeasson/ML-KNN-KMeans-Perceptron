import numpy as np
import matplotlib.pyplot as plt
import perceptron as p
import nearest_neighbors as knn
import clustering as km

# KNN TESTING
X_train = np.array( [[1,5], [2,6], [2,7], [3,7], [3,8], [4,8], [5,1], [5,9], [6,2], [7,2], [7,3], [8,3], [8,4], [9,5]] )
Y_train = np.array( [[-1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [1]] )

X_test = np.array( [[1,1], [2,1], [0,10], [10,10], [5,5], [3,10], [9,4], [6,2], [2,2], [8,7]] )
Y_test = np.array( [[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]] )

acc_K1 = knn.KNN_test(X_train, Y_train, X_test, Y_test, 1)
acc_K2 = knn.KNN_test(X_train, Y_train, X_test, Y_test, 3)
acc_K3 = knn.KNN_test(X_train, Y_train, X_test, Y_test, 5)
print("Accuracy K=1:",acc_K1)
print("Accuracy K=3:",acc_K2)
print("Accuracy K=5:",acc_K3)
print()

best_K = knn.choose_K(X_train, Y_train, X_test, Y_test)
print("Best K:",best_K,"\n")
print()

# K-Means TESTING
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Part ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
X = np.array( [[0], [1], [2], [7], [8], [9], [12], [14], [15]] )
K = 3
C = km.K_Means(X, K)

# Visuals for debugging, Uncomment matplot header to use
print("C: \n", C)
plt.scatter(C, np.ones((C.shape[0],1)), label='centers')
plt.scatter(X, np.zeros((X.shape[0],1)), label='samples')
plt.title('X, K=3')
# plt.savefig("k_means_results_1.png")  #Uncomment to save plot as file
plt.show()

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Part ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
X_2 = np.array( [ [1, 0], [7, 4], [9, 6], [2, 1], [4, 8], [0, 3], [13, 5], [6, 8], [7, 3], [3, 6], [2, 1], [8, 3], [10, 2], [3, 5], [5, 1], [1, 9], [10, 3], [4, 1], [6, 6], [2, 2] ] )
K_2 = 2
C_2 = km.K_Means_better(X_2, K_2)

# Visuals for debugging, Uncomment matplot header to use
print("C_2: \n", C_2)
plt.scatter(C_2[:,0], C_2[:,1], label='centers')
plt.scatter(X_2[:,0], X_2[:,1], label='samples')
plt.title('X_2, K=2')
# plt.savefig("k_means_results_2.png")  #Uncomment to save plot as file
plt.show()


K_3 = 3
C_3 = km.K_Means_better(X_2, K_3)
# Visuals for debugging, Uncomment matplot header to use
print("C_3: \n", C_3)
plt.scatter(C_3[:,0], C_3[:,1], label='centers')
plt.scatter(X_2[:,0], X_2[:,1], label='samples')
plt.title('X_2, K=3')
# plt.savefig("k_means_results_3.png")  #Uncomment to save plot as file
plt.show()


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