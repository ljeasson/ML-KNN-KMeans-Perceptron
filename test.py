import numpy as np
import perceptron as p

# Hand-Tested Data
X = np.array( [[1,1], [1,-1], [-1,1], [-1,-1]] )
Y = np.array( [[1], [-1], [-1], [-1]] )
W = p.perceptron_train(X,Y)
print("Hand-Tested Data    W1: ",W[0][0],"  W2: ",W[0][1],"  b:",W[1][0])
test_acc = p.perceptron_test(X,Y,W[0],W[1])
print("Accurancy:",test_acc,"\n")

# Percepton Test Data 1
X = np.array( [[0,1], [1,0], [5,4], [1,1], [3,3], [2,4], [1,6]] )
Y = np.array( [[1], [1], [-1], [1], [-1], [-1], [-1]] )
W = p.perceptron_train(X,Y)
print("Hand-Tested Data    W1: ",W[0][0],"  W2: ",W[0][1],"  b:",W[1][0])
test_acc = p.perceptron_test(X,Y,W[0],W[1])
print("Accurancy:",test_acc,"\n")

# Percepton Test Data 2
X = np.array( [[0,1], [1,0], [5,4], [1,1], [3,3], [2,4], [1,6]])
Y = np.array( [[1], [1], [-1], [1], [-1], [-1], [-1]] )
W = p.perceptron_train(X,Y)
print("Hand-Tested Data    W1: ",W[0][0],"  W2: ",W[0][1],"  b:",W[1][0])
test_acc = p.perceptron_test(X,Y,W[0],W[1])
print("Accurancy:",test_acc,"\n")

# Perceptron Test Data - Writeup
X = np.array( [[-2,1], [1,1], [1.5,-0.5], [-2,-1], [-1,-1.5], [2,-2]] )
Y = np.array( [[1], [1], [1], [-1], [-1], [-1]] )
W = p.perceptron_train(X,Y)
print("Hand-Tested Data    W1: ",W[0][0],"  W2: ",W[0][1],"  b:",W[1][0])
test_acc = p.perceptron_test(X,Y,W[0],W[1])
print("Accurancy:",test_acc,"\n")

'''
# Graph weight vector
#slope = -(b/w2)/(b/w1)  
#intercept = -b/w2

x = np.linspace(-3, 3, 10)
y = (-3/2)*x + b

plt.title('Graph of Resulting Weight Vector')
plt.plot(x, y, '-r', label='')

plt.show(block=False)
input('press <ENTER> to continue')
'''