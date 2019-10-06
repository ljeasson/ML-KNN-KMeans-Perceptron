import numpy as np
import matplotlib.pyplot as plt

# class K_Means():
#     def _init_(self, K, mean, )


def K_Means(X,K):
    n = X.shape[0]  # number of samples
    f = X.shape[1]  # number of features

    # Generate random centers
    C = np.zeros([K,f])     # cluster centers, updated and returned by this function
    C_n = np.zeros([K,f])   # number of samples in cluster of same index
    for i in range(K):
        x = np.random.randint(0, n)
        if(x != C[i-1]):
            C[i] = X[x]
        else:
            i -= 1

    # For each sample
    for i in range(X.shape[0]):
        min_dist = 9999999999   # arbitrarily large initialization
        c_i = 0                 # index of the center that sample X[i] belongs too 
        for j in range(K):
            dist = calc_dist(C[j], X[i])    # C[j] : center value being tried
            if(dist < min_dist):
                min_dist = dist
                c_i = j
        # Update the center
        current_sum = C[c_i] * C_n[c_i]
        C_n[c_i] += 1 # update number of samples in cluster
        C[c_i] = ( current_sum + X[i] ) / C_n[c_i]

    return C

def K_Means_better(X,K):
    return 0

def calc_dist(a, b):
    dist = 0
    # print("center: ", center)
    # print("sample: ", sample)
    # print("shape: ", )
    for i in range(a.shape[0]):
        dist = dist + np.square(a[i]-b[i])
    return np.sqrt(dist)

#unfinished
def plotClusters(samples, groupings):
    # Separating x_1 and x_2 for clusters a and b
    x_1_a = np.array([])
    x_2_a = np.array([])
    x_1_b = np.array([])
    x_2_b = np.array([])
    for i in range(samples.shape[0]):
        if (groupings[i] == 1):
            x_1_a = np.append(x_1_a, samples[i][0])
            x_2_a = np.append(x_2_a, samples[i][1])
        else:
            x_1_b = np.append(x_1_b, samples[i][0])
            x_2_b = np.append(x_2_b, samples[i][1])
    # Plotting
    plt.scatter(x_1_a, x_2_a, label='test a')
    plt.scatter(x_1_b, x_2_b, label='test b')
    plt.title('Samples')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend()
    plt.show()


# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Part ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
X = np.array( [[0], [1], [2], [7], [8], [9], [12], [14], [15]] )
K = 3
C = K_Means(X, K)

# Visuals for debugging
print(C)
y_c = np.ones((K,1))
y_s = np.zeros((X.shape[0],1))

for i in range(K):
    plt.scatter(C, y_c, label='centers')
    plt.scatter(X, y_s, label='samples')
plt.title('X')
plt.show()

# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Part ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# X_2 = np.array( [ [1, 0], [7, 4], [9, 6], [2, 1], [4, 8], [0, 3], [13, 5], [6, 8], [7, 3], [3, 6], [2, 1], [8, 3], [10, 2], [3, 5], [5, 1], [1, 9], [10, 3], [4, 1], [6, 6], [2, 2] ] )
#C_2 = K_Means_better(X_2, K)
#print(C_2)

