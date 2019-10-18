import numpy as np
import matplotlib.pyplot as plt

# K_Means(X, K) , X is samples, K is number of clusters
#   Computes K cluster centers based on Euclidean distance
#     1. Generates unique random indexes to use as initial cluster centers
#     2. Assigns each sample to a cluster center based on smallest Euclidean distance using calc_dist()
#     3. Updates cluster center as a sample is added to its cluster
#     4. Returns cluster centers
def K_Means(X, K):
    n = X.shape[0]  # number of samples
    f = X.shape[1]  # number of features

    # print("n: ", n, "\tf: ", f, "\tK: ", K)

    # Generate random indeces for centers
    C = np.empty([K,f])     # cluster centers, updated and returned by this function
    C_n = np.zeros(K)       # number of samples in cluster of same index
    # print("Cz: \n", C)

    i = 0
    while i < K :
        x = np.random.randint(0, n)
        # print("x: ", x, ",\ti: ", i)
        # print("X[x]: \n", X[x])
        if(X[x] not in C):    # if already a random center, set back i to regenerate (so we have unique centers)
            C[i] = X[x]
            i += 1

    # print("C: \n", C)
    # For each sample, assign to a center and update that center
    for i in range(X.shape[0]):
        min_dist = 9999999999   # arbitrarily large initialization
        c_i = 0                 # index of the center that sample X[i] belongs too
        # choose a center c_i that sample X[i] is closest to
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
    f = X.shape[1]              # number of features
    n = 20                      # minimum number of times to run K_Means(X,K)
    sets = np.zeros((n, K, f))  # generated sets of cluster centers, n sets of shape [K,f]
    counts = np.zeros(n)        # how many times some cluster set is returned

    # Generate cluster centers many times
    # Run minimum number of times, then start checking if a set of cluster centers is a majority
    # print("here0")
    while (np.sum(counts) < n) or (np.amax(counts) / np.sum(counts) <= .5) :
        if np.sum(counts) > 198 :
            print(np.amax(counts) / np.sum(counts))
        # Generate clusters
        C = K_Means(X, K)
        print("C: \n", C)
        # Compare against every previously generated set of cluster centers
        for i in range(n):
            # If same as one generated before, add to that one's count
            if( np.array_equal(C, sets[i]) ):
                counts[i] += 1
                break
            # If not and this set of cluster centers is empty (all zeroes), fill it in
            elif ( np.array_equal(sets[i], np.zeros([K,f])) ):
                sets[i] = C
                counts[i] += 0
                break
            # If we're out of room, attach it
            else:
                sets = np.append(sets, [C], axis=0)
                break
    # return the set cluster centers with the most returns
    print("counts: ", counts)
    print("sets: ", sets)
    return sets[0:1]

# calc_dist(a,n) , a is a point, b is a point
#     Finds Euclidean distance between two points
def calc_dist(a, b):
    dist = 0
    for i in range(a.shape[0]):
        dist = dist + np.square(a[i]-b[i])
    return np.sqrt(dist)

# unfinished
def plotClusters(X,C):
    return 0
    



# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Test Zone (Ignore) (it's just me learning python lol) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# X = np.array( [ [1, 0], [7, 4], [9, 6], [2, 1], [4, 8], [0, 3], [13, 5], [6, 8], [7, 3], [3, 6], [2, 1], [8, 3], [10, 2], [3, 5], [5, 1], [1, 9], [10, 3], [4, 1], [6, 6], [2, 2] ] )
# K = 2
# f = 3
# n = 20                     # minimum number of times to run K_Means(X,K)
# sets = np.zeros((K,f,n))    # hold generated n cluster sets of shape [K,f]
# set_counts = np.zeros(n)    # how many times some cluster set is returned

# print("X: \n", X)
# print("K: \n", K)
# print("f: \n", f)
# print("n: \n", n)
# print("sets: \n", sets)
# print("set_counts: \n", set_counts)

# a = np.array([[[1,2], [4,5]], [[1,2], [4,5]]])
# b = np.array([[[1,2], [4,5]], [[2,2], [4,5]]])
# c = np.zeros((4,2,3))
# print("c: \n", c)
# p = np.array([[1,2,1],[4,5,4]])
# d = np.array(p)
# c = np.append(c, [d], axis=0)
# print("c: \n", c)
# print("c[]: \n", c[1])

# print("d: \n", d)


# if(np.array_equal(a[1,0,0],b[1,0,0])):
#     print("Yes")
# else:
#     print("No")


# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ First Part ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# X = np.array( [[0], [1], [2], [7], [8], [9], [12], [14], [15]] )
# K = 3
# C = K_Means(X, K)

# # Visuals for debugging
# print("X: \n", X)
# print("C: \n", C)
# plt.scatter(C, np.ones((C.shape[0],1)), label='centers')
# plt.scatter(X, np.zeros((X.shape[0],1)), label='samples')
# plt.title('X')
# plt.show()

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Second Part ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
X_2 = np.array( [ [1, 0], [7, 4], [9, 6], [2, 1], [4, 8], [0, 3], [13, 5], [6, 8], [7, 3], [3, 6], [2, 1], [8, 3], [10, 2], [3, 5], [5, 1], [1, 9], [10, 3], [4, 1], [6, 6], [2, 2] ] )
K_2 = 2
C_2 = K_Means_better(X_2, K_2)

# Visuals for debugging
print("X_2: \n", X_2)
print("C_2: \n", C_2)
plt.scatter(C_2[:,0], C_2[:,1], label='centers')
plt.scatter(X_2[:,0], X_2[:,1], label='samples')
plt.title('X_2')
plt.show()