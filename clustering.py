import numpy as np

# K_Means(X, K) , X is samples, K is number of clusters
#   Returns a set of K cluster centers based on Euclidean distance
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

# K_Means_better(X, K) , X is samples, K is number of clusters
#   Returns the best set of K cluster centers. Runs K_Means(X,K) a large minimum amount of times.
#   Chooses to return the set of K clusters centers that is returned a majority of the time.
#     1. Generates a set of K-Cluster centers with K_Means(X,K)
#     2. Keep count of how many times each unique set is generated
#     3. Run K_Means(X,K) many times
#     4. Run until set of cluster centers is returned a majority of the time
#     5. Return that set of cluster centers
def K_Means_better(X,K):
    f = X.shape[1]              # number of features
    n = 20                      # minimum number of times to run K_Means(X,K), 20 is arbitrary
    sets = np.zeros((n, K, f))  # generated sets of cluster centers, n sets of shape [K,f]
    counts = np.zeros(n)        # how many times some cluster set is returned

    # Generate cluster centers many times
    # Run minimum number of times, then start checking if a set of cluster centers is a majority
    # print("here0")
    while (np.sum(counts) < n) or (np.amax(counts) / np.sum(counts) <= .5) :
        # # Debugging, keep track of majority
        # if np.sum(counts) > 20 :
        #     print(np.amax(counts), "/", np.sum(counts), "=", np.amax(counts) / np.sum(counts))
        # Generate clusters
        C = K_Means(X, K)
        # Compare against every previously generated set of cluster centers
        for i in range(n):
            # If same as one generated before, add to that one's count
            if( np.array_equal(C, sets[i]) ):
                counts[i] += 1
                break
            # If not and this set of cluster centers is empty (all zeroes), fill it in
            elif ( np.array_equal(sets[i], np.zeros([K,f])) ):
                sets[i] = C
                counts[i] += 1
                break
            # If we're out of room, attach it
            else:
                sets = np.append(sets, [C], axis=0)
                # counts = np.append(counts, 1)
                break
    majority = np.argmax(counts)        #index of set of cluster centers with majority
    # #Debugging, show variables
    # print("counts: ", counts)
    # print("sets: ", sets)
    # print("majority: sets[", majority, "]:\n", sets[majority] )

    # return the set cluster centers with the most returns
    return sets[majority]

# calc_dist(a,n) , a is a point, b is a point
#   Finds Euclidean distance between two points
def calc_dist(a, b):
    dist = 0
    for i in range(a.shape[0]):
        dist = dist + np.square(a[i]-b[i])
    return np.sqrt(dist)