import numpy as np


def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''

    centroids = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    indexes = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[indexes]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    # make sure you return a numpy array
    return np.asarray(centroids).astype(float)


def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    exp = 1 / p
    for centroid in centroids:
        distance = np.abs(X - centroid)
        distance = np.sum(distance ** p, axis=1)
        distance = distance ** exp
        distances.append(distance)
    distances = np.array(distances)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances


def kmeans_converge_centroids(X, k, p, max_iter, centroids):
    assignments = np.array([])
    for i in range(max_iter):
        distances = lp_distance(X, centroids, p)  # np.array of (k,num_of_pixels)
        assignments = np.argmin(distances, axis=0)
        # calculating new centroids:
        prev_centroids = centroids
        next_centroids = []
        for assignment in range(k):
            assignment_vector = np.argwhere(assignments == assignment)
            centroid_i = np.mean(X[assignment_vector], axis=0)
            next_centroids.append(centroid_i)
        centroids = np.array(next_centroids)
        # if np.all(prev_centroids == centroids): break

        if np.allclose(centroids, prev_centroids, rtol=0, atol=2):  # epsilon is atol, checking elementwise
            break
    classes = assignments
    return centroids, classes


def kmeans(X, k, p, max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    ###########################################################################
    # TODO: Implement the function.
    ###########################################################################
    centroids, classes = kmeans_converge_centroids(X, k, p, max_iter, centroids)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes


def kmeans_pp(X, k, p, max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = None
    centroids = []
    ###########################################################################
    # TODO: Implement the function.
    ###########################################################################
    first_centroid = X[np.random.randint(X.shape[0])]
    centroids.append(first_centroid)
    for i in range(k - 1):
        distances = lp_distance(X, centroids, 2)  # np.array of (k,num_of_pixels)
        # min value at every row will distance to closest centroid
        distances_to_closest_centroid = (np.amin(distances, axis=0)) ** 2
        probability_vector = distances_to_closest_centroid / np.sum(distances_to_closest_centroid)
        new_centroid_index = np.random.choice(X.shape[0], p=probability_vector)
        centroid = X[new_centroid_index]
        centroids.append(centroid)

    centroids = np.array(centroids)
    centroids, classes = kmeans_converge_centroids(X, k, p, max_iter, centroids)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes


def inertia(X, centroids, classes):
    distances = lp_distance(X, centroids, 2)
    distances_to_closest_centroid = np.amin(distances, axis=0)
    agg_distance = np.sum(distances_to_closest_centroid)

    return agg_distance


def kmeans_test(X, k, p, max_iter=100):
    centroids = get_random_centroids(X, k)
    centroids, classes = kmeans_converge_centroids_test(X, k, p, max_iter, centroids)
    return centroids, classes


def kmeans_pp_test(X, k, p, max_iter=100):
    centroids = []
    first_centroid = X[np.random.randint(X.shape[0])]
    centroids.append(first_centroid)
    for i in range(k - 1):
        distances = lp_distance(X, centroids, 2)  # np.array of (k,num_of_pixels)
        # min value at every row will distance to closest centroid
        distances_to_closest_centroid = (np.amin(distances, axis=0)) ** 2
        probability_vector = distances_to_closest_centroid / np.sum(distances_to_closest_centroid)
        new_centroid_index = np.random.choice(X.shape[0], p=probability_vector)
        centroid = X[new_centroid_index]
        centroids.append(centroid)

    centroids = np.array(centroids)
    centroids, classes = kmeans_converge_centroids_test(X, k, p, max_iter, centroids)
    return centroids, classes


def kmeans_converge_centroids_test(X, k, p, max_iter, centroids):
    assignments = np.array([])
    iterations = 0
    for i in range(max_iter):
        distances = lp_distance(X, centroids, p)  # np.array of (k,num_of_pixels)
        assignments = np.argmin(distances, axis=0)
        # calculating new centroids:
        prev_centroids = centroids
        next_centroids = []
        for assignment in range(k):
            assignment_vector = np.argwhere(assignments == assignment)
            centroid_i = np.mean(X[assignment_vector], axis=0)
            next_centroids.append(centroid_i)
        centroids = np.array(next_centroids)
        if np.allclose(centroids, prev_centroids, rtol=0, atol=6):  # epsilon is atol, checking elementwise
            break
    classes = assignments
    return centroids, classes
