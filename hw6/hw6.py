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
    np.random.seed(42)
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

    # diff = X - centroids[:, np.newaxis]
    # abs = np.abs(diff)
    # powered = abs ** p
    # summed = np.sum(powered, axis=-1)
    # distances = summed ** (1.0 / p)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return distances

def kmeans(X, k, p ,max_iter=100):
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
    # TODO: Implement the function.                                           #
    ###########################################################################
    prev_centroids = centroids
    for n in range(max_iter):
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)
        prev_centroids = centroids
        centroids = np.array([np.mean(X[classes == i, :], axis=0) for i in range(k)])
        if np.all(prev_centroids == centroids): break
        # if np.allclose(centroids, prev_centroids, rtol=0, atol=2): break
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes

def kmeans_pp(X, k, p ,max_iter=100):
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
    # TODO: Implement the function.                                           #
    ###########################################################################
    first_centroid = X[np.random.randint(X.shape[0])]
    centroids.append(first_centroid)
    for i in range(k - 1):
        distances = lp_distance(X, centroids, 2)  # np.array of (k, num_of_pixels)
        # min value at every row will distance to closest centroid
        distances_to_closest_centroid = (np.amin(distances, axis=0)) ** 2
        probability_vector = distances_to_closest_centroid / np.sum(distances_to_closest_centroid)
        new_centroid_index = np.random.choice(X.shape[0], p=probability_vector)
        centroid = X[new_centroid_index]
        centroids.append(centroid)

    centroids = np.array(centroids)
    prev_centroids = centroids
    for n in range(max_iter):
        distances = lp_distance(X, centroids, p)
        classes = np.argmin(distances, axis=0)
        prev_centroids = centroids
        centroids = np.array([np.mean(X[classes == i, :], axis=0) for i in range(k)])
        if np.all(prev_centroids == centroids): break
        # if np.allclose(centroids, prev_centroids, rtol=0, atol=2):  break
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return centroids, classes


def get_compressed_img(rows, cols, classes, centroids): # build the image by the given centroids and classes
    classes = classes.reshape(rows, cols) 
    img_comp = np.zeros((classes.shape[0], classes.shape[1],3), dtype=np.uint8)
    for i in range(classes.shape[0]):
        for j in range(classes.shape[1]):
                img_comp[i,j,:] = centroids[classes[i,j],:]

    return img_comp

def get_img_diff(img_org, img_comp):
    img_comp = img_comp.reshape(img_comp.shape[0]*img_comp.shape[1],3) # flatten the composed image, due to other function output format
    img_org = np.array(img_org, dtype=float)
    img_comp = np.array(img_comp, dtype=float)
    diff = ((img_org - img_comp)**2).mean(axis=0).mean() # we use the MSE function to measure the difference between the two images
    return diff