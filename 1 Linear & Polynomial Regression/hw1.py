###### Your ID ######
# ID1: 205947856
# ID2: 204300925
#####################

# imports
import numpy as np
import pandas as pd


def preprocess(X, y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    ###########################################################################
    # TODO: Implement the normalization function.                             #
    ###########################################################################
    meanX = np.mean(X, axis=0)  # computing mean, max, min for X
    maxX = np.max(X, axis=0)
    minX = np.min(X, axis=0)
    X = (X-meanX)/(maxX-minX)  # using the formula

    meanY = np.mean(y)  # same for y
    maxY = np.max(y)
    minY = np.min(y)
    y = (y-meanY)/(maxY-minY)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X, y


def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """
    ###########################################################################
    # TODO: Implement the bias trick by adding a column of ones to the data.                             #
    ###########################################################################
    ones = [1]*len(X)  # creating an aproptiate sized array
    X = np.c_[ones, X]  # appending to X
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return X


def compute_cost(X, y, theta):
    """
    Computes the average squared difference between an observation's actual and
    predicted values for linear regression.  

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: the parameters (weights) of the model being learned.

    Returns:
    - J: the cost associated with the current set of parameters (single number).
    """

    J = 0  # We use J for the cost.
    ###########################################################################
    # TODO: Implement the MSE cost function.                                  #
    ###########################################################################
    distance = X.dot(theta)
    distance = distance - y
    # square distance of a point (of all the array)
    distance = np.square(distance)
    sum = np.sum(distance)
    J = sum/(2*len(X))  # all the costs^2 summed up and averaged
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the gradient descent optimization algorithm.            #
    ###########################################################################
    distance = 0

    for i in range(num_iters):  # run the algorithm the specified number of times
        distance = X.dot(theta) - y # compute theta according to the formula
        theta -= (1/(len(X)))*alpha*X.T.dot(distance) # add the current cost to the J_history
        J_history.insert(i, compute_cost(X, y, theta))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """

    pinv_theta = []
    ###########################################################################
    # TODO: Implement the pseudoinverse algorithm.                            #
    ###########################################################################
    pinvX = X.T.dot(X)  # (X^T)*X
    pinvX = np.linalg.inv(pinvX)  # inverse of prev
    pinvX = pinvX.dot(X.T)  # prev * X^T
    pinv_theta = pinvX.dot(y)  # prev * y
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pinv_theta


def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """

    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the cost value in every iteration
    ###########################################################################
    # TODO: Implement the efficient gradient descent optimization algorithm.  #
    ###########################################################################
    err = 1e-8  # specified error rate for stopping

    for i in range(num_iters):  # max number of iterations (if not stopped by theta first)
        distance = X.dot(theta) - y # same as in the non-efficient function
        theta = theta - (1/(len(X)))*alpha*X.T.dot(distance)
        cost = compute_cost(X, y, theta) # if cost hasn't changed by err, stop
        if (i > 0) and (J_history[-1]-cost) < err:
            break
        else:
            J_history.insert(i, cost)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return theta, J_history


def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """

    alphas = [0.00001, 0.00003, 0.0001, 0.0003,
              0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {}  # {alpha_value: validation_loss}
    ###########################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    thetaRandom = np.random.random(size=X_train[1].shape) # randimise opening thatas
    for a in alphas: # do for all alphas provided
        tempTheta = efficient_gradient_descent(X_train, y_train, thetaRandom, a, iterations)[0] # train model using current alpha
        alpha_dict[a] = compute_cost(X_val, y_val, tempTheta) # add cost to dict
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return alpha_dict


def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    selected_features = []
    ##### c######################################################################
    # TODO: Implement the function and find the best alpha value.             #
    ###########################################################################
    cost = {}
    for num in range(5): # run specified number of times (5)
        for i in range(X_train.shape[1]): # for each feature, do
            if selected_features.count(i) == 0: # if not already chosen previously
                selected_features.append(i) # add temporarily
                thetaRandom = np.random.random(len(selected_features)+1)
                tempTheta = efficient_gradient_descent(apply_bias_trick(X_train[:, selected_features]), y_train, thetaRandom, best_alpha, iterations)[0]
                cost[i] = compute_cost(apply_bias_trick(X_val[:, selected_features]), y_val, tempTheta) # check alg and add to cost list
                selected_features.remove(i) 
        bestFeature = min(cost, key=cost.get)
        selected_features.append(bestFeature) # adds the minimum cost feature to the selected list
        cost.clear()

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return selected_features


def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

    df_poly = df.copy()
    ###########################################################################
    # TODO: Implement the function to add polynomial features                 #
    ###########################################################################
    polX = df_poly.values
    labelNames = df_poly.columns.tolist()
    lenDF = polX.shape[1]
    for i in range(lenDF):
        for j in range(i, lenDF):
            col_ij = polX[:, i]*polX[:, j]
            polX = np.c_[polX, col_ij]
            if (i == j):
                labelNames.append(df_poly.columns[i]+"^2")
            else:
                labelNames.append(df_poly.columns[i]+"*"+df_poly.columns[j])
    df_poly = pd.DataFrame(polX, columns=labelNames)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return df_poly
