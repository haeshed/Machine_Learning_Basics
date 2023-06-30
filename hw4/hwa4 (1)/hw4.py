import numpy as np


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def h_theta(self, X):
        return 1 / (1 + np.exp(-X.dot(self.theta)))

    def cost_func(self, X, y):
        h_x = self.h_theta(X)
        m = X.shape[0]
        a = - y * np.log(h_x)
        b = (1-y) * np.log(1 - h_x)
        # num_of_true = np.sum(a - b)
        cost = np.mean(a - b)
        # return num_of_true/m
        return cost

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = np.column_stack((np.ones(X.shape[0]), X))
        self.theta = np.random.randn(X.shape[1])

        for iteration in range(self.n_iter):
            self.Js.append(self.cost_func(X, y))
            self.thetas.append(self.theta)
            # print(self.Js[-1])

            if iteration > 0 and abs(self.Js[-1] - self.Js[-2]) < self.eps:
                break

            self.theta = self.theta - \
                (self.eta * np.dot(X.T, (self.h_theta(X) - y)))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        X = np.column_stack((np.ones(X.shape[0]), X))
        preds = self.h_theta(X)
        # prob is a vector with probability [0-1], need to over-ride
        preds = np.around(preds)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

    def accuracy(self, X_val, y_val):
        """
        returns accuracy of the model avg of prediction (#correct predictions / #all data-points)
        """
        predict_labels = self.predict(X_val)
        acc = np.mean(predict_labels == y_val)
        return acc


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = 0

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    acc = None
    X_shuff, y_shuff = unison_shuffled_copies(X, y)

    X_folds, y_folds = np.split(X_shuff, folds), np.split(y_shuff, folds)
    X_train, X_test, y_train, y_test = None, None, None, None
    # print(X, y)
    # print(X_shuff, y_shuff)
    # print(len(X_folds[0].shape))

    for fold in range(folds):
        X_train, X_test = np.concatenate(
            np.delete(X_folds, fold, axis=0)), X_folds[fold]
        y_train, y_test = np.concatenate(
            np.delete(y_folds, fold, axis=0)), y_folds[fold]
        algo.fit(X_train, y_train)
        acc = algo.accuracy(X_test, y_test)
        # print("fold: ", fold, "acc: ", acc)
        cv_accuracy += acc
    cv_accuracy /= folds
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.

    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.

    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    denominator = sigma * np.sqrt(2 * np.pi)
    nominator = np.exp(-0.5 * ((data - mu) / sigma) ** 2)
    return (nominator / denominator)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        np.random.seed(self.random_state)
        self.weights = np.full(self.k, (1 / self.k))  # uniform
        self.mus = np.random.random(self.k)  # random vector size k
        self.sigmas = np.random.random(self.k)  # random vector size k
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        likelihood = []
        for i in range(self.k):
            likelihood.append(
                self.weights[i] * norm_pdf(data, self.mus[i], self.sigmas[i]))
        self.responsibilities = (np.array(likelihood) / sum(likelihood)).T
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.weights = self.responsibilities.mean(axis=0)
        m = data.shape[0]
        for i in range(self.k):
            self.mus[i] = self.responsibilities[:, i].dot(data) / (m * self.weights[i])
            self.sigmas[i] = np.sqrt(self.responsibilities[:, i].dot((data - self.mus[i]) ** 2) / (m * self.weights[i]))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def step_maximization(self, data):
        r = []
        for j in range(self.k):
            r.append((self.weights[j] * self.responsibilities[j]) / (np.sum([self.weights[i] * self.responsibilities[i] for i in range(self.k)], axis=0)))
            self.mus[j] = np.sum(r[j] * data) / (np.sum(r[j]))
            self.sigmas[j] = np.sum(r[j] * np.square(data - self.mus[j])) / (np.sum(r[j]))
            self.responsibilities[j] = np.mean(r[j])
    
    def max_3(self, data):
        self.weight = self.responsibilities.mean(axis=0)
        self.mean = np.dot(data, self.responsibilities/len(data)) / self.weight
        data_duplicated = np.array([data for _ in range(self.k)]).T
        self.std = np.sqrt((((data_duplicated - self.mean)**2) * self.responsibilities).mean(axis=0) / self.weight)

    def compute_cost(self, data):
        """cost function - log probability"""
        cost_array = np.zeros(self.k)
        for gaus in range(self.k):
            cost_array[gaus] = - np.log(self.weights[gaus] * norm_pdf(data, self.mus[gaus], self.sigmas[gaus])).sum()
        return cost_array

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.init_params(data)
        self.cost_history = [self.compute_cost(data)]
        for iter in range(self.n_iter):
            if iter > 0 and np.max(abs(self.cost_history[-1] - self.cost_history[-2])) < self.eps:
                break
            self.expectation(data)
            # self.maximization(data)
            # self.step_maximization(data)
            self.max_3(data)
            self.cost_history.append(self.compute_cost(data))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.

    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.

    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = []
        self.labels = []
        self.params = {}
        
    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.labels = np.unique(y)
        for label in self.labels:
            tmp = np.where(y == label)[0].shape[0] / y.shape[0]
            self.prior.append(tmp)
          
        for label in self.labels:
            X_of_label = X[y == label]
            for feature in range(X.shape[1]):
                X_feature_of_label = X_of_label[:,feature]
                em_model = EM(self.k)
                em_model.fit(X_feature_of_label)
                self.params[(feature, label)] = em_model.get_dist_params()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if len(X.shape) == 1:
            X = np.array([X])
        # perform likelihood estimation
        likelihood_mat = []
        for label in self.labels:
            likelihood_of_label = np.zeros(X.shape)
            for feature in self.labels:
                mean, std, weights = self.params[(feature, label)]
                for i in range(self.k):
                    likelihood_of_label[:, feature] += weights[i] * norm_pdf(X[:, feature], mean[i], std[i])
            likelihood_mat.append(np.prod(likelihood_of_label, axis=1))

        preds = np.empty(X.shape[0])
        for instance_idx, instance in enumerate(X):
            #(label, probability to be in the label)
            most_likely_label = (None, -1)
            for label_idx in enumerate(self.labels):
                prior = self.prior[label_idx]
                likelihood_cur = likelihood_mat[label_idx][instance_idx]
                prob = prior * likelihood_cur
                if prob > most_likely_label[1]:
                    most_likely_label = (label_idx, prob)
            preds[instance_idx] = most_likely_label[0]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds
    
    def predict2(self, X):
        """Return the predicted class label"""
        if len(X.shape) == 1:
            X = np.array([X])
        # perform likelihood estimation
        likelihood_mat = []
        for label in self.labels:
            likelihood_of_label = np.zeros(X.shape)
            for feature in range(X.shape[1]):
                mean, std, weights = self.params[(feature, label)]
                for i in range(self.k):
                    likelihood_of_label[:, feature] += weights[i] * norm_pdf(X[:, feature], mean[i], std[i])
            likelihood_mat.append(np.prod(likelihood_of_label, axis=1))

        prediction = np.empty(X.shape[0])
        for instance_idx, instance in enumerate(X):
            #(label, probability to be in the label)
            most_likely_label = (None, -1)
            for label_idx, label in enumerate(self.labels):
                prior = self.prior[label_idx]
                likelihood_cur = likelihood_mat[label_idx][instance_idx]
                prob = prior * likelihood_cur
                if prob > most_likely_label[1]:
                    most_likely_label = (label, prob)
            prediction[instance_idx] = most_likely_label[0]
        return prediction

    def accuracy(self, X_val, y_val):
        """
        returns accuracy of the model avg of prediction (#correct predictions / #all data-points)
        """
        predict_labels = self.predict2(X_val)
        # diff = np.where(predict_labels == y_val)
        # print(diff)
        sum = np.sum(predict_labels == y_val)
        print(sum)
        return sum / len(X_val)


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    '''

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    logistic_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    naive_em_model = NaiveBayesGaussian(k)
    # training models
    logistic_model.fit(x_train, y_train)
    naive_em_model.fit(x_train, y_train)

    lor_train_acc = logistic_model.accuracy(x_train, y_train)
    lor_test_acc = logistic_model.accuracy(x_test, y_test)

    bayes_train_acc = naive_em_model.accuracy(x_train, y_train)
    bayes_test_acc = naive_em_model.accuracy(x_test, y_test)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'dataset_a_features': dataset_a_features,
            'dataset_a_labels': dataset_a_labels,
            'dataset_b_features': dataset_b_features,
            'dataset_b_labels': dataset_b_labels
            }
