import numpy as np
from scipy.stats import multivariate_normal

THRESHOLD = 1e-7
MAX_ITERATION = 100


class GMM:
    def __init__(self, X, C):
        """
        :param X: image in RGB format
        :param C: number of classes in GMM, here default as 2, to discriminate foreground and background
        """
        self.X = X
        self.C = C
        self.h, self.w = X.shape[0], X.shape[1]

        self._initialize_parameters()

    def _initialize_parameters(self):

        self.pi_c = np.random.rand(self.C)
        self.mu = []
        self.sigma = np.zeros((self.C, self.w, self.w))

        covariance = np.cov(self.X, rowvar=False)
        for i in range(self.C):
            self.sigma[i, :, :] = covariance

        # tried to assign mu randomly but the segmentation results are unstable.
        # So instead manually pick the corner and the center point as mu
        # since by default the object in the foreground should be located close to the center: self.X[self.X.shape[0] // 2]
        # and the upper left corner: self.X[0] should be part of the background
        self.mu = [self.X[0], self.X[self.X.shape[0] // 2]]

        # update mixture gaussian distributions based on knowing updated mu and sigma in each iteration
        self.update_gmm_distributions()

        self.lower_bound = 0
        self.log_likelihood = 0

    def update_gmm_distributions(self):
        self.distributions = []
        for i in range(self.C):
            self.distributions.append(multivariate_normal(self.mu[i], self.sigma[i]))

    def update_prob_of_c(self):
        prob_of_c = np.ndarray((self.h, self.C))
        for c in range(self.C):
            prob_of_c[:, c] = self.pi_c[c] * self.distributions[c].pdf(self.X)
        return prob_of_c

    def update_q_function_of_c(self):
        """q(c_i) = p(c_i | x_i, \mu, \sigma)"""
        prob_of_c = self.update_prob_of_c()

        norm_across_c = np.sum(prob_of_c, axis=1).reshape(-1, 1)
        self.q_function_of_c = prob_of_c / norm_across_c

    def update_mu(self):
        # numerator: \sum_i q(t_i = c) X_i 
        # denominator: \sum_i q(t_i = c)

        # \sum_i q(t_i = c) X_i 
        numerator = np.matmul(np.transpose(self.q_function_of_c), self.X)
        sum_q_functions_of_c = np.sum(self.q_function_of_c, axis=0)  # sum_q_functions_of_c shape: (c, )

        # \sum_i q(t_i = c)
        self.sum_q_functions_of_c = sum_q_functions_of_c.reshape(sum_q_functions_of_c.shape[0], -1)
        self.mu = np.divide(numerator, self.sum_q_functions_of_c)

    def update_sigma(self):
        # numerator: \sum_i (x_i - \mu_c)^2 q(t_i=c)
        # denominator: \sum_i q(t_i = c)
        for c in range(self.C):
            # q(t_i = c)
            q_function_of_c = self.q_function_of_c[:, c]
            q_function_of_c = q_function_of_c.reshape(q_function_of_c.shape[0], -1)
            # \mu_c
            mean = self.mu[c, :]
            mean = mean.reshape(-1, mean.shape[0])
            # x_i - \mu_c
            norm_X = np.subtract(self.X, mean)
            # \sum_i (x_i - \mu_c)^2 q(t_i=c)
            numerator = np.matmul(np.transpose(norm_X), np.multiply(q_function_of_c, norm_X))
            numerator = np.transpose(numerator)

            updated_covariance_of_c = np.divide(numerator, self.sum_q_functions_of_c[c])
            self.sigma[c, :, :] = updated_covariance_of_c

    def update_pi(self):
        # pi_c = \sum_i^N q(t_i = c) / N, where N = image.shape[0]
        self.pi_c = np.divide(self.sum_q_functions_of_c, self.h)

    def kl_divergence(self):
        # in the M step, in order to maximize the lower bound
        # which can also be considered as minimizing the kl divergence between lower bound and p(c|\mu, \sigma)
        self.lower_bound = self.log_likelihood
        prob_of_c = self.update_prob_of_c()
        self.log_likelihood = np.sum(np.log(np.sum(prob_of_c, axis=1)), axis=0)
        if abs(self.log_likelihood - self.lower_bound) <= THRESHOLD:
            return

    def EM(self):
        for iteration in range(MAX_ITERATION):
            print('iteration:', iteration)
            # E step
            self.update_q_function_of_c()

            # M step
            self.update_mu()
            self.update_sigma()
            self.update_pi()

            # KL Divergence
            self.update_gmm_distributions()
            self.kl_divergence()
