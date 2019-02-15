from sklearn.neighbors import NearestNeighbors

from util.util import *


class Kmeans:

    def __init__(self, k, threshold=1e-6):
        self.threshold = threshold
        self.k = k

    def rand_center(self, data):
        """
        :param data: transformed images as 5-dim vectors, (R,G,B,x,y)
        :return: k centroids, shape: k x 5
        """
        n_samples, n_features = np.shape(data)
        print('n_features', n_features)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = data[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        print(">>> initial centroids")
        print(centroids)
        print('\n')
        return centroids

    def converged(self, centroids1, centroids2):
        """
        :param centroids1: centroids before update, shape: k x 5
        :param centroids2: centroids after update, shape: k x 5
        :return: True / False
        """
        diff = np.mean(np.sqrt(np.sum(np.square(centroids1 - centroids2), axis=1)), axis=0)
        print('distance between previous and new centroids: ', diff)
        print('\n')
        if diff < self.threshold:
            return True
        else:
            return False

    def closest_centroid(self, val, centroids):
        """
        :param val: each data point of input, shape: 1 x 5
        :param centroids: current centroids, shape: k x 5
        """
        return np.argmin(np.sqrt(np.sum(np.square(val - centroids), axis=1)))

    def update_centroids(self, data, centroids):
        """
        :param data: all data points of input, shape: (h*w) x 5
        :param centroids: shape: k x 5
        :return: updated centroids, shape: k x 5
                clusters of each k, shape: k x _
                cluster labels for each data point, shape: (h*w) x 1
        """
        clusters = [[] for _ in range(self.k)]

        # n_samples, n_features = np.shape(data)
        # labels = np.zeros((n_samples))
        #
        # for idx, val in enumerate(data):
        #     val_label = self.closest_centroid(val, centroids)
        #     clusters[val_label].append(val)
        #     labels[idx] = val_label
        # centroids = np.zeros((self.k, n_features))

        # in this part, to determine the class label for each pixel (R, G, B, x, y), use NearestNeighbors in sklearn
        # for faster computation, because KD-Tree is constructed to boost up indexing efficiency when calling NN
        nbrs = NearestNeighbors(n_neighbors=1).fit(centroids)
        labels = nbrs.kneighbors(data, return_distance=False)
        labels = labels.flatten()

        for idx, label in enumerate(labels):
            clusters[label].append(data[idx])

        centroids = np.zeros((self.k, data.shape[1]))

        for idx, cluster_val in enumerate(clusters):
            centroid = np.mean(cluster_val, axis=0)
            centroids[idx] = centroid
        return centroids, clusters, labels

    def cluster(self, data):
        """
        :param data: transformed images as 5-dim vectors, (R,G,B,x,y)
        :return: final_centroids, shape: k x 5
                 final clusters of each k, shape: k x _
                 final cluster labels for each data point, shape: (h*w) x 1
        """
        centroids = self.rand_center(data)
        converge = False
        iteration = 0
        while not converge:
            old_centroids = np.copy(centroids)
            centroids, clusters, labels = self.update_centroids(data, old_centroids)
            print('iteration: ', iteration)
            print('centroids: ')
            print(centroids)
            converge = self.converged(old_centroids, centroids)
            iteration += 1
        print('number of iterations to converge: ', iteration)
        print(">>> final centroids")
        print(centroids)
        return centroids, clusters, labels

