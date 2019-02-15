from sklearn.neighbors import NearestNeighbors

from util.util import *


class Kmeans:

    def __init__(self, k, threshold=1e-6):
        self.threshold = threshold
        self.k = k

    def rand_center(self, data):
        n_samples, n_features = np.shape(data)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = data[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        print(">>> initial centroids")
        print(centroids)
        print('\n')
        return centroids

    def converged(self, centroids1, centroids2):
        diff = np.sum(np.abs(np.sum(centroids1 - centroids2, axis=1)), axis=0)
        print('distance between previous and new centroids: ', diff)
        print('\n')
        if diff < self.threshold:
            return True
        else:
            return False

    def closest_centroid(self, val, centroids):
        return np.argmin(np.sqrt(np.sum(np.square(val - centroids), axis=1)))

    def update_centroids(self, data, centroids):
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
