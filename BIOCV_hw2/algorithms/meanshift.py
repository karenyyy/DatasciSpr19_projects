from sklearn.neighbors import NearestNeighbors
from util.util import *

THRESHOLD = 1e-6


class Meanshift:

    def __init__(self, X, r):
        self.w, self.h, c = X.shape
        X = X.reshape(self.w * self.h, c)
        # _, _, X = transform_img_5_dim(X)
        self.X = X
        self.clusters = {}
        self.radius = r
        self.nbrs = NearestNeighbors(radius=self.radius).fit(self.X)
        self.centroids = self.X[np.random.choice(range(self.X.shape[0]), 100)]

        print('centroids:')
        print(self.centroids)

    def update_centroids(self):
        cnt = 0

        print('final controids: ')
        print()
        for centroid in self.centroids:
            iteration = 0
            current_centroid = centroid

            while True:
                neighbor_idxs = self.nbrs.radius_neighbors([current_centroid], self.radius, return_distance=False)[0]
                # neighbor_idxs = [idx for idx, pixel in enumerate(self.X) if euclidean_dist(current_centroid, pixel) < self.radius]
                neighbor_to_centroid = self.X[neighbor_idxs]

                new_centroid = np.mean(neighbor_to_centroid, 0)
                dist = euclidean_dist(new_centroid, current_centroid)

                print('iteration: ', iteration)
                print('dist: ', dist)

                if dist < THRESHOLD * self.radius or iteration == 300:
                    self.clusters[tuple(new_centroid)] = len(neighbor_idxs)
                    break
                else:
                    current_centroid = new_centroid

                iteration += 1

            cnt += 1

    def cluster(self):

        sorted_density = sorted(self.clusters.items(), key=lambda x: x[1], reverse=True)
        final_centroids = np.array([x[0] for x in sorted_density])

        print('final_centroids:')
        print(final_centroids)

        nbrs = NearestNeighbors(n_neighbors=1).fit(final_centroids)
        labels = nbrs.kneighbors(self.X, return_distance=False)
        labels = labels.flatten()
        return final_centroids, labels

    def segment(self):
        final_centroids, labels = self.cluster()
        labels = np.reshape(labels, [self.w, self.h])
        segmented = np.zeros((self.w, self.h, 3), np.uint8)
        for i in range(self.w):
            for j in range(self.h):
                segmented[i][j] = final_centroids[labels[i][j]][0:3]
        cv2.imwrite('segmented.png', segmented)
        return segmented
