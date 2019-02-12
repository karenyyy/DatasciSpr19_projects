import numpy as np

THRESHOLD = 10


class RegionGrowing:
    def __init__(self, X, seeds):
        """
        :param X: RGB image
        :param seeds: user input starting points through the GUI
        """
        self.X = X
        self.h, self.w = X.shape
        self.seeds = seeds
        self.segmented_img = np.zeros((self.h, self.w))
        self.region_stack = []

        for idx in range(len(self.seeds)):
            # loop through all user input seeds and push them into stack
            x = int(self.seeds[idx][0])
            y = int(self.seeds[idx][1])

            self.segmented_img[x, y] = 255.0

            self.region_stack.append([x, y])

    def region_growing(self):
        iteration = 0
        while len(self.region_stack) > 0:
            print('iteration:', iteration)

            seed = self.region_stack.pop(0)
            tmp_x = seed[0]
            tmp_y = seed[1]

            # starting with this point
            pixel = self.X[tmp_x, tmp_y]

            # assign lower and upper bound, push all unvisited adjacent pixels within the range into stack
            lowerbound = pixel - THRESHOLD
            upperbound = pixel + THRESHOLD

            for x_shift in [-1, 0, 1]:
                for y_shift in [-1, 0, 1]:
                    try:
                        if self.segmented_img[tmp_x + x_shift, tmp_y + y_shift] != 255 and \
                                lowerbound < self.X[tmp_x + x_shift, tmp_y + y_shift] < upperbound:
                            self.segmented_img[tmp_x + x_shift, tmp_y + y_shift] = 255
                            if [tmp_x + x_shift, tmp_y + y_shift] not in self.region_stack and \
                                    0 < tmp_x + x_shift < self.h and \
                                    0 < tmp_y + y_shift < self.w:
                                self.region_stack.append([tmp_x + x_shift, tmp_y + y_shift])
                            else:
                                # otherwise put it back into background
                                self.segmented_img[tmp_x + x_shift, tmp_y + y_shift] = 0
                    except Exception as e:
                        print(e)

            iteration += 1
