from PyQt5.QtCore import pyqtSlot, QBasicTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QProgressBar
from PyQt5.uic import loadUi
from qtpy import QtCore

from algorithms.gmm import *
from algorithms.grabcut_opencv import GrabcutOpenCV
from algorithms.kmeans import *
from algorithms.meanshift import *
from algorithms.region_growing import RegionGrowing
from algorithms.watershed_opencv import WatershedOpenCV

seed_list = []

class GUI(QDialog):
    def __init__(self):
        super(GUI, self).__init__()
        loadUi('gui/mainframe.ui', self)
        self.image = None
        self.gray_image = None
        self.segmented_image = None
        self.restore_segmented_image = None
        self.progressBar.hide()
        self.kmeansComboBox.hide()
        self.meanshiftComboBox.hide()
        self.runKmeansButton.hide()
        self.runMeanshiftButton.hide()
        self.timer = QBasicTimer()
        self.step = 0
        self.loadButton.clicked.connect(self.loadClick)
        self.saveButton.clicked.connect(self.saveClick)
        self.undoButton.clicked.connect(self.undoClick)
        self.drawContourButton.clicked.connect(self.drawContourClick)
        self.kmeansButton.clicked.connect(self.chooseK)
        self.meanshiftButton.clicked.connect(self.chooseBandwidth)
        self.runKmeansButton.clicked.connect(self.runKmeans)
        self.runMeanshiftButton.clicked.connect(self.runMeanshift)
        self.regionGrowingButton.clicked.connect(self.regionGrowingClick)
        self.watershedButton.clicked.connect(self.watershedClick)
        self.grabcutButton.clicked.connect(self.grabcutClick)
        self.gmmButton.clicked.connect(self.gmmClick)


    @pyqtSlot()
    def loadClick(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Load Image',
                                                    'test_images/', '(*.jpg *.png *.jpeg *.tif)')
        if fname:
            self.loadOriImage(fname)
        else:
            print('Invalid Image')

    @pyqtSlot()
    def undoClick(self):
        self.segmented_image = self.restore_segmented_image
        self.displaySegImage()

    @pyqtSlot()
    def saveClick(self):
        fname, filter = QFileDialog.getSaveFileName(self, 'Save Image',
                                                    '.', '(*.jpg *.png *.jpeg *.tif)')
        if fname:
            cv2.imwrite(fname, self.segmented_image)
        else:
            print('Error')

    @pyqtSlot()
    def drawContourClick(self):
        self.segmentedImage.clear()

        self.restore_segmented_image = self.segmented_image.copy()
        self.segmented_image = draw_contours(self.restore_image, self.segmented_image)
        self.displaySegImage()

    @pyqtSlot()
    def chooseK(self):
        self.kmeansComboBox.show()
        self.runKmeansButton.show()

    @pyqtSlot()
    def chooseBandwidth(self):
        self.meanshiftComboBox.show()
        self.runMeanshiftButton.show()

    @pyqtSlot()
    def runKmeans(self):
        k = int(self.kmeansComboBox.currentText())
        self.kmeansClick(k)

    @pyqtSlot()
    def runMeanshift(self):
        bandwidth = int(self.meanshiftComboBox.currentText())
        self.meanshiftClick(bandwidth)

    @pyqtSlot()
    def kmeansClick(self, k):
        self.kmeansComboBox.hide()
        self.runKmeansButton.hide()

        self.segmentedImage.clear()
        self.progressBar.show()
        self.timer.start(100, self)
        _, _, data_points_scaled = transform_img_5_dim(self.image)

        if isinstance(k, int):
            K = Kmeans(k)
        # centroids, clusters, labels = K.cluster(data_points_scaled)
        centroids = K.rand_center(data_points_scaled)
        converge = False
        iteration = 0
        while not converge:
            old_centroids = np.copy(centroids)
            centroids, clusters, labels = K.update_centroids(data_points_scaled, old_centroids)
            print('iteration: ', iteration)
            print('centroids: ')
            print(centroids)
            converge = K.converged(old_centroids, centroids)
            iteration += 1
            self.progressBar.setValue(iteration * 3)
        print('number of iterations to converge: ', iteration)
        print(">>> final centroids")
        print(centroids)

        labels = labels.astype(np.int64)

        self.segmented_image = color_clusters(self.image, labels, data_points_scaled, centroids)

        self.timer.stop()
        self.progressBar.hide()
        self.displaySegImage()

    @pyqtSlot()
    def meanshiftClick(self, bandwidth):
        self.meanshiftComboBox.hide()
        self.runMeanshiftButton.hide()

        self.segmentedImage.clear()
        self.progressBar.show()
        self.timer.start(100, self)
        mf = Meanshift(self.image, bandwidth)

        cnt = 0
        for centroid in mf.centroids:
            iteration = 0
            current_centroid = centroid

            while True:
                neighbor_idxs = mf.nbrs.radius_neighbors([current_centroid], mf.radius, return_distance=False)[0]
                neighbor_to_centroid = mf.X[neighbor_idxs]

                new_centroid = np.mean(neighbor_to_centroid, 0)
                dist = euclidean_dist(new_centroid, current_centroid)

                print('iteration: ', iteration)
                print('dist: ', dist)

                if dist < THRESHOLD * mf.radius or iteration == 300:
                    mf.clusters[tuple(new_centroid)] = len(neighbor_idxs)
                    break
                else:
                    current_centroid = new_centroid

                iteration += 1

            cnt += 1
            self.progressBar.setValue(cnt * 10)

        self.segmented_image = mf.segment()

        self.timer.stop()
        self.progressBar.hide()
        self.displaySegImage()

    @pyqtSlot()
    def regionGrowingClick(self):
        self.segmentedImage.clear()
        self.timer.start(100, self)

        self.restore_image = self.image.copy()

        def capture_seed(event, x, y, flags, param):
            global seed_list
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(self.restore_image, (x, y), 5, (0, 0, 255), -1)
                seed_list.append((x, y))

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', capture_seed)

        while (1):
            cv2.imshow('image', self.restore_image)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('a'):
                print(seed_list)
        cv2.destroyAllWindows()

        self.gray_image = cv2.resize(self.gray_image, (500, 500))
        img = self.gray_image

        img = cv2.equalizeHist(img)

        rg = RegionGrowing(img, seed_list)
        rg.region_growing()
        self.segmented_image = rg.segmented_img

        cv2.imshow('segmentation', rg.segmented_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.timer.stop()
        self.progressBar.hide()
        self.displaySegImage()

    @pyqtSlot()
    def watershedClick(self):

        watershed = WatershedOpenCV(self.image)

        binary = watershed.apply_threshold()
        opening, background = watershed.get_background(binary)
        foreground = watershed.get_foreground(opening)
        boundary = watershed.get_boundary(background, foreground)

        watershed.get_markers(foreground, boundary)

        self.segmented_image = watershed.X

        self.timer.stop()
        self.progressBar.hide()
        self.displaySegImage()

    @pyqtSlot()
    def grabcutClick(self):
        cut = GrabcutOpenCV(self.image)
        cut.grabcut()
        self.segmented_image = cut.X
        self.timer.stop()
        self.progressBar.hide()
        self.displaySegImage()

    @pyqtSlot()
    def gmmClick(self):
        self.segmentedImage.clear()
        self.progressBar.show()
        x, y, image = transform_img_5_dim(self.image)

        gmm = GMM(image, 2)

        # EM Algorithm applied in GMM:
        for iteration in range(MAX_ITERATION):
            print('iteration:', iteration)
            self.progressBar.setValue(iteration)

            # E step
            gmm.update_q_function_of_c()

            # M step
            gmm.update_mu()
            gmm.update_sigma()
            gmm.update_pi()

            # KL Divergence
            gmm.update_gmm_distributions()
            gmm.kl_divergence()

        # Generate mask
        mask = np.argmax(gmm.q_function_of_c, axis=1)
        mask = mask.reshape(mask.shape[0], 1)

        # Separate background and foreground
        class1 = np.multiply(mask, image)
        class2 = image - class1
        class1 = class1[:, :3]
        class2 = class2[:, :3]
        class2 = class2.reshape((x, y, 3))
        class1 = class1.reshape((x, y, 3))

        cv2.imshow('gmm results', np.hstack((class1, class2)))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.timer.stop()
        self.progressBar.hide()
        self.displaySegImage()


    def loadOriImage(self, filename):
        self.image = cv2.imread(filename)
        self.image = cv2.resize(self.image, (500, 500))
        self.restore_image = self.image.copy()
        self.gray_image = cv2.imread(filename, 0)
        self.displayOriImage()

    def displayOriImage(self):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            # RGB
            if self.image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image,
                     self.image.shape[1],
                     self.image.shape[0],
                     self.image.strides[0],
                     qformat)
        # BGR -> RGB
        img = img.rgbSwapped()
        self.originalImage.setPixmap(QPixmap.fromImage(img))
        self.originalImage.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def displaySegImage(self):
        qformat = QImage.Format_Indexed8
        self.segmented_image = cv2.resize(self.segmented_image, (500, 500))

        if len(self.segmented_image.shape) == 3:
            # RGB
            if self.segmented_image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        seg_img = QImage(self.segmented_image,
                         self.segmented_image.shape[1],
                         self.segmented_image.shape[0],
                         self.segmented_image.strides[0],
                         qformat)
        # BGR -> RGB
        seg_img = seg_img.rgbSwapped()
        self.segmentedImage.setPixmap(QPixmap.fromImage(seg_img))
        self.segmentedImage.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
