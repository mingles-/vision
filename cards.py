__author__ = 'Sam Davies'
import cv2
import numpy as np
from matplotlib import pyplot as plt


class CardClassifier(object):

    def __init__(self):
        self.train = []
        self.train_labels = []

    @staticmethod
    def get_feature_vector(cnt, img, gray_img):
        """
        Extract the feature vector of the given contour
        :param cnt: the contour to extract from
        :return: the feature vector extracted
        """
        moments = cv2.moments(cnt)

        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area

        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w*h
        extent = float(area)/rect_area

        mask = np.zeros(gray_img.shape, np.uint8)
        mean_val = cv2.mean(img, mask=mask)
        total_val = mean_val[0] + mean_val[1] + mean_val[2]
        mean_red = mean_val[2]/(0.001+float(total_val))

        feature_vector = [[area], [perimeter], [solidity], [extent], [mean_red]]
        return feature_vector

    def get_objects_with_label(self, img, label):
        """
        Extract all the objects from an image and add each
        object's feature vector to the training data
        :param img: the image to extract from
        :param label: the label for all objects in this image
        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        contours_sorted = self.img_to_contours(gray_img)
        relevant_contours = self.find_relevant_contours(contours_sorted)

        feature_vectors = [self.get_feature_vector(cnt, img, gray_img) for cnt in relevant_contours]

        for feature_vector in feature_vectors:
            self.train.append(feature_vector)
            self.train_labels.append(label)

    @staticmethod
    def img_to_contours(gray_img):
        """
        Get a list of all contours in this image sorted by area descending
        :param img: the image to get contours from
        :return: contours sorted by area descending
        """
        # turn the image into binary (black and white, no grey)
        blur = cv2.GaussianBlur(gray_img, (1, 1), 1000)
        ret, thresh = cv2.threshold(blur, 129, 255, cv2.THRESH_BINARY)
        # find all the contours in the image, all areas of joint white/black
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # figure out the largest contour areas
        return sorted(contours, key=cv2.contourArea, reverse=True)

    @staticmethod
    def find_relevant_contours(contours_sorted):
        """
        Using a heuristic, find the meaningful contours from a list of contours
        :param contours_sorted: the full list of contours
        :return: only the meaningful contours
        """
        # draw all the contours who's area is between 2 thresholds
        min_area = 500
        max_area = cv2.contourArea(contours_sorted[0])/25

        relevant_contours = []
        # print "max area {0}".format(max_area)
        for cnt in contours_sorted[1:]:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                relevant_contours.append(cnt)
            else:
                if min_area > area:
                    break

        return relevant_contours

    @staticmethod
    def count_red_pixels(img, threshold=0.5):
        """
        counts the number of red pixels in an image.
        Red being where the ratio of red to all colors is greater than a threshold.
        The count is weighted so to not unbalance other elements of the feature vector
        :param img: the image
        :param threshold:
        :return: a weighted count of the number of red pixels
        """
        red_count = 0
        for x in range(0, len(img)):
            for y in range(0, len(img[x])):
                b = img.item(x, y, 0)
                g = img.item(x, y, 1)
                r = img.item(x, y, 2)
                total = r + b + g
                if r/(0.1+float(total)) > threshold:
                    red_count += 1
        return 400.0 * red_count / (len(img) * len(img[0]))

    def classify_card(self, img):
        """
        Classify the image by matching each feature of the card to features
        from the test set, then voting on the most occurring card.
        :param img: the image to classify
        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        contours_sorted = self.img_to_contours(gray_img)
        relevant_contours = self.find_relevant_contours(contours_sorted)

        feature_vectors = [self.get_feature_vector(cnt, img, gray_img) for cnt in relevant_contours]
        print feature_vectors
        to_classify = np.array(feature_vectors).astype(np.float32)

        train = np.array(self.train).astype(np.float32)
        train_labels = np.array(self.train_labels).astype(np.float32)

        knn = cv2.KNearest()
        knn.train(train, train_labels)
        ret, result, neighbours, dist = knn.find_nearest(to_classify, 1)
        print result

    def add_training_images(self, lbls):
        """
        Add all the cards in training set to the training data
        :param lbls: a list of labels for the training cards
        """
        for x in range(1, len(lbls)):
            image = cv2.imread('Images/ivr1415pract1data1/train{0}.jpg'.format(x))
            self.get_objects_with_label(image, lbls[x])


if __name__ == "__main__":
    card_classifier = CardClassifier()
    labels = []
    for i in range(1, 33):
        labels.append(i)
    card_classifier.add_training_images(labels)
    image = cv2.imread('Images/ivr1415pract1data2/test1.jpg')
    card_classifier.classify_card(image)
