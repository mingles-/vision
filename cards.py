__author__ = 'Sam Davies'
import cv2
import numpy as np
from matplotlib import pyplot as plt


class CardClassifier(object):

    def __init__(self):
        self.train = []
        self.train_labels = []

    @staticmethod
    def get_feature_vector(cnt):
        """
        Extract the feature vector of the given contour
        :param cnt: the contour to extract from
        :return: the feature vector extracted
        """
        feature_vector = []
        moments = cv2.moments(cnt)
        feature_vector.append(cv2.HuMoments(moments))
        return feature_vector

    def get_objects_with_label(self, img, label):
        """
        Extract all the objects from an image and add each
        object's feature vector to the training data
        :param img: the image to extract from
        :param label: the label for all objects in this image
        """
        red_count = self.count_red_pixels(img)

        contours_sorted = self.img_to_contours(img)
        relevant_contours = self.find_relevant_contours(contours_sorted)

        feature_vectors = [self.get_feature_vector(cnt) for cnt in relevant_contours][0]
        print np.array(feature_vectors).shape

        for feature_vector in feature_vectors:
            feature_vector = np.append(feature_vector, [[float(red_count)]], axis=0)
            self.train.append(feature_vector)
            self.train_labels.append(label)

    @staticmethod
    def img_to_contours(img):
        """
        Get a list of all contours in this image sorted by area descending
        :param img: the image to get contours from
        :return: contours sorted by area descending
        """
        # turn the image into binary (black and white, no grey)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (1, 1), 1000)
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

        contours_sorted = self.img_to_contours(img)

        relevant_contours = self.find_relevant_contours(contours_sorted)
        feature_vectors = [self.get_feature_vector(cnt)[0] for cnt in relevant_contours]
        print np.array(feature_vectors).shape

        red_count = self.count_red_pixels(img)
        print str(red_count) + " new"
        for j in range(0, len(feature_vectors)):
            feature_vectors[j] = np.append(feature_vectors[j], [[float(red_count)]], axis=0)

        contour_hu_moments = np.array(feature_vectors).astype(np.float32)

        train = np.array(self.train).astype(np.float32)
        train_labels = np.array(self.train_labels).astype(np.float32)


        # Initiate kNN, train the data, then test it with test data for k=1
        knn = cv2.KNearest()
        knn.train(train, train_labels)
        ret, result, neighbours, dist = knn.find_nearest(contour_hu_moments, 1)
        print result
        # Now we check the accuracy of classification
        # For that, compare the result with test_labels and check which are wrong

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
