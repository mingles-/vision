__author__ = 'Sam Davies'
import cv2
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
import operator


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
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        mean_val = cv2.mean(img, mask=mask)
        total_val = mean_val[0] + mean_val[1] + mean_val[2]
        mean_red = mean_val[2]/(0.001+float(total_val))

        # feature_vector = [area, perimeter, solidity, extent, mean_red]
        feature_vector = [[area], [perimeter], [solidity], [extent], [mean_red]]
        # print feature_vector
        # print "------------------"
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

    def img_to_contours(self, gray_img):
        """
        Get a list of all contours in this image sorted by area descending
        :param gray_img: the image to get contours from
        :return: contours sorted by area descending
        """
        # turn the image into binary (black and white, no grey)
        blur = cv2.GaussianBlur(gray_img, (1, 1), 1000)
        ret, thresh = cv2.threshold(blur, 129, 255, cv2.THRESH_BINARY)
        # find all the contours in the image, all areas of joint white/black
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        card_cnt_index, card_cnt = self.max_contour_area_index(contours)

        # removed all non childs of the card
        good_cnts = [card_cnt]
        for n in range(0, len(contours)):
            # make sure that the contours parent is the card
            if hierarchy[0][n][3] == card_cnt_index:
                good_cnts.append(contours[n])

        # figure out the largest contour areas
        return sorted(good_cnts, key=cv2.contourArea, reverse=True)

    @staticmethod
    def max_contour_area_index(contours):
        max_area = 0
        max_area_index = 0
        for b in range(0, len(contours)):
            if cv2.contourArea(contours[b]) > max_area:
                max_area = cv2.contourArea(contours[b])
                max_area_index = b
        return max_area_index, contours[max_area_index]

    @staticmethod
    def find_relevant_contours(contours_sorted):
        """
        Using a heuristic, find the meaningful contours from a list of contours
        :param contours_sorted: the full list of contours
        :return: only the meaningful contours
        """
        if contours_sorted:
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
        else:
            return []

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
        to_classify = np.array(feature_vectors).astype(np.float32)

        train = np.array(self.train).astype(np.float32)
        train_labels = np.array(self.train_labels).astype(np.float32)

        if to_classify.shape != (0, ):
            # gnb = GaussianNB()
            # gnb.fit(train, train_labels)
            # result = gnb.predict(to_classify)
            #
            knn = cv2.KNearest()
            knn.train(train, train_labels)
            ret, result, neighbours, dist = knn.find_nearest(to_classify, 1)
            # return the most occurring card
            return stats.mode(result, axis=None)[0]
        else:
            return -1

    def add_training_images(self, labels):
        """
        Add all the cards in training set to the training data
        :param labels: a list of labels for the training cards
        """
        for x in range(1, len(labels)):
            img = cv2.imread('Images/ivr1415pract1data1/train{0}.jpg'.format(x))
            self.get_objects_with_label(img, labels[x])

    def classify_all_test_cards(self, labels):
        """
        Counts the number of cards in the test set which are classified correctly
        :param labels: the labels of the test set cards
        :return: the percent correct
        """
        count = 0

        for x in range(1, len(labels)):
            img = cv2.imread('Images/ivr1415pract1data2/test{0}.jpg'.format(x))
            classification = self.classify_card(img)
            if classification != 1 and classification == labels[x]:
                count += 1
        return 100.0 * count / len(labels)


if __name__ == "__main__":
    c = CardClassifier()
    training_labels = []
    testing_labels = []
    for i in range(1, 33):
        training_labels.append(i)
        testing_labels.append(33-i)

    c.add_training_images(training_labels)
    correctly_classified = c.classify_all_test_cards(testing_labels)
    print "-------------------------------"
    print "{0}% Correctly Classified".format(correctly_classified)
    print "-------------------------------"
