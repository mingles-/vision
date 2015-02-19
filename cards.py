__author__ = 'Sam Davies'
import cv2
import numpy as np
from matplotlib import pyplot as plt


class CardClassifier(object):

    def __init__(self):
        self.train = []
        self.train_labels = []

    def get_objects_with_label(self, img, label):
        contours_sorted = self.img_to_contours(img)

        relevant_contours = self.find_relevant_contours(contours_sorted)
        for cnt in relevant_contours:
            self.train.append(cnt)
            self.train_labels.append(label)

    @staticmethod
    def img_to_contours(img):
        # turn the image into binary (black and white, no grey)
        blur = cv2.GaussianBlur(img, (1, 1), 1000)
        ret, thresh = cv2.threshold(blur, 129, 255, cv2.THRESH_BINARY)
        # find all the contours in the image, all areas of joint white/black
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # figure out the largest contour areas
        return sorted(contours, key=cv2.contourArea, reverse=True)

    @staticmethod
    def find_relevant_contours(contours_sorted):
        # draw all the contours who's area is between 2 thresholds
        min_area = 500
        max_area = cv2.contourArea(contours_sorted[0])/25

        relevant_contours = []
        # print "max area {0}".format(max_area)
        for cnt in contours_sorted[1:]:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                # print "object area {0}".format(area)
                # cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
                moments = cv2.moments(cnt)
                relevant_contours.append(cv2.HuMoments(moments))
            else:
                if min_area > area:
                    break

        return relevant_contours

    def classify_card(self, img):

        contours_sorted = self.img_to_contours(img)
        relevant_contours = self.find_relevant_contours(contours_sorted)

        relevant_contours = np.array(relevant_contours).astype(np.float32)

        train = np.array(self.train).astype(np.float32)
        train_labels = np.array(self.train_labels).astype(np.float32)


        # Initiate kNN, train the data, then test it with test data for k=1
        knn = cv2.KNearest()
        knn.train(train, train_labels)
        ret, result, neighbours, dist = knn.find_nearest(relevant_contours, 1)
        print result
        # Now we check the accuracy of classification
        # For that, compare the result with test_labels and check which are wrong
        """
        matches = result == test_labels
        correct = np.count_nonzero(matches)
        accuracy = correct*100.0/result.size
        print "KNN - {0}% correct".format(accuracy)
        return result

        # draw a rotated box around the biggest contour
        rect = cv2.minAreaRect(contours_sorted[0])
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)"""

    def add_training_images(self, lbls):
        for x in range(1, len(lbls)):
            image = cv2.imread('Images/ivr1415pract1data1/train{0}.jpg'.format(x), 0)
            self.get_objects_with_label(image, lbls[x])


if __name__ == "__main__":
    card_classifier = CardClassifier()
    labels = []
    for i in range(1, 33):
        labels.append(i)
    card_classifier.add_training_images(labels)
    image = cv2.imread('Images/ivr1415pract1data2/test1.jpg', 0)
    card_classifier.classify_card(image)
