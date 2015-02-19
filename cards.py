__author__ = 'Sam Davies'
import cv2
import numpy as np
from matplotlib import pyplot as plt


class CardClassifier(object):

    def __init__(self):
        self.train = []
        self.train_labels = []

    def get_objects_with_label(self, img, label):
        # turn the image into binary (black and white, no grey)
        blur = cv2.GaussianBlur(img, (1, 1), 1000)
        ret, thresh = cv2.threshold(blur, 129, 255, cv2.THRESH_BINARY)
        # find all the contours in the image, all areas of joint white/black
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # figure out the largest contour areas
        card_contour = sorted(contours, key=cv2.contourArea, reverse=True)

        # draw a rotated box around the biggest contour
        rect = cv2.minAreaRect(card_contour[0])
        box = cv2.cv.BoxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

        # draw all the contours who's area is between 2 thresholds
        min_area = 500
        max_area = cv2.contourArea(card_contour[0])/25
        # print "max area {0}".format(max_area)
        for cnt in card_contour[1:]:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                # print "object area {0}".format(area)
                # cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
                moments = cv2.moments(cnt)
                self.train.append(cv2.HuMoments(moments))
                self.train_labels.append(label)
            else:
                if min_area > area:
                    break

    def add_training_images(self, labels):

        for i in range(1, len(labels)):
            image = cv2.imread('Images/ivr1415pract1data1/train{0}.jpg'.format(i), 0)
            self.get_objects_with_label(image, labels[i])


if __name__ == "__main__":
    card_classifier = CardClassifier()
    labels = []
    for i in range(1, 33):
        labels.append(i)
    card_classifier.add_training_images(labels)
