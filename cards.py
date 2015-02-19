__author__ = 'Sam Davies'
import cv2
import numpy as np
from matplotlib import pyplot as plt


class CardClassifier(object):

    def __init__(self):
        self.train = []
        self.train_labels = []

    def get_objects_with_label(self, img, label):
        red_count = self.count_red_pixels(img)
        print red_count

        contours_sorted = self.img_to_contours(img)
        contour_hu_moments = self.find_relevant_contour_hu_moments(contours_sorted)

        for hu_moments in contour_hu_moments:
            hu_moments = np.append(hu_moments, [[float(red_count)]], axis=0)
            self.train.append(hu_moments)
            self.train_labels.append(label)

    @staticmethod
    def img_to_contours(img):
        # turn the image into binary (black and white, no grey)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (1, 1), 1000)
        ret, thresh = cv2.threshold(blur, 129, 255, cv2.THRESH_BINARY)
        # find all the contours in the image, all areas of joint white/black
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # figure out the largest contour areas
        return sorted(contours, key=cv2.contourArea, reverse=True)

    @staticmethod
    def find_relevant_contour_hu_moments(contours_sorted):
        # draw all the contours who's area is between 2 thresholds
        min_area = 500
        max_area = cv2.contourArea(contours_sorted[0])/25

        hu_moments = []
        # print "max area {0}".format(max_area)
        for cnt in contours_sorted[1:]:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                # print "object area {0}".format(area)
                # cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
                moments = cv2.moments(cnt)
                hu_moments.append(cv2.HuMoments(moments))
            else:
                if min_area > area:
                    break

        return hu_moments

    @staticmethod
    def count_red_pixels(img, threshold=0.5):
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

        contours_sorted = self.img_to_contours(img)
        contour_hu_moments = self.find_relevant_contour_hu_moments(contours_sorted)

        red_count = self.count_red_pixels(img)
        print str(red_count) + " new"
        for j in range(0, len(contour_hu_moments)):
            contour_hu_moments[j] = np.append(contour_hu_moments[j], [[float(red_count)]], axis=0)

        contour_hu_moments = np.array(contour_hu_moments).astype(np.float32)

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
        for x in range(1, len(lbls)):
            image = cv2.imread('Images/ivr1415pract1data1/train{0}.jpg'.format(x))
            self.get_objects_with_label(image, lbls[x])


if __name__ == "__main__":
    card_classifier = CardClassifier()
    labels = []
    for i in range(1, 33):
        labels.append(i)
    card_classifier.add_training_images(labels)
    image = cv2.imread('Images/ivr1415pract1data2/test19.jpg')
    card_classifier.classify_card(image)
