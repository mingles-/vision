__author__ = 'Sam Davies'
import cv2
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
import operator


class Featuriser(object):

    def __init__(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        contours_sorted, _, _ = self.img_to_contours(gray_img)

        min_area = 500
        max_area = cv2.contourArea(contours_sorted[0])/25
        relevant_contours = self.find_relevant_contours(contours_sorted, min_area, max_area)

        self.feature_vectors = [self.get_feature_vector(cnt, img, gray_img) for cnt in relevant_contours]

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

        hu_moment = cv2.HuMoments(moments)
        #print hu_moment


        compactness = perimeter * perimeter / (4 * np.pi * area)
        feature_vector = [compactness, extent, mean_val[2]]
        return feature_vector

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
        return sorted(good_cnts, key=cv2.contourArea, reverse=True), contours, hierarchy

    @staticmethod
    def max_contour_area_index(contours, excluding=[]):
        max_area = 0
        max_area_index = 0
        for b in range(0, len(contours)):
            if cv2.contourArea(contours[b]) > max_area and b not in excluding:
                max_area = cv2.contourArea(contours[b])
                max_area_index = b
        return max_area_index, contours[max_area_index]

    @staticmethod
    def find_relevant_contours(contours_sorted, min_area, max_area):
        """
        Using a heuristic, find the meaningful contours from a list of contours
        :param contours_sorted: the full list of contours
        :return: only the meaningful contours
        """
        if contours_sorted:
            # draw all the contours who's area is between 2 thresholds

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
