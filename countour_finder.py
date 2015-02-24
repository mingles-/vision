import cv2
from matplotlib import pyplot as plt

__author__ = 'Sam Davies and Mingles'


class ContourFinder(object):

    def __init__(self, img):
        img = cv2.medianBlur(img, 5)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.grey_image = gray_img

        contours, hierarchy = self.image_to_contours()

        card_index, card_contour = self.find_card_contour(contours)

        # self.draw_contours(img, [card_contour])
        self.min_area = 0.07 * cv2.contourArea(card_contour)
        self.max_area = 0.5 * cv2.contourArea(card_contour)
        # print "card area " + str(cv2.contourArea(card_contour))
        # print "min " + str(self.min_area)
        # print "max " + str(self.max_area)

        # inner_index, inner_contour = self.find_inner_contour(con# tours)

        good_contours = self.remove_non_child(0, contours, hierarchy)
        good_contours = self.remove_bad_contours(good_contours)
        self.good_contours = sorted(good_contours, key=cv2.contourArea, reverse=True)
        # print "num good contours: " + str(len(good_contours))

        # self.draw_contours(img, good_contours)
        self.symbol_contours = self.find_symbol_contours(self.good_contours)

    def find_card_contour(self, contours):
        image_index, image_contour = self.max_contour_area_index(contours)
        return self.max_contour_area_index(contours, excluding=[image_index])

    def image_to_contours(self):
        # turn the image into binary (black and white, no grey)
        blur = cv2.medianBlur(self.grey_image, 5)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # find all the contours in the image, all areas of joint white/black
        return cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    @staticmethod
    def remove_non_child(parent_index, contours, hierarchy):
        # removed all non childs of the card
        good_cnts = []
        for n in range(0, len(contours)):
            # make sure that the contours parent is the card
            if hierarchy[0][n][3] == parent_index:
                good_cnts.append(contours[n])
        return good_cnts

    def find_symbol_contours(self, contours):
        symbol_contours = []

        for cnt in contours:
            # print "symbol area of " + str(cv2.contourArea(cnt))
            if self.max_area > cv2.contourArea(cnt):
                if cv2.contourArea(cnt) > self.min_area:
                    symbol_contours.append(cnt)
                else:
                    break
        return symbol_contours

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
    def draw_contours(card, contours):
        for cnt in contours:
            cv2.drawContours(card, [cnt], 0, (0, 255, 0), 3)

        plt.imshow(card)
        plt.title("draw_contours")
        plt.show()

    def remove_bad_contours(self, contours):

        good_contours = []

        # remove stretched shapes
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            width = rect[1][0]
            height = rect[1][1]
            if 3 > (width/height) > (1.0/3):
                good_contours.append(cnt)
        return good_contours
