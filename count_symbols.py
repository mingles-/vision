from countour_finder import ContourFinder
from matplotlib import pyplot as plt

__author__ = 'Sam Davies'
import cv2


class CountSymbols(ContourFinder):

    def __init__(self, img):
        super(CountSymbols, self).__init__(img)

        self.symbol_count = len(self.symbol_contours)
        print "------"

