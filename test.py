from count_symbols import CountSymbols
from matplotlib import pyplot as plt
__author__ = 'Sam Davies'

import unittest
import cv2
from cards import CardClassifier
import numpy as np
from featuriser import Featuriser



class CardClassifierTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_can_get_label(self):
        image = cv2.imread('Images/ivr1415pract1data1/train1.jpg')
        c = CardClassifier()
        c.get_objects_with_label(image, 1)
        vectors = c.train[0]
        label = c.train_labels[0]
        self.assertEquals(label, 1)
        #print vectors.shape
        #self.assertEquals(vectors.shape, (8, 1))

    def test_convert_class_to_digit(self):
        self.assertEquals(CardClassifier.convert_class_to_digit(5), 3)
        self.assertEquals(CardClassifier.convert_class_to_digit(32), 9)
        self.assertEquals(CardClassifier.convert_class_to_digit(1), 2)
        self.assertEquals(CardClassifier.convert_class_to_digit(23), 7)

    def test_convert_suit_and_num_to_card(self):
        self.assertEquals(CardClassifier.convert_suit_and_num_to_card(2, 1), 1)
        self.assertEquals(CardClassifier.convert_suit_and_num_to_card(3, 2), 6)
        self.assertEquals(CardClassifier.convert_suit_and_num_to_card(4, 0), 12)
        self.assertEquals(CardClassifier.convert_suit_and_num_to_card(9, 0), 32)


class CountSymbolsTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_count_symobols(self):
        card_1 = cv2.imread('Images/ivr1415pract1data1/train1.jpg')
        f1 = CountSymbols(card_1)
        # f1.draw_contours(card_1, f1.symbol_contours)
        self.assertEquals(f1.symbol_count, 2+4)

        card_32 = cv2.imread('Images/ivr1415pract1data1/train32.jpg')
        f1 = CountSymbols(card_32)
        # f1.draw_contours(card_32, f1.symbol_contours)
        self.assertEquals(f1.symbol_count, 9+4)

        card_18 = cv2.imread('Images/ivr1415pract1data1/train18.jpg')
        f1 = CountSymbols(card_18)
        # f1.draw_contours(card_18, f1.symbol_contours)
        self.assertEquals(f1.symbol_count, 6+4)

        card_5 = cv2.imread('Images/ivr1415pract1data1/train5.jpg')
        f1 = CountSymbols(card_5)
        # f1.draw_contours(card_5, f1.symbol_contours)
        self.assertEquals(f1.symbol_count, 3+4)

        card_t30 = cv2.imread('Images/ivr1415pract1data2/test30.jpg')
        f1 = CountSymbols(card_t30)
        f1.draw_contours(card_5, f1.symbol_contours)
        self.assertEquals(f1.symbol_count, 9+4)


if __name__ == "__main__":
    unittest.main()