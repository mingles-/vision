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
        self.check_train_card(1, False)
        self.check_train_card(32, False)
        # self.check_train_card(18, False)

        self.check_train_card(27, True)

    def check_train_card(self, card_num, to_draw):
        self.check_card('Images/ivr1415pract1data1/train', card_num, card_num, to_draw)

    def check_test_card(self, card_num, to_draw):
        file_num = card_num
        card_num = CardClassifier.get_test_label(card_num)
        self.check_card('Images/ivr1415pract1data2/test', card_num, file_num, to_draw)

    def check_card(self, base, card_num, file_num, to_draw):
        card = cv2.imread(base + str(file_num) + '.jpg')
        f1 = CountSymbols(card)
        if to_draw:
            f1.draw_contours(card, f1.symbol_contours)
        print f1.symbol_count
        self.assertEquals(f1.symbol_count, ((int(card_num)-1)/4)+6)


if __name__ == "__main__":
    unittest.main()