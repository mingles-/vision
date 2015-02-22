__author__ = 'Sam Davies'

import unittest
import cv2
from cards import CardClassifier
import numpy as np


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

    def test_red_count(self):
        black_card = cv2.imread('Images/ivr1415pract1data1/train1.jpg')
        red_card = cv2.imread('Images/ivr1415pract1data1/train2.jpg')
        c = CardClassifier()
        black_count = c.count_red_pixels(black_card)
        red_count = c.count_red_pixels(red_card)
        self.assertTrue(black_count < red_count)

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



if __name__ == "__main__":
    unittest.main()