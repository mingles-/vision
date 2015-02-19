__author__ = 'Sam Davies'

import unittest
import cv2
from cards import CardClassifier


class CardClassifierTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_can_get_label(self):
        image = cv2.imread('Images/ivr1415pract1data1/train1.jpg', 0)
        card_classifier = CardClassifier()
        card_classifier.get_objects_with_label(image, 1)
        label = card_classifier.train_labels[0]
        self.assertEquals(label, 1)


if __name__ == "__main__":
    unittest.main()