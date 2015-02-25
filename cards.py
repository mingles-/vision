from count_symbols import CountSymbols
from featuriser import Featuriser

__author__ = 'Sam Davies'
import cv2
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

import operator


class CardClassifier(object):

    def __init__(self):
        self.train = []
        self.train_labels = []


    def get_objects_with_label(self, img, label):
        """
        Extract all the objects from an image and add each
        object's feature vector to the training data
        :param img: the image to extract from
        :param label: the label for all objects in this image
        """

        feature_vectors = Featuriser(img).feature_vectors

        for feature_vector in feature_vectors:
            self.train.append(feature_vector)
            self.train_labels.append(label)
            # print "{0} -- label {1}".format(feature_vector, label)
            # print "------------------"

    def classify_card(self, img):
        """
        Classify the image by matching each feature of the card to features
        from the test set, then voting on the most occurring card.
        :param img: the image to classify
        """
        feature_vectors = Featuriser(img).feature_vectors
        to_classify = np.array(feature_vectors).astype(np.float32)

        train = np.array(self.train).astype(np.float32)
        train_labels = np.array(self.train_labels).astype(np.float32)


        if to_classify.shape != (0, ):
            gnb = GaussianNB()
            gnb.fit(train, train_labels)
            results = gnb.predict(to_classify)

            suit = []

            for result in results:
                digit = self.convert_class_to_digit(result)
                print "class: {0} -- suit: {1}".format(result, result % 4)
                suit.append(result % 4)


            single_suit = stats.mode(suit, axis=None)[0]
            single_card_number = CountSymbols(img).symbol_count - 4.0

            single_class = self.convert_suit_and_num_to_card(single_card_number, single_suit)
            print "AVERAGE class: {0} --  suit: {1} --  digit: {2}"\
                .format(single_class[0], single_suit[0], single_card_number)
            return single_class, single_suit[0], single_card_number
        else:
            return -1, -1, -1

    @staticmethod
    def convert_class_to_digit(card_num):
        return ((int(card_num) - 1)/4) + 2

    @staticmethod
    def convert_suit_and_num_to_card(card_num, suit):
        card_num_bit = (card_num - 1) * 4
        mod_bit = ((suit - 1) % 4) + 1
        return card_num_bit - (4 - mod_bit)

    def add_training_images(self, labels):
        """
        Add all the cards in training set to the training data
        :param labels: a list of labels for the training cards
        """
        for x in range(1, len(labels)):
            img = cv2.imread('Images/ivr1415pract1data1/train{0}.jpg'.format(x))
            self.get_objects_with_label(img, labels[x-1])

    def classify_all_test_cards(self, labels):
        """
        Counts the number of cards in the test set which are classified correctly
        :param labels: the labels of the test set cards
        :return: the percent correct and the individual classes and numbers
        """
        count = 0
        single_suits = []
        single_nums = []

        for x in range(1, len(labels)+1):
            img = cv2.imread('Images/ivr1415pract1data2/test{0}.jpg'.format(x))
            label = labels[x-1]
            print("-----")
            print "Classifying card " + str(label)
            classification, single_suit, single_num = self.classify_card(img)
            single_suits.append(single_suit)
            single_nums.append(single_num)
            if classification != -1 and classification == label:
                count += 1
        return (100.0 * count / len(labels)), single_suits, single_nums

    @staticmethod
    def get_test_label(num):
        return (29 - (4 * (int(num)/4))) + (int(num) % 4)

if __name__ == "__main__":
    c = CardClassifier()
    training_labels = []
    testing_labels = []
    for i in range(0, 32):
        training_labels.append(i+1)
        testing_label = c.get_test_label(i)
        testing_labels.append(testing_label)
        print("training_label: " + str(i+1) + " testing_label: " + str(testing_label))


    c.add_training_images(training_labels)
    correctly_classified, single_suits, single_nums = c.classify_all_test_cards(testing_labels)
    print "-------------------------------"
    print "{0}% Correctly Classified".format(correctly_classified)
    print "-------------------------------"

    # confusion matrix stuff
    print map(int, single_suits)
    print map(int, single_nums)

    conf_training_numbers = []
    conf_training_suits = []
    for i in range(0, 32):
        conf_training_numbers.append(c.convert_class_to_digit(training_labels[i]))
        #conf_training_suits.append((training_labels[i]+1) % 4)

    print conf_training_numbers
    print conf_training_suits