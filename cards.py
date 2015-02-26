from count_symbols import CountSymbols
from featuriser import Featuriser

__author__ = 'Sam Davies and Mingles'
import cv2
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


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
            cs = CountSymbols(img)
            single_card_number = cs.symbol_count - 4.0


            if single_suit % 4 == 1:
                suite = "Spades"
            elif single_suit % 4 == 2:
                suite = "Hearts"
            elif single_suit % 4 == 3:
                suite = "Clubs"
            elif single_suit % 4 == 0:
                suite = "Diamonds"

            # CARD DISPLAY
            #cs.draw_contours(img, cs.symbol_contours + [cs.box], str(int(single_card_number)) + " of " + str(suite))

            single_class = self.convert_suit_and_num_to_card(single_card_number, single_suit)
            print "AVERAGE class: {0} --  suit: {1} --  digit: {2}"\
                .format(single_class[0], single_suit[0], single_card_number)
            return single_class, single_suit[0], single_card_number
        else:
            return -1, -1, -1

    @staticmethod
    def convert_class_to_digit(card_num):
        """
        Given a card number, find the corresponding digit
        :param card_num: card number
        :return: digit
        """
        return ((int(card_num) - 1)/4) + 2

    @staticmethod
    def convert_suit_and_num_to_card(digit, suit):
        """
        Given a suit and a digit, find the card number
        :param digit: digit
        :param suit: suit
        :return: the card number
        """
        card_num_bit = (digit - 1) * 4
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
        """
        given the card number in the test set, find the corresponding label in the training set
        :param num: the number of the card in test set
        :return: the label from the training set
        """
        return (29 - (4 * (int(num)/4))) + (int(num) % 4)

    def get_confusion_matrices(self, test_labels, suits_pred, nums_pred):
        """
        display the confusion matrices for the suits and the digits
        :param test_labels: the real labels for the images
        :param suits_pred: the predicted suits
        :param nums_pred: the predicted digits
        """
        nums_test = []
        suits_test = []

        for i in range(0, 32):
            nums_test.append(c.convert_class_to_digit(test_labels[i]))
            suits_test.append((test_labels[i]) % 4)

        print "----------"
        print "SUIT CONFUSION MATRIX"
        suits_test, suits_pred, bad_suits = self.removed_bad_classes(suits_test, suits_pred)
        print "Removed cards - {0}".format(bad_suits)
        conf_suit = confusion_matrix(suits_test, suits_pred)
        print conf_suit

        print "----------"
        print "DIGIT CONFUSION MATRIX"
        nums_test, nums_pred, bad_num = self.removed_bad_classes(nums_test, nums_pred)
        print "Removed cards - {0}".format(bad_num)
        conf_num = confusion_matrix(nums_test, nums_pred)
        print conf_num

    def removed_bad_classes(self, test, pred):
        """
        Remove the cards from for which now feature vectors were found
        :param test: the real labels
        :param pred: the predicted labels
        :return: the edited test adn pred along with the bad cards
        """
        bad = []
        new_test = []
        new_pred = []
        for p in range(0, len(pred)):
            if pred[p] < 0:
                bad.append(p)
            else:
                new_pred.append(int(pred[p]))
                new_test.append(test[p])
        return new_test, new_pred, bad


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
    correctly_classified, suits_pred, nums_pred = c.classify_all_test_cards(testing_labels)
    print "-------------------------------"
    print "{0}% Correctly Classified".format(correctly_classified)
    print "-------------------------------"

    c.get_confusion_matrices(testing_labels, suits_pred, nums_pred)