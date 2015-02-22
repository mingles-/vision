__author__ = 'Sam Davies'
import cv2
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
import operator


class CardClassifier(object):

    def __init__(self):
        self.train = []
        self.train_labels = []

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
        feature_vector = [compactness, mean_val[2]]
        return feature_vector

    @staticmethod
    def get_y_pos(cnt):
        moments = cv2.moments(cnt)
        return int(moments['m01']/moments['m00'])

    def get_objects_with_label(self, img, label):
        """
        Extract all the objects from an image and add each
        object's feature vector to the training data
        :param img: the image to extract from
        :param label: the label for all objects in this image
        """
        img_vector = self.get_image_vector(img)

        # print "{0} -- label {1}".format(img_vector, label)
        # print "------------------"
        self.train.append(img_vector)
        self.train_labels.append(label)

    def get_image_vector(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        contours_sorted = self.img_to_contours(gray_img)
        relevant_contours = self.find_relevant_contours(contours_sorted)

        relevant_contours_sorted_by_y = sorted(relevant_contours, key=self.get_y_pos, reverse=False)

        feature_vectors = [self.get_feature_vector(cnt, img, gray_img) for cnt in relevant_contours_sorted_by_y]

        img_vector = []

        for feature_vector in feature_vectors:
            for attribute in feature_vector:
                img_vector.append(attribute)

        return img_vector

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
        return sorted(good_cnts, key=cv2.contourArea, reverse=True)

    @staticmethod
    def max_contour_area_index(contours):
        max_area = 0
        max_area_index = 0
        for b in range(0, len(contours)):
            if cv2.contourArea(contours[b]) > max_area:
                max_area = cv2.contourArea(contours[b])
                max_area_index = b
        return max_area_index, contours[max_area_index]

    @staticmethod
    def find_relevant_contours(contours_sorted):
        """
        Using a heuristic, find the meaningful contours from a list of contours
        :param contours_sorted: the full list of contours
        :return: only the meaningful contours
        """
        out_contours = []

        if contours_sorted:
            # draw all the contours who's area is between 2 thresholds
            max_area = cv2.contourArea(contours_sorted[0])/25

            relevant_contours = []
            # print "max area {0}".format(max_area)
            for cnt in contours_sorted[1:]:
                if len(relevant_contours) == 4:
                    break
                area = cv2.contourArea(cnt)
                if area < max_area:
                    relevant_contours.append(cnt)


            out_contours = relevant_contours

        if len(out_contours) > 4:
            return out_contours[:4]
        else:
            if len(out_contours) < 4:
                assert False
            else:
                return out_contours
    @staticmethod
    def count_red_pixels(img, threshold=0.5):
        """
        counts the number of red pixels in an image.
        Red being where the ratio of red to all colors is greater than a threshold.
        The count is weighted so to not unbalance other elements of the feature vector
        :param img: the image
        :param threshold:
        :return: a weighted count of the number of red pixels
        """
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
        """
        Classify the image by matching each feature of the card to features
        from the test set, then voting on the most occurring card.
        :param img: the image to classify
        """
        to_classify = self.get_image_vector(img)
        to_classify = np.array(to_classify).astype(np.float32)

        train = np.array(self.train).astype(np.float32)
        train_labels = np.array(self.train_labels).astype(np.float32)

        if to_classify.shape != (0, ):
            gnb = GaussianNB()
            gnb.fit(train, train_labels)
            # print to_classify
            results = gnb.predict(to_classify)
            #
            #knn = cv2.KNearest()
            #knn.train(train, train_labels)
            #ret, results, neighbours, dist = knn.find_nearest(to_classify, 1)
            # print("-----")
            return results
            # return the most occurring card
            """suit = []
            card_number = []


            for result in results:
                digit = self.convert_class_to_digit(result)
                print "class: {0} -- suit: {1} -- digit: {2}".format(result, result % 4, digit)
                suit.append(result % 4)
                card_number.append(digit)

            single_suit = stats.mode(suit, axis=None)[0]
            single_card_number = stats.mode(card_number, axis=None)[0]

            single_class = self.convert_suit_and_num_to_card(single_card_number, single_suit)
            print "AVERAGE class: {0} --  suit: {1} --  digit: {2}"\
                .format(single_class[0], single_suit[0], single_card_number[0])
            return single_class"""
            #print("Mode: "+ str((stats.mode(results, axis=None)[0]) % 4))
            #return stats.mode(results, axis=None)[0]
        else:
            return -1

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
        for x in range(1, len(labels)+1):
            img = cv2.imread('Images/ivr1415pract1data1/train{0}.jpg'.format(x))
            self.get_objects_with_label(img, labels[x-1])

    def classify_all_test_cards(self, labels):
        """
        Counts the number of cards in the test set which are classified correctly
        :param labels: the labels of the test set cards
        :return: the percent correct
        """
        count = 0

        for x in range(1, len(labels)+1):
            img = cv2.imread('Images/ivr1415pract1data2/test{0}.jpg'.format(x))
            classification = self.classify_card(img)
            if classification != -1 and classification == labels[x-1]:
                count += 1
        return 100.0 * count / len(labels)


if __name__ == "__main__":
    c = CardClassifier()
    training_labels = []
    testing_labels = []
    for i in range(0, 32):
        training_labels.append(i+1)
        print (i+1)
        new_num = (29 - (4 * (i/4))) + (i % 4)
        #print new_num
        testing_labels.append(new_num)

    c.add_training_images(training_labels)
    correctly_classified = c.classify_all_test_cards(testing_labels)
    print "-------------------------------"
    print "{0}% Correctly Classified".format(correctly_classified)
    print "-------------------------------"
