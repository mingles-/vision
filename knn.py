__author__ = 'Sam Davies'

import numpy as np
import cv2
from matplotlib import pyplot as plt


def knn_classifier(k):
    # Now we prepare train_data and test_data.
    train = np.array([[1, 2], [5, 3]]).astype(np.float32)
    test = np.array([[1, 2], [5, 7]]).astype(np.float32)

    # Create labels for train and test data
    train_labels = np.array([[1], [2]]).astype(np.float32)
    test_labels = np.array([[1], [2]]).astype(np.float32)

    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv2.KNearest()
    knn.train(train, train_labels)
    ret, result, neighbours, dist = knn.find_nearest(test, k)

    # Now we check the accuracy of classification
    # For that, compare the result with test_labels and check which are wrong
    matches = result == test_labels
    correct = np.count_nonzero(matches)
    accuracy = correct*100.0/result.size
    print "KNN - {0}% correct".format(accuracy)
    return result


if __name__ == "__main__":
    knn_classifier(3)