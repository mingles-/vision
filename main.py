import cv2

import numpy as np
from matplotlib import pyplot as plt
import cards
from count_symbols import CountSymbols
from featuriser import Featuriser


def threshes_example(img):
    f1 = CountSymbols(img)
    blur = cv2.medianBlur(f1.grey_image, 5)
    thresh1 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 8)
    thresh2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)
    thresh3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
    thresh4 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh5 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in xrange(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()


def contour_example(img):
    ret, thresh = cv2.threshold(img, 129, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[4]
    print "{0} contours found".format(len(contours))
    cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
    plt.imshow(img)
    plt.title("Contour 4")
    plt.show()


def bounding_box_example(img):
    # turn the image into binary (black and white, no grey)
    ret, thresh = cv2.threshold(img, 129, 255, cv2.THRESH_BINARY)
    # find all the contours in the image, all areas of joint white/black
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # figure out the largest contour area
    card_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # draw a rotated box around the biggest contour
    rect = cv2.minAreaRect(card_contour)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # display the image
    plt.imshow(img)
    plt.title("bounding box")
    plt.show()


def sift_example(img):
    sift = cv2.SIFT()
    kp = sift.detect(img, None)

    img = cv2.drawKeypoints(img, kp)
    # display the image
    plt.imshow(img)
    plt.title("SIFT image")
    plt.show()


def surf_example(img):
    # Create SURF object. You can specify params here or later.
    # Here I set Hessian Threshold to 400 (best to be between 300-500)
    surf = cv2.SURF(5000)

    # Find keypoints and descriptors directly
    kp, des = surf.detectAndCompute(img, None)
    print des

    img = cv2.drawKeypoints(img, kp)
    # display the image
    plt.imshow(img)
    plt.title("SURF image")
    plt.show()


def orb_example(img):
    # Initiate STAR detector
    orb = cv2.ORB()

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # draw only keypoints location, not size and orientation
    img2 = cv2.drawKeypoints(img, kp, color=(0, 255, 0), flags=0)

    # display the image
    plt.imshow(img2)
    plt.title("ORB image")
    plt.show()


def normalise_image(img):
    print img.shape
    for x in range(0, len(img)):
        for y in range(0, len(img[x])):
            b = img.item(x, y, 0)
            g = img.item(x, y, 1)
            r = img.item(x, y, 2)
            total = r + b + g
            img.itemset((x, y, 0), (255.0 * b / total))
            img.itemset((x, y, 1), (255.0 * g / total))
            img.itemset((x, y, 2), (255.0 * r / total))

    # display the image
    plt.imshow(img)
    plt.title("")
    plt.show()


def features_example(img):
    f = Featuriser(img)

    relevant_contours = f.relevant_contours

    for cnt in relevant_contours:
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
    plt.imshow(img)
    plt.title("Features")
    plt.show()


def count_symbols_example(img):
    f1 = CountSymbols(img)
    """print f1.symbol_count

    for cnt in f1.symbol_contours:
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)"""
    blur = cv2.medianBlur(f1.grey_image, 5)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    plt.imshow(thresh)
    plt.title("Features")
    plt.show()


def adaptive_threshold_example(img):
    img = cv2.medianBlur(img, 5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)

    plt.imshow(img)
    plt.title("Features")
    plt.show()


if __name__ == "__main__":
    image = cv2.imread('Images/ivr1415pract1data2/test2.jpg')
    threshes_example(image)