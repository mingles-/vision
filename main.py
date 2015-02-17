import cv2

import numpy as np
from matplotlib import pyplot as plt


def threshes_example(img):
    ret, thresh1 = cv2.threshold(img, 129, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

    for i in xrange(6):
        plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
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
    biggest_cnt = None
    biggest_cnt_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > biggest_cnt_area:
            biggest_cnt = cnt
            biggest_cnt_area = area

    # draw a rotated box around the biggest contour
    rect = cv2.minAreaRect(biggest_cnt)
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
    # Here I set Hessian Threshold to 400
    surf = cv2.SURF(400)

    # Find keypoints and descriptors directly
    kp, des = surf.detectAndCompute(img, None)

    img = cv2.drawKeypoints(img, kp)
     # display the image
    plt.imshow(img)
    plt.title("SURF image")
    plt.show()


if __name__ == "__main__":
    image = cv2.imread('Images/ivr1415pract1data1/train32.jpg', 0)
    surf_example(image)