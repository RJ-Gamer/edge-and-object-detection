import cv2
import numpy as np
import os
import sys

def get_new(old):
    new = np.ones(old.shape, np.uint8)
    cv2.bitwise_not(new, new)
    return new

def corner_detection():
    filename = 'test.jpg'
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # find Harris Corners
    gray = np.float32(gray)
    print(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    image[dst > 0.01 * dst.max()] = (0, 0, 255)
    cv2.imwrite('cornered.jpg', image)
    cv2.imshow('dst.jpg', image)
    cv2.waitKey(500)
corner_detection()
#
#
# if __name__ == '__main__':
#     img = cv2.imread('test.jpg')
#
#     # Const
#     MORPH = 9
#     CANNY = 84
#     HOUGH = 25
#
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#
#     cv2.GaussianBlur(gray_img, (3, 3), 0, gray_img)
#
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
#     dilated = cv2.dilate(gray_img, kernel)
#
#     edges = cv2.Canny(dilated, 0, CANNY, apertureSize=3)
#
#     lines = cv2.HoughLinesP(edges, 1, 3.14/180, HOUGH)
#
#     for line in lines[0]:
#         cv2.line(edges, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2, 0)
#
#     contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
#
#     rects = []
#
#     for c in contours:
#         rect = cv2.approxPolyDP(c, 40, True).copy().reshape(-1, 2)
#         rects.append(rect)
#
#     cv2.drawContours(img, rects, -1, (0, 255, 0), 1)
#
#     new_img = get_new(img)
#     cv2.drawContours(new_img, rects, -1, (0, 255, 0), 1)
#     cv2.GaussianBlur(new_img, (9,9), 0, new_img)
#     new_img = cv2.Canny(new_img, 0, CANNY, apertureSize=3)
#
#     cv2.namedWindow('result', cv2.WINDOW_NORMAL)
#     cv2.imshow('result', img)
#     cv2.waitKey(5000)
#     cv2.imwrite('result.jpg', img)
#
#     cv2.destroyAllWindows()
