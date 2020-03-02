import cv2
import numpy as np
import os
import sys
#
# def detect_paper():
#     image = cv2.imread('test.jpg', -1)
#     # image = cv2.resize(image, (500, 500))
#     ret, thresh_gray = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     for c in contours:
#         rect = cv2.minAreaRect(c)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)
#         cv2.drawContours(image, [box], 0, (0, 255, 0), 1)
#
#     cv2.imwrite('image.jpg', image)
#
# detect_paper()
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
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    image[dst > 0.01 * dst.max()] = (0, 0, 255)
    print(image)
    print(image.shape)
    cv2.imwrite('cornered.jpg', image)
    cv2.imshow('dst.jpg', image)
    cv2.waitKey(500)

def crop_and_save_max_area():
    img = cv2.imread('test.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)
    contours, hier = cv2.findContours(thresh_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    mx = (0, 0, 0, 0) # biggest box so far
    mx_area = 0
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = w * h
        print("area: {}".format(area))
        if area > mx_area:
            mx = x, y, w, h
            mx_area = area
            print("max area : {}".format(mx_area))
    x, y, w, h = mx
    roi = img[y:y+h, x:x+w]
    cv2.imwrite('crop_rect.jpg', roi)

    cv2.rectangle(img, (x, y), (x+w, y+h), (230, 0, 0), 2)
    cv2.imwrite('img_cont.jpg', img)

crop_and_save_max_area()

if __name__ == '__main__':
    img = cv2.imread('test.jpg')

    # Const
    MORPH = 9
    CANNY = 84
    HOUGH = 25

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    cv2.GaussianBlur(gray_img, (3, 3), 0, gray_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
    dilated = cv2.dilate(gray_img, kernel)

    edges = cv2.Canny(dilated, 0, CANNY, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, 3.14/180, HOUGH)

    for line in lines[0]:
        cv2.line(edges, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2, 0)

    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)

    rects = []

    for c in contours:
        rect = cv2.approxPolyDP(c, 40, True).copy().reshape(-1, 2)
        rects.append(rect)

    cv2.drawContours(img, rects, -1, (0, 255, 0), 1)

    new_img = get_new(img)
    cv2.imwrite('first.jpg', new_img)
    cv2.drawContours(new_img, rects, -1, (0, 255, 0), 1)
    cv2.GaussianBlur(new_img, (9,9), 0, new_img)
    new_img = cv2.Canny(new_img, 0, CANNY, apertureSize=3)
    cv2.imwrite('sec.jpg', new_img)

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.imshow('result', img)
    cv2.waitKey(5000)
    cv2.imwrite('result.jpg', img)

    cv2.destroyAllWindows()
    corner_detection()
