import cv2
import numpy as np

image = cv2.imread('test.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(image, 10, 250)
# cv2.imshow('edged', edged)
# cv2.waitKey(5000)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
#
(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
cv2.imwrite('image.jpg', image)
cv2.imshow('output', image)
cv2.waitKey(60000)
#
# idx = 0
#
# for c in cnts:
#     x, y, w, h = cv2.boundingRect(c)
#     if w > 300 and h > 300:
#         idx += 1
#         new_img = image[y:y+h, x:x+w]
#         cv2.imwrite(str(idx) + '.jpg', new_img)
# cv2.imshow('output', image)
# cv2.waitKey(7000)
#
# image = cv2.pyrDown(cv2.imread('test.jpg', cv2.IMREAD_UNCHANGED))
#
# ret, thresh = cv2.threshold(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# for c in contours:
#     x, y, w, h = cv2.boundingRect(c)
#     cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)
#
#     rect = cv2.minAreaRect(c)
#
#     box = cv2.boxPoints(rect)
#
#     box = np.int0(box)
#
#     cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
#
#     cv2.imshow('counters', image)
#     cv2.waitKey(5000)
