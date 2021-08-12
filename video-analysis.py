import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_EXPOSURE,-10)

while True:

    _, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_green = np.array([30, 55, 55])
    upper_green = np.array([90, 255, 255])

    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])

    mask = cv.inRange(hsv, lower_green, upper_green)
    res = cv.bitwise_and(frame, frame, mask = mask)

    kernal = np.ones((5, 5), np.uint8)

    dilation = cv.dilate(mask, kernal, iterations= 2)
    erosion = cv.erode(mask, kernal, iterations= 1)
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernal) #'first erosion then dilation'
    closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernal) #'first dilation then erosion'

    contours, hierarchy = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame, contours, -1, (255, 0, 0), 2)

    for cnt in contours:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        img = cv.drawContours(frame, [box], 0, (0, 255, 0), 2)

    cv.imshow('opening', opening)
    cv.imshow('frame', frame)
    # cv.imshow('mask', mask)
    cv.imshow('res', res)

    k = cv.waitKey(1)
    if k == 27:
        break

cv.destroyAllWindows()
cap.relese()
