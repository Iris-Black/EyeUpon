import cv2 as cv
import numpy as np

img = cv.imread('/Users/efecete06/PycharmProjects/photoAnalysis/res/opencv.jpg', cv.IMREAD_COLOR)
justimg = cv.imread('/Users/efecete06/PycharmProjects/photoAnalysis/res/opencv.jpg', cv.IMREAD_COLOR)
detectimg = cv.imread('/Users/efecete06/PycharmProjects/photoAnalysis/res/opencv.jpg', cv.IMREAD_COLOR)

while True:

    frame = img
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    detectimg = cv.cvtColor(detectimg, cv.COLOR_BGR2GRAY)

    lower_green = np.array([40, 140, 20])
    upper_green = np.array([90, 255, 255])

    mask = cv.inRange(hsv, lower_green, upper_green)
    res = cv.bitwise_and(frame, frame, mask=mask)

    kernal = np.ones((5, 5), np.uint8)

    dilation = cv.dilate(mask, kernal, iterations=2)
    erosion = cv.erode(mask, kernal, iterations=1)
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernal)  # 'first erosion then dilation'
    closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernal)  # 'first dilation then erosion'

    sift = cv.SIFT_create()
    op_keypoints = sift.detect(opening, None)
    all_keypoints = sift.detect(detectimg, None)

    op_detectimg = cv.drawKeypoints(opening, op_keypoints, opening, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    all_detectimg = cv.drawKeypoints(detectimg, all_keypoints, opening, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    contours, hierarchy = cv.findContours(opening, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(contours[0])

    cv.drawContours(frame, contours, -1, (0, 0, 255), 1)

    for cnt in contours:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        img = cv.drawContours(frame, [box], 0, (0, 255, 0), 1)

    cv.imshow('opening keypoints', op_detectimg)
    cv.imshow('all keypoints', all_detectimg)
    cv.imshow('opening', opening)
    # cv.imshow('mask', mask)
    cv.imshow('res', res)
    cv.imshow('frame', frame)
    cv.imshow('Normal Image', justimg)

    k = cv.waitKey(0)
    if k == 27:
        break

cv.destroyAllWindows()
img.relese()
