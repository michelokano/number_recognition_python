# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np

# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 1, 0): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

def image_recognition():
    # load the example image
    image = cv2.imread("3.jpg")

    # pre-process the image by resizing it, converting it to
    # graycale, blurring it, and computing an edge map
    image = imutils.resize(image, height=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)

    # find contours in the edge map, then sort them by their
    # size in descending order
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if the contour has four vertices, then we have found
        # the thermostat display
        if len(approx) == 4:
            displayCnt = approx
            break

    # extract the thermostat display, apply a perspective transform
    # to it
    warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    output = four_point_transform(image, displayCnt.reshape(4, 2))

    # threshold the warped image, then apply a series of morphological
    # operations to cleanup the thresholded image


    # https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html

    # threshold = 3
    # kernel = 4
    # transform = 7


    thresh11 = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh12 = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh21 = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    thresh22 = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY_INV,11,2)
    thresh31 = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    thresh32 = cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY_INV,11,2)

    thresh_list = {
        "thresh11" : thresh11,
        "thresh12" : thresh12,
        "thresh21" : thresh21,
        "thresh22" : thresh22,
        "thresh31" : thresh31,
        "thresh32" : thresh32,
    }

    images_list = {
        "thresh11" : thresh11,
        "thresh12" : thresh12,
        "thresh21" : thresh21,
        "thresh22" : thresh22,
        "thresh31" : thresh31,
        "thresh32" : thresh32,
    }

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))

    # Rectangular Kernel
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

    # Elliptical Kernel
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    # Cross-shaped Kernel
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

    kernel = np.ones((5,5),np.uint8)

    for thresh in thresh_list:
        images_list["erode" + thresh] = cv2.erode(thresh_list[thresh],kernel,iterations = 1)
        images_list["dilate" + thresh] = cv2.dilate(thresh_list[thresh], kernel, iterations=1)
        images_list["opening" + thresh] = cv2.morphologyEx(thresh_list[thresh], cv2.MORPH_OPEN, kernel)
        images_list["closing" + thresh] = cv2.morphologyEx(thresh_list[thresh], cv2.MORPH_CLOSE, kernel)
        images_list["gradient" + thresh] = cv2.morphologyEx(thresh_list[thresh], cv2.MORPH_GRADIENT, kernel)
        images_list["tophat" + thresh] = cv2.morphologyEx(thresh_list[thresh], cv2.MORPH_TOPHAT, kernel)
        images_list["blackhat" + thresh] = cv2.morphologyEx(thresh_list[thresh], cv2.MORPH_BLACKHAT, kernel)

    # for image in images_list:
    #     cv2.imshow("1", images_list[image])
    #     print(image)
    # cv2.waitKey(0)

    for image in images_list:
        cnts = cv2.findContours(images_list[image].copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        digitCnts = []

        # loop over the digit area candidates
        for c in cnts:
            # compute the bounding box of the contour
            (x, y, w, h) = cv2.boundingRect(c)

            # if the contour is sufficiently large, it must be a digit
            if w >= 15 and (h >= 30 and h <= 40):
                digitCnts.append(c)

        # sort the contours from left-to-right, then initialize the
        # actual digits themselves
        try:
            digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
            digits = []

            # loop over each of the digits
            for c in digitCnts:
                # extract the digit ROI
                (x, y, w, h) = cv2.boundingRect(c)
                roi = images_list[image][y:y + h, x:x + w]

                # compute the width and height of each of the 7 segments
                # we are going to examine
                (roiH, roiW) = roi.shape
                (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
                dHC = int(roiH * 0.05)

                # define the set of 7 segments
                segments = [
                    ((0, 0), (w, dH)),  # top
                    ((0, 0), (dW, h // 2)),  # top-left
                    ((w - dW, 0), (w, h // 2)),  # top-right
                    ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
                    ((0, h // 2), (dW, h)),  # bottom-left
                    ((w - dW, h // 2), (w, h)),  # bottom-right
                    ((0, h - dH), (w, h))  # bottom
                ]
                on = [0] * len(segments)

                # loop over the segments
                for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
                    # extract the segment ROI, count the total number of
                    # thresholded pixels in the segment, and then compute
                    # the area of the segment
                    segROI = roi[yA:yB, xA:xB]
                    total = cv2.countNonZero(segROI)
                    area = (xB - xA) * (yB - yA)

                    # if the total number of non-zero pixels is greater than
                    # 50% of the area, mark the segment as "on"
                    if total / float(area) > 0.5:
                        on[i] = 1

                # lookup the digit and draw it on the image
                digit = DIGITS_LOOKUP[tuple(on)]
                digits.append(digit)
        except:
            digits = 0

        cv2.imwrite("after_images/" + image + '.jpg', images_list[image])

        images_list[image] = None
        images_list[image] = digits

    return images_list