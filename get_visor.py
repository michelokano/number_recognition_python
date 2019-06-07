from imutils.perspective import four_point_transform
import imutils
import cv2
import os
from get_numbers import getting_numbers_from_visor


def getting_visor(path, full_filename):
    filename, file_extension = os.path.splitext(full_filename)

    image = cv2.imread(path + full_filename)
    image = imutils.resize(image, height=500)
    cv2.imshow("Input", image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 200, 255)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            displayCnt = approx
            break

    warped = four_point_transform(image, displayCnt.reshape(4, 2))

    resized = imutils.resize(warped, height=300)
    cv2.imshow("Input", resized)
    save_path = 'images/visor/' + filename + '_visor' + file_extension
    cv2.imwrite(save_path, resized)
    cv2.waitKey(0)

    return save_path

path = 'images/source/'
filename = '1.jpg'

# getting_visor(path, filename)

getting_numbers_from_visor(getting_visor(path, filename))