import cv2
import os
import sys

input_dir = os.listdir("Data")

counter = 1
for item in input_dir:
    print(item)
    image = cv2.imread("Data/" + item)
    image = cv2.resize(image, (480, 300), interpolation=cv2.INTER_AREA)
    cv2.imshow("", image)
    cv2.waitKey(1)
    cv2.imwrite("Train/" + str(counter) + ".jpg", image)
    counter += 1
