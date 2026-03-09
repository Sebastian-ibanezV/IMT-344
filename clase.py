import cv2
import numpy as np

img = cv2.imread("example.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

low1, high1 = (0, 120, 70), (10, 255, 255)
low2, high2 = (170, 120, 70), (179, 255, 255)

m1 = cv2.inRange(hsv, np.array(low1), np.array(high1))
m2 = cv2.inRange(hsv, np.array(low2), np.array(high2))

mask = cv2.bitwise_or(m1, m2)

kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

result = cv2.bitwise_and(img, img, mask=mask)

cv2.imwrite("mask.png", mask)
cv2.imwrite("result.png", result)