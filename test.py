import cv2
import numpy as np
from hand_recognition import *

hist = capture_histogram(0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hand = detect_hand(frame, hist)

    cv2.imshow("Raw", hand["raw"])
    cv2.imshow("Enhanced Binary", hand["binary"])
    cv2.imshow("Masked", hand["masked"])

    key_e = cv2.waitKey(10)
    if key_e == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
