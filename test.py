import cv2
import numpy as np
from hand_recognition import *

hist = capture_histogram(0)
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    hand_detected, hand = detect_hand(frame, hist)
    if hand_detected:
        hand_image = hand["boundaries"]
        cv2.imshow("Hand Detector", hand_image)
    else:
        cv2.imshow("Hand Detector", frame)

    key_e = cv2.waitKey(10)
    if key_e == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
