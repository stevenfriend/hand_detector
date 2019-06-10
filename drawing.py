import cv2
import numpy as np
from hand_recognition import *

hist = capture_histogram(0)
cap = cv2.VideoCapture(0)

# initialize a black canvas
screen = np.zeros((600, 1000))

curr = None
prev = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1000, 600))

    hand_detected, hand = detect_hand(frame, hist)
    if hand_detected:
        hand_image = hand["boundaries"]
        fingertips = extract_fingertips(hand)
        plot(hand_image, fingertips)
        prev = curr
        curr = fingertips[0]
        if prev and curr:
            cv2.line(screen, prev, curr, (255, 0, 0), 5)
        cv2.imshow("Drawing", screen)
        cv2.imshow("Hand Detector", hand_image)
    else:
        cv2.imshow("Hand Detector", frame)

    key_e = cv2.waitKey(10)
    if key_e == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
