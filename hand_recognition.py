import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

def capture_histogram(source):
    cap = cv2.VideoCapture(source)

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1000, 600))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, """Place hand inside the box and press 'A'""",
            (round(50), 50), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (500, 100), (580, 180), (105, 105, 105), 2)
        box = frame[105:175, 505:575]
        cv2.imshow("Hand Capture", frame)
        key_e = cv2.waitKey(10)
        win_e = cv2.getWindowProperty("Hand Capture", 1)
        if key_e == ord('a'):
            object_color = box
            cv2.destroyAllWindows()
            break
        if key_e == ord('q') or win_e == -1:
            cv2.destroyAllWindows()
            cap.release()
            break

    object_color_hsv = cv2.cvtColor(object_color, cv2.COLOR_BGR2HSV)
    object_hist = cv2.calcHist([object_color_hsv], [0, 1], None,
        [12, 15], [0, 180, 0, 256])
    cv2.normalize(object_hist, object_hist, 0, 255, cv2.NORM_MINMAX)
    return object_hist


def locate_object(frame, object_hist):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    object_segment = cv2.calcBackProject(
        [hsv_frame], [0, 1], object_hist, [0, 180, 0, 256], 1)

    _, segment_thresh = cv2.threshold(
        object_segment, 70, 255, cv2.THRESH_BINARY)

    kernel = None
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    filtered = cv2.filter2D(segment_thresh, -1, disc)
    eroded = cv2.erode(filtered, kernel, iterations=2)
    dilated = cv2.dilate(eroded, kernel, iterations=2)
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
    masked = cv2.bitwise_and(frame, frame, mask=closing)
    return closing, masked, segment_thresh

def detect_hand(frame, hist):
    return_value = {}

    detected_hand, masked, raw = locate_object(frame, hist)
    return_value["binary"] = detected_hand
    return_value["masked"] = masked
    return_value["raw"] = raw

    return return_value
