import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

def capture_histogram(source):
    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()

    while ret:
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (900, 600))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, """Place hand inside the box and press 'A'""",
            (round(50), 50), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (500, 100), (580, 180), (105, 105, 105), 2)
        box = frame[105:175, 505:575]
        cv2.imshow("Hand Capture", frame)
        ret, frame = cap.read()
        key_e = cv2.waitKey(10)
        win_e = cv2.getWindowProperty("Hand Capture", 1)
        if key_e == ord('a'):
            object_color = box
            cv2.destroyAllWindows()
            break
        if key_e == ord('q') or win_e == -1:
            cap.release()
            cv2.destroyAllWindows()
            break

    object_color_hsv = cv2.cvtColor(object_color, cv2.COLOR_BGR2HSV)
    object_hist = cv2.calcHist([object_color_hsv], [0, 1], None,
        [12, 15], [0, 180, 0, 256])
    cv2.normalize(object_hist, object_hist, 0, 255, cv2.NORM_MINMAX)
    return object_hist

plt.plot(capture_histogram(0))
plt.show()
