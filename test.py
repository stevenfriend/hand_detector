import cv2
import numpy as np

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

while ret:
    cv2.imshow("Video", frame)
    ret, frame = cap.read()
    key_e = cv2.waitKey(10)
    win_e = cv2.getWindowProperty("Video", 1)
    if key_e == ord('q') or win_e == -1:
        break

cap.release()
cv2.destroyAllWindows()
