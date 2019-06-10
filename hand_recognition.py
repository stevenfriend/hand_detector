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

    image, contours, _ = cv2.findContours(
        detected_hand, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    palm_area = 0
    flag = None
    cnt = None

    for (i, c) in enumerate(contours):
        area = cv2.contourArea(c)
        if area > palm_area:
            palm_area = area
            flag = i

    if flag is not None and palm_area > 10000:
        cnt = contours[flag]
        return_value["contours"] = cnt
        cpy = frame.copy()
        cv2.drawContours(cpy, [cnt], 0, (0, 255, 0), 2)
        return_value["boundaries"] = cpy
        return True, return_value
    else:
        return False, return_value

def extract_fingertips(hand):
    cnt = hand["contours"]
    points = []
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    # get all the "end points" using the defects and contours
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        end = tuple(cnt[e][0])
        points.append(end)

    # filter out the points which are too close to each other
    filtered = filter_points(points, 50)

    # sort the fingertips in order of increasing value of the y coordinate
    filtered.sort(key=lambda point: point[1])

    #return the fingertips, at most 5
    return [pt for idx, pt in zip(range(5), filtered)]

def dist(a, b):
    return math.sqrt((a[0] - b[0])**2 + (b[1] - a[1])**2)

def filter_points(points, filterValue):
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if points[i] and points[j] and dist(points[i], points[j]) < filterValue:
                points[j] = None
    filtered = []
    for point in points:
        if point is not None:
            filtered.append(point)
    return filtered

def plot(frame, points):
    radius = 5
    colour = (0, 0, 255)
    thickness = -1
    for point in points:
        cv2.circle(frame, point, radius, colour, thickness)
