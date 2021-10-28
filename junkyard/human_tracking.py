import cv2
import os
import numpy as np


def load_video(path):
    res = []
    cap = cv2.VideoCapture(path)
    ret = True
    while ret:
        ret, frame = cap.read()
        if ret:
            res.append(frame)
    return res


def detect(frame):
    frame = frame.copy()
    HOGCV = cv2.HOGDescriptor()
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    bounding_box_cordinates, weights = HOGCV.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)

    person = 1
    for x, y, w, h in bounding_box_cordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'person {person}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        person += 1

    cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f'Total Persons : {person - 1}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)
    cv2.imshow('output', frame)
    return frame

def detect_lines_p(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    low_threshold = 200
    high_threshold = 250
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(frame) * 0  # creating a blank to draw lines on

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    return line_image




if __name__ == "__main__":
    test_sample_1 = "../videos/sample_1_Trim.mp4"
    frames = load_video(test_sample_1)
    line_image_previous = np.ones_like(frames[0])
    previous_frame = np.ones_like(frames[0])
    lines_windowed = []
    for i, frame in enumerate(frames):

        print(i)
        # cv2.imshow("ping", frame)
        frameDelta = cv2.absdiff(previous_frame, frame)
        frameDelta[frameDelta < 15] = 0
        mask = np.ones_like(frames[0])
        frameDelta[:,:,0] = frameDelta[:,:,0] + frameDelta[:,:,1] + frameDelta[:,:,2]
        mask[frameDelta[:,:,0]>0] = 255
        cv2.imshow("thr_motion", mask)
        if np.all(frameDelta == 0):

            continue
        cv2.waitKey()
        previous_frame = frame.copy()
    for i, frame in enumerate(frames):
        print(i)
        cv2.imshow("ping", frame)
        # detect(frame)
        line_image = detect_lines_p(frame)
        constant_lines = np.logical_and(line_image_previous[:,:,0], line_image[:,:,0])
        line_image[np.logical_not(constant_lines)] = [0,0,0]
        line_image_previous = line_image

        frameDelta = cv2.absdiff(previous_frame, frame)
        cv2.imshow("thr_motion", frameDelta)
        cv2.waitKey()
        previous_frame = frame.copy()
        if i%100 == 0:

            lines_windowed.append(line_image)
            line_image_previous = np.ones_like(frames[0])
            cv2.imshow("semi_strong_lines", line_image)
            cv2.waitKey()
    strong_lines = np.zeros_like(line_image)
    for line_window in lines_windowed:
        strong_lines[line_window > 0] = line_window[line_window > 0]
    cv2.imshow("strong_lines", strong_lines)
    cv2.waitKey()
