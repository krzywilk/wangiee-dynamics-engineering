import cv2
import os
import numpy as np
import math
import heapq

from video_utils import load_video


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
    test_sample_1 = "../videos/sample_1.mp4"
    frames = load_video(test_sample_1)
    # frames = frames
    line_image_previous = np.ones_like(frames[0])
    previous_frame = np.ones_like(frames[0])
    lines_windowed = []
    aa= []
    for i, frame in enumerate(frames):

        print(i)
        # cv2.imshow("ping", frame)
        frameDelta = cv2.absdiff(previous_frame, frame)
        frameDelta[frameDelta < 5] = 0
        mask = np.zeros_like(frames[0])
        frameDelta[:,:,0] = frameDelta[:,:,0] + frameDelta[:,:,1] + frameDelta[:,:,2]
        mask[frameDelta[:,:,0]>0] = 255
        mask = cv2.erode(mask, np.ones((3,3)), iterations=1)
        mask = cv2.dilate(mask, np.ones((10,6)), iterations=1)
        cv2.imshow("thr_motion", mask)
        output = cv2.connectedComponentsWithStats(
            mask[:,:,0])

        (numLabels, labels, stats, centroids) = output
        indexes = [x[0] for x in heapq.nlargest(2, enumerate(stats[1:, cv2.CC_STAT_AREA]), key=lambda x: x[1])]
        output = mask.copy()
        bbmask = np.zeros_like(frame)
        for i in indexes:
            i=i+1
            # if this is the first component then we examine the
            # *background* (typically we would just ignore this
            # component in our loop)
            if i == 0:
                text = "examining component {}/{} (background)".format(
                    i + 1, numLabels)
            # otherwise, we are examining an actual connected component
            else:
                text = "examining component {}/{}".format(i + 1, numLabels)
            # print a status message update for the current connected
            # component
            print("[INFO] {}".format(text))
            # extract the connected component statistics and centroid for
            # the current label
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            # if area < 200 or area > 0.5 * frame.shape[0]* frame.shape[1]:
            #     continue
            (cX, cY) = centroids[i]
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
            bbmask[y:y + h, x:x + w] = 255
            if not math.isnan(cX) and not math.isnan(cY):
                cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

        ff = frame.copy()
        ff[bbmask == 0] = 0
        cv2.imshow("frame", frame)
        cv2.imshow("bbmask", bbmask)
        cv2.imshow("frame_trimed", ff)
        cv2.imshow("connected components", output)
        cv2.waitKey()
        previous_frame = frame.copy()
        # aa.append(ff)

    # cap = cv2.VideoCapture(0)
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30.0, frame.shape[:2][::-1])
    # for frame in aa:
    #         # ret, frame = cap.read()
    #         if True:
    #             # frame = cv2.flip(frame, 0)
    #
    #             # write the flipped frame
    #             out.write(frame)
    #
    #             # cv2.imshow('frame', frame)
    #             # if cv2.waitKey(1) & 0xFF == ord('q'):
    #             #     break
    #         else:
    #             break
    #
    #     # Release everything if job is finished
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()
    # for i, frame in enumerate(frames):
    #     print(i)
    #     cv2.imshow("ping", frame)
    #     # detect(frame)
    #     line_image = detect_lines_p(frame)
    #     constant_lines = np.logical_and(line_image_previous[:,:,0], line_image[:,:,0])
    #     line_image[np.logical_not(constant_lines)] = [0,0,0]
    #     line_image_previous = line_image
    #
    #     frameDelta = cv2.absdiff(previous_frame, frame)
    #     cv2.imshow("thr_motion", frameDelta)
    #     cv2.waitKey()
    #     previous_frame = frame.copy()
    #     if i%100 == 0:
    #
    #         lines_windowed.append(line_image)
    #         line_image_previous = np.ones_like(frames[0])
    #         cv2.imshow("semi_strong_lines", line_image)
    #         cv2.waitKey()
    # strong_lines = np.zeros_like(line_image)
    # for line_window in lines_windowed:
    #     strong_lines[line_window > 0] = line_window[line_window > 0]
    # cv2.imshow("strong_lines", strong_lines)
    # cv2.waitKey()
