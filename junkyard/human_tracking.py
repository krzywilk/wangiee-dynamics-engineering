from random import random, randrange

import cv2
import os
import numpy as np
import math
import heapq

from bbox_utils import bb_intersection
from detections.player_detection import framewise_difference_segmentation, two_players_blob_detection
from detections.table_detection import table_rectangle, table_detection, detect_intersection_points
from video_utils import load_video, draw_players_bboxes


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


from sklearn.cluster import KMeans, DBSCAN
# import the necessary packages
import numpy as np
import cv2


def filter_close_lines(lines, atol_rho=10, atol_theta=np.pi / 36):
    """
    Filters lines close to each other.
    :param lines: Lines np.ndarray
    :param atol_rho: The absolute tolerance parameter used to filter close rho values of lines.
    :param atol_theta: The absolute tolerance parameter used to filter close theta values of lines.
    :return: Filtered lines np.ndarray
    """
    res_lines_alloc = np.zeros([len(lines), 1, 2])
    res_lines_alloc[0] = lines[0]
    end_alloc_pointer = 1
    for line in lines:
        rho, theta = line[0]
        if rho < 0:
            rho *= -1
            theta -= np.pi
        closeness_rho = np.isclose(rho, res_lines_alloc[0:end_alloc_pointer, 0, 0], atol=atol_rho)
        closeness_theta = np.isclose(theta, res_lines_alloc[0:end_alloc_pointer, 0, 1], atol=atol_theta)
        closeness = np.all([closeness_rho, closeness_theta], axis=0)
        if not any(closeness):
            res_lines_alloc[end_alloc_pointer] = line
            end_alloc_pointer += 1
    return res_lines_alloc[:end_alloc_pointer]


import numpy as np
import cv2
from collections import defaultdict
from sklearn import preprocessing
from sklearn.cluster import KMeans


def cluster_lines(lines):
    """
    Clusters lines into vertical and horizontal groups with the DBSCAN algorithm. Lines recognized as an
    outliers (e.g. 45 degrees lines) are not returned
    :param lines: Lines np.ndarray
    :return: List of lines clusters
    """
    angles = np.array([line[0][1] for line in lines])
    pts = np.array([[np.cos(2 * theta), np.sin(2 * theta)]
                    for theta in angles], dtype=np.float32)
    pts = preprocessing.normalize(pts)
    db = KMeans(n_clusters=2).fit(pts)
    labels = db.labels_
    segmented = defaultdict(list)
    for line, i in zip(lines, range(len(lines))):
        if labels[i] > -1:
            segmented[labels[i]].append(line)
    return list(segmented.values())


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def intersection(line1, line2):
    """
    Finds intersection point of two lines.
    src: https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
    :param line1: First line array containing rho and theta
    :param line2: second line array containing rho and theta
    :return: x and y of intersection point
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return x0, y0


def find_intersections(lines_clusters):
    """
    Finds the intersections between groups of lines.
    src: https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
    :param lines_clusters: List of clustered lines
    :return: np.ndarray of intersection points
    """
    intersections = []
    first_group_lines = lines_clusters[0]
    second_group_lines = lines_clusters[1]
    for first_line in first_group_lines:
        for second_line in second_group_lines:
            intersections.append(intersection(first_line, second_line))
    return np.array(intersections)


def find_intersections_centroids(intersections, eps, min_samples=1):
    """
    Clusters intersections with the DBSCAN algorithm, and compute mean point for each cluster.
    :param intersections: np.ndarray of intersection points
    :param eps: maximum distance between points
    :param min_samples: minimum samples in centroid cluster
    :return: np.ndarray of mean point for each centroid
    """
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(intersections)
    labels = db.labels_
    centroids = np.array(
        [intersections[labels == i].mean(axis=0).astype(int) for i in range(np.min(labels), np.max(labels) + 1)])
    centroids = centroids[np.argsort(centroids[:, 1])]
    return centroids


def organize_intersection_centroids(intersections, eps):
    """
    Organizes intersections in rows and lines with correct order.
    :param intersections: np.ndarray of intersection points
    :param eps: max position difference between points in line
    :return:
        intersections: np.ndarray of intersection points organised from left to right and from top to the bottom
        centroids_sparse_matrix: sparse np.ndarray of intersection points organised in matrix
    """
    centroids_sparse_matrix = np.empty((len(intersections), len(intersections), 2))
    centroids_sparse_matrix[:] = np.nan
    pointer = 0
    row = 0
    max_col = 0
    for i in range(1, len(intersections) + 1):
        if i == len(intersections) or intersections[i][1] > intersections[i - 1][1] + eps:
            intersections[pointer: i] = intersections[pointer: i][np.argsort(intersections[pointer:i, 0])]
            centroids_sparse_matrix[row][0: i - pointer] = intersections[pointer: i]
            if max_col < i - pointer:
                max_col = i - pointer
            row += 1
            pointer = i
    return intersections, centroids_sparse_matrix


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    net = np.zeros((2, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    left_top_ind = np.argmin(s)
    rect[0] = pts[left_top_ind]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    left_bottom_ind = np.argmax(diff)
    rect[3] = pts[left_bottom_ind]
    net = pts[np.logical_and(pts[:, 1] > rect[0][1], pts[:, 1] < rect[3][1])]  ## to improve
    centroid = rect.mean(axis=0)
    # return the ordered coordinates
    return rect, centroid


def clear_doubled_frames(vid_path):
    frames = load_video(vid_path)
    previous_frame = np.zeros_like(frames[0])
    out2 = []
    for i, frame in enumerate(frames):
        frameDelta = cv2.absdiff(previous_frame, frame)
        frameDelta[frameDelta < 30] = 0
        cv2.imshow("", frameDelta)
        cv2.imshow("2", frame)
        xx = cv2.waitKey()
        if xx == ord('a'):
            print("asd")
            continue

        previous_frame = frame
        out2.append(frame)
    cap = cv2.VideoCapture(0)
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 30.0, frame.shape[:2][::-1])
    for frame in out2:
        # ret, frame = cap.read()
        if True:
            # frame = cv2.flip(frame, 0)

            # write the flipped frame
            out.write(frame)

            # cv2.imshow('frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


from selective_search import *

if __name__ == "__main__":
    test_sample_1 = "../videos/sample_1.mp4"
    # clear_doubled_frames(test_sample_1)

    frames = load_video(test_sample_1)
    # frames = frames
    line_image_previous = np.ones_like(frames[0])
    previous_frame = np.ones_like(frames[0])
    lines_windowed = []
    aa = []
    ratios = []
    table_mask = table_detection(frames)
    table_intersection_points, centroid = detect_intersection_points(table_mask)
    difference_segmentation = framewise_difference_segmentation(frames)
    for i, mask in enumerate(difference_segmentation):
        if i == 0:
            continue
        if i==145:
            print()
        # mask = cv2.erode(mask, np.ones((3, 3)), iterations=1)
        # mask = cv2.dilate(mask, np.ones((10, 6)), iterations=1)
        output_bottom = cv2.connectedComponentsWithStats(
            mask[int(max(table_intersection_points[0][1], table_intersection_points[1][1])):, :, 0])

        (numLabels, labels, stats, centroids) = output_bottom

        bottom_player = [x[0] + 1 for x in
                           heapq.nlargest(1, enumerate(stats[1:, cv2.CC_STAT_AREA]), key=lambda x: x[1])]
        bottom_player = (stats[bottom_player[0]], centroids[bottom_player[0]])
        bottom_player[0][cv2.CC_STAT_TOP] = bottom_player[0][cv2.CC_STAT_TOP] + int(max(table_intersection_points[0][1], table_intersection_points[1][1]))
        bottom_player[1][1] = bottom_player[1][1] + int(max(table_intersection_points[0][1], table_intersection_points[1][1]))
        msk = mask.copy()
        bbox_bottom = [bottom_player[0][cv2.CC_STAT_LEFT],bottom_player[0][cv2.CC_STAT_TOP], bottom_player[0][cv2.CC_STAT_LEFT]+bottom_player[0][cv2.CC_STAT_WIDTH],bottom_player[0][cv2.CC_STAT_TOP]+bottom_player[0][cv2.CC_STAT_HEIGHT]]

        msk[bbox_bottom[1]:bbox_bottom[3],bbox_bottom[0]:bbox_bottom[2]]= [0,0,0]

        output_top = cv2.connectedComponentsWithStats(
            msk[:abs(
                int(max(table_intersection_points[0][1], table_intersection_points[1][1])) - int(
                    max(table_intersection_points[2][1], table_intersection_points[3][1]))
            ) // 2 + int(max(table_intersection_points[0][1], table_intersection_points[1][1])), :, 0])
        (numLabels, labels, stats, centroids) = output_top
        top_player = [x[0] + 1 for x in
                      heapq.nlargest(1, enumerate(stats[1:, cv2.CC_STAT_AREA]), key=lambda x: x[1])]
        top_player = (stats[top_player[0]], centroids[top_player[0]])


        bbox_top = [top_player[0][cv2.CC_STAT_LEFT],top_player[0][cv2.CC_STAT_TOP], top_player[0][cv2.CC_STAT_LEFT]+top_player[0][cv2.CC_STAT_WIDTH],top_player[0][cv2.CC_STAT_TOP]+top_player[0][cv2.CC_STAT_HEIGHT]]
        # players_indexes = [x[0] + 1 for x in
        #                    heapq.nlargest(2, enumerate(stats[1:, cv2.CC_STAT_AREA]), key=lambda x: x[1])]
        # if True:
        #     if centroids[players_indexes[0]][1] < centroids[players_indexes[1]][1]:
        #         top_player = (stats[players_indexes[0]], centroids[players_indexes[0]])
        #         bottom_player = (stats[players_indexes[1]], centroids[players_indexes[1]])
        #     else:
        #         top_player = (stats[players_indexes[1]], centroids[players_indexes[1]])
        #         bottom_player = (stats[players_indexes[0]], centroids[players_indexes[0]])
        ratio = min(
            [top_player[0][cv2.CC_STAT_AREA], bottom_player[0][cv2.CC_STAT_AREA]]) / max(
            [top_player[0][cv2.CC_STAT_AREA], bottom_player[0][cv2.CC_STAT_AREA]])
        ratios.append(ratio)
        # CC_STAT_AREAprint(ratio)
        box = []
        area = []
        # if i == 46:
        #     print()
        # if ratio > 0.6:
        #     if top_player[0][cv2.CC_STAT_AREA] < bottom_player[0][cv2.CC_STAT_AREA]:
        #         if bottom_player[0][cv2.CC_STAT_TOP] < max(table_intersection_points[0][0],
        #                                                 table_intersection_points[1][0]) \
        #                 and bottom_player[0][cv2.CC_STAT_TOP] + bottom_player[0][cv2.CC_STAT_HEIGHT] > max(
        #                 table_intersection_points[2][1], table_intersection_points[3][1]):
        #             top_player[0][cv2.CC_STAT_TOP] = bottom_player[0][cv2.CC_STAT_TOP] + player[0][cv2.CC_STAT_HEIGHT] // 2
        #             top_player[0][cv2.CC_STAT_HEIGHT] = player[0][cv2.CC_STAT_HEIGHT] // 2
        #             print(i)
        #     else:
        #         if top_player[0][cv2.CC_STAT_TOP] < max(table_intersection_points[0][0],
        #                                                 table_intersection_points[1][0]) \
        #                 and top_player[0][cv2.CC_STAT_TOP] + top_player[0][cv2.CC_STAT_HEIGHT] > max(
        #                 table_intersection_points[2][1], table_intersection_points[3][1]):
        #             bottom_player[0][cv2.CC_STAT_TOP] = top_player[0][cv2.CC_STAT_TOP] + player[0][cv2.CC_STAT_HEIGHT] // 2
        #             bottom_player[0][cv2.CC_STAT_HEIGHT] = player[0][cv2.CC_STAT_HEIGHT] // 2
        #             print(i)
        for j, player in enumerate([top_player, bottom_player]):
            x = player[0][cv2.CC_STAT_LEFT]
            y = player[0][cv2.CC_STAT_TOP]
            w = player[0][cv2.CC_STAT_WIDTH]
            h = player[0][cv2.CC_STAT_HEIGHT]
            cv2.rectangle(frames[i], (x, y), (x + w, y + h), (255, 0, 0), 4)
            area.append(player[0][cv2.CC_STAT_AREA])

        # box = non_max_suppression_fast(np.array(box[1:]), 0.8)
        # players_indexes_2 = [x[0] + 1 for x in
        #                      heapq.nlargest(2, enumerate(box[1:]), key=lambda x: (x[1][2] - x[1][0]) * (x[1][3] - x[1][1]))]
        # for p in players_indexes_2:
        #     player = box[p]
        #     x = player[cv2.CC_STAT_LEFT]
        #     y = player[cv2.CC_STAT_TOP]
        #     w = player[cv2.CC_STAT_WIDTH]
        #     h = player[cv2.CC_STAT_HEIGHT]
        #     cv2.rectangle(frames[i], (x, y), ( w, h), (0, 255, 0), 2)

        # box.append((x, y,x + w, y + h))
    for i, frame in enumerate(frames):
        print(i, ratios[i])
        cv2.imshow("frame", frame)

        segs = difference_segmentation[i]
        seg_top = segs[:abs(
                int(max(table_intersection_points[0][1], table_intersection_points[1][1])) - int(
                    max(table_intersection_points[2][1], table_intersection_points[3][1]))
            ) // 2 + int(max(table_intersection_points[0][1], table_intersection_points[1][1])),:,:]
        cv2.imshow("top seg", seg_top)
        seg_bot = segs[int(max(table_intersection_points[0][1], table_intersection_points[1][1])):, :, :]
        cv2.imshow("top bottom", seg_bot)

        cv2.waitKey()
