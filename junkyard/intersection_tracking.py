import math

from detections.player_detection import framewise_difference_segmentation
from video_utils import load_video
import cv2
import numpy as np


if __name__ == "__main__":
    test_sample_1 = "../videos/sample_1.mp4"
    frames = load_video(test_sample_1)
    mostly_whites = []
    # for frame in frames:
    #     t, mostly_white = cv2.threshold(frame, 170, 255, cv2.THRESH_BINARY)
    #     white_frame = frame.copy()
    #     white_frame[mostly_white == 0] = 0
    #     mostly_whites.append(white_frame)


    difference_segmentation = framewise_difference_segmentation(frames)
    for i, frame in enumerate(frames):
        white_frame = frame.copy()
        white_frame[difference_segmentation[i][:, :, 0] == 0] = 0
        t, mostly_white = cv2.threshold(white_frame, 100, 255, cv2.THRESH_BINARY)

        white_frame[mostly_white.all(axis=2) == False] = 0
        mostly_whites.append(white_frame)
    connections = []
    differences_intersection_previous = np.zeros_like(frames[0])
    _, previous_labels, _, previous_centroids = cv2.connectedComponentsWithStats(difference_segmentation[0][:, :, 0])
    curr_labs = [previous_labels]
    curr_centroids = [previous_centroids]
    for i, frame in enumerate(frames):
        if i == 0:
            continue
        _, current_labels, _, current_centroids = cv2.connectedComponentsWithStats(difference_segmentation[i][:, :, 0])
        differences_intersection = np.logical_and(difference_segmentation[i], difference_segmentation[i - 1])
        current_labels_and_part = current_labels[differences_intersection[:,:,0]]
        previous_labels_and_part = previous_labels[differences_intersection[:,:,0]]
        current_connection = set(list(zip(previous_labels_and_part, current_labels_and_part)))






        intersections_map = difference_segmentation[i].copy()
        intersections_map[differences_intersection[:, :, 0]] = [255,0,0]
        # intersections_map[np.logical_and(differences_intersection[:, :, 0],mostly_white[:,:,0])] = [0,255,0]
        # intersections_map[differences_intersection_previous[:, :, 0]] = [0,255,0]
        differences_intersection_previous = differences_intersection

        current_labels_to_show = current_labels.copy()
        current_labels_to_show[differences_intersection[:, :, 0]] = 0
        previous_labels_to_show = previous_labels.copy()
        previous_labels_to_show[differences_intersection[:, :, 0]] = 0

        cv2.imshow("curr", current_labels_to_show*255/current_labels_to_show.max())
        cv2.imshow("prev", previous_labels_to_show*255/previous_labels_to_show.max())
        cv2.imshow("intersections_map", intersections_map)
        cv2.imshow("mostly_whites", mostly_whites[i])
        cv2.waitKey()
        previous_labels = current_labels
        previous_centroids = current_centroids
    #     connections_tmp = []
    #     added = [False for _ in range(len(connections))]
    #     for new_conn in current_connection:
    #         found = False
    #
    #         for z, conn in enumerate(connections):
    #             if new_conn[0] == conn[-1] and not added[z]:
    #                 connections[z].append(new_conn[1])
    #                 # connections_centroids[z].append()
    #                 found = True
    #                 added[z] = True
    #                 break
    #         if not found:
    #             connections_tmp.append(list(new_conn))
    #     for z, conn in enumerate(connections):
    #         if not added[z]:
    #             conn.append(0)
    #     connections += connections_tmp
    #
    #     curr_labs.append(current_labels)
    #     curr_centroids.append(current_centroids)
    #     previous_labels = current_labels
    #     previous_centroids = current_centroids
    #
    # for c in connections:
    #     while len(c) < len(frames):
    #         c.insert(0, 0)
    # previous_frame_centroids = None
    # for i, frame in enumerate(frames):
    #     if i == 179:
    #         print()
    #     current_frame_centroids = []
    #     connections_map = np.zeros_like(curr_labs[i])
    #     for j, l in enumerate(connections):
    #         if l[i] != 0:
    #             connections_map[curr_labs[i] == l[i]] = l[i]
    #             if l[i - 1] !=0:
    #                 current_frame_centroids.append((curr_centroids[i-1][l[i-1]], curr_centroids[i][l[i]]))
    #
    #     connections_only = frames[i].copy()
    #     connections_only[connections_map == 0] = 0
    #     for centroidss in current_frame_centroids:
    #         a = int(centroidss[0][0]), int(centroidss[0][1])
    #         b = int(centroidss[1][0]), int(centroidss[1][1])
    #         dist = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    #         if True:#dist < 20:
    #             cv2.line(connections_only, a, b, (255, 0, 0), 1)
    #     print(i)
    #     cv2.imshow("", connections_only)
    #     cv2.waitKey()
