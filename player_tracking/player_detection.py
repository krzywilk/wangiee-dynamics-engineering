import cv2
import os
import numpy as np
import math
import heapq


def framewise_difference_segmentation(frames, frame_delta_threshold=5, opening_erosion_kernel = np.ones((3, 3)), opening_dilation_kernel= np.ones((10, 6))):
    previous_frame = np.ones_like(frames[0])
    result_masks = []
    for i, frame in enumerate(frames):
        frame_delta = cv2.absdiff(previous_frame, frame)
        frame_delta[frame_delta < frame_delta_threshold] = 0
        mask = np.zeros_like(frames[0])
        frame_delta_single_channel = frame_delta[:, :, 0] + frame_delta[:, :, 1] + frame_delta[:, :, 2]
        mask[frame_delta_single_channel > 0] = 255
        mask = cv2.erode(mask, opening_erosion_kernel, iterations=1)
        mask = cv2.dilate(mask,opening_dilation_kernel, iterations=1)
        result_masks.append(mask)
        previous_frame = frame
    return result_masks


def blob_2_players_detection(difference_segmentation_masks, ):
    result = []
    for i, mask in enumerate(difference_segmentation_masks):
        output = cv2.connectedComponentsWithStats(
            mask[:, :, 0])

        (numLabels, labels, stats, centroids) = output
        players_indexes = [x[0]+1 for x in heapq.nlargest(2, enumerate(stats[1:, cv2.CC_STAT_AREA]), key=lambda x: x[1])]
        if len(players_indexes) > 1:
            if centroids[players_indexes[0]][1] < centroids[players_indexes[1]][1]:
                top_player = (stats[players_indexes[0]], centroids[players_indexes[0]])
                bottom_player = (stats[players_indexes[1]], centroids[players_indexes[1]])
            else:
                top_player = (stats[players_indexes[1]], centroids[players_indexes[1]])
                bottom_player = (stats[players_indexes[0]], centroids[players_indexes[0]])
            result.append((top_player, bottom_player))
        else:
            result.append(((stats[players_indexes[0]], centroids[players_indexes[0]]), None))

    return result

