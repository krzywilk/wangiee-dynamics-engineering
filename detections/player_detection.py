import cv2
import os
import numpy as np
import math
import heapq

from bbox_utils import bb_intersection_over_union, bb_intersection


def framewise_difference_segmentation(frames, frame_delta_threshold=10, opening_erosion_kernel=np.ones((3, 3)),
                                      opening_dilation_kernel=np.ones((10, 6))):
    previous_frame = np.ones_like(frames[0])
    result_masks = []
    for i, frame in enumerate(frames):
        frame_delta = cv2.absdiff(previous_frame, frame)
        frame_delta[frame_delta < frame_delta_threshold] = 0
        mask = np.zeros_like(frames[0])
        frame_delta_single_channel = cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY)
        mask[frame_delta_single_channel > 0] = 255
        mask = cv2.erode(mask, opening_erosion_kernel, iterations=1)
        mask = cv2.dilate(mask, opening_dilation_kernel, iterations=1)

        result_masks.append(mask)
        previous_frame = frame
    return result_masks

def two_players_blob_detection(difference_segmentation_masks, table_intersection_points):
    result = []
    players_masks = []
    for i, mask in enumerate(difference_segmentation_masks):
        if i == 0:
            continue
        players_mask = np.zeros_like(mask[:,:,0])
        output_bottom = cv2.connectedComponentsWithStats(
            mask[int(max(table_intersection_points[0][1], table_intersection_points[1][1])):, :, 0])
        (numLabels, labels, stats, centroids) = output_bottom
        bottom_player = [x[0] + 1 for x in
                         heapq.nlargest(1, enumerate(stats[1:, cv2.CC_STAT_AREA]), key=lambda x: x[1])]
        labels[labels!=bottom_player[0]] = 0
        # cv2.imshow("",mask[int(max(table_intersection_points[0][1], table_intersection_points[1][1])):, :, 0])
        # cv2.imshow("labels",labels/labels.max())

        players_mask[int(max(table_intersection_points[0][1], table_intersection_points[1][1])):] = labels
        # cv2.imshow("labels2", players_mask / players_mask.max())
        # cv2.waitKey()
        bottom_player = (stats[bottom_player[0]], centroids[bottom_player[0]])
        bottom_player[0][cv2.CC_STAT_TOP] = bottom_player[0][cv2.CC_STAT_TOP] + int(
            max(table_intersection_points[0][1], table_intersection_points[1][1]))
        bottom_player[1][1] = bottom_player[1][1] + int(
            max(table_intersection_points[0][1], table_intersection_points[1][1]))


        output_top = cv2.connectedComponentsWithStats(
            mask[:abs(
                int(max(table_intersection_points[0][1], table_intersection_points[1][1])) - int(
                    max(table_intersection_points[2][1], table_intersection_points[3][1]))
            ) // 2 + int(max(table_intersection_points[0][1], table_intersection_points[1][1])), :, 0])
        (numLabels, labels, stats, centroids) = output_top
        top_player = [x[0] + 1 for x in
                      heapq.nlargest(1, enumerate(stats[1:, cv2.CC_STAT_AREA]), key=lambda x: x[1])]
        labels[labels != top_player[0]] = 0
        players_mask[:abs(
                int(max(table_intersection_points[0][1], table_intersection_points[1][1])) - int(
                    max(table_intersection_points[2][1], table_intersection_points[3][1]))
            ) // 2 + int(max(table_intersection_points[0][1], table_intersection_points[1][1]))][labels == top_player[0]] = top_player[0]
        top_player = (stats[top_player[0]], centroids[top_player[0]])
        result.append((top_player, bottom_player))

        players_masks.append(players_mask)

    return [result[0]] + result, [players_masks[0]]+players_masks


def player_backwards_bbox_tracking(players_detections, frames):  # TODO: do not work, main idea only
    previous_stats, previous_centroid = players_detections[0][1]
    for i, player_detection in zip(range(1, len(players_detections)), players_detections[1:]):
        current_stats, current_centroid = player_detection[1]
        x_current, y_current, w_current, h_current = current_stats[cv2.CC_STAT_LEFT], current_stats[cv2.CC_STAT_TOP], \
                                                     current_stats[cv2.CC_STAT_WIDTH], current_stats[cv2.CC_STAT_HEIGHT]
        x_previous, y_previous, w_previous, h_previous = previous_stats[cv2.CC_STAT_LEFT], previous_stats[
            cv2.CC_STAT_TOP], \
                                                         previous_stats[cv2.CC_STAT_WIDTH], previous_stats[
                                                             cv2.CC_STAT_HEIGHT]
        iou = bb_intersection_over_union([x_current, y_current, x_current + w_current, y_current + h_current],
                                         [x_previous, y_previous, x_previous + w_previous, y_previous + h_previous])
        intersection = bb_intersection([x_current, y_current, x_current + w_current, y_current + h_current],
                                       [x_previous, y_previous, x_previous + w_previous, y_previous + h_previous])

        current_not_inside_previous = abs(w_current * h_current - intersection) / max(w_current * h_current,
                                                                                      intersection)
        if not (current_not_inside_previous < 0.1 and iou < 0.5):

            previous_stats, previous_centroid = player_detection[1]
        else:
            print("smoth")
            x_current, y_current, w_current, h_current = previous_stats[cv2.CC_STAT_LEFT], previous_stats[
                cv2.CC_STAT_TOP], \
                                                         previous_stats[cv2.CC_STAT_WIDTH], previous_stats[
                                                             cv2.CC_STAT_HEIGHT]

        tmp_frame = frames[i].copy()
        tmp_frame = cv2.rectangle(tmp_frame, (x_current, y_current), (x_current + w_current, y_current + h_current),
                                  (0, 255, 0), 3)
        # tmp_frame = cv2.rectangle(tmp_frame, (x_previous, y_previous), (x_previous + w_previous, y_previous + h_previous), (255, 0, 0), 3)
        cv2.imshow("", tmp_frame)
        print(iou)
        cv2.waitKey()
