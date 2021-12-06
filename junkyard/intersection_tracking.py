import math

from detections.player_detection import framewise_difference_segmentation, two_players_blob_detection
from detections.table_detection import detect_intersection_points, table_detection
from video_utils import load_video
import cv2
import numpy as np
import random


def cropped_iou(crop_image1, crop_image2):
    x_max = crop_image1.shape[0]
    if x_max < crop_image2.shape[0]:
        x_max = crop_image2.shape[0]
    y_max = crop_image1.shape[1]
    if y_max < crop_image2.shape[1]:
        y_max = crop_image2.shape[1]
    iou_tmp_img = np.zeros((x_max, y_max))
    iou_tmp_img[
    round((x_max - crop_image1.shape[0]) / 2):round((x_max - crop_image1.shape[0]) / 2) + crop_image1.shape[0],
    round((y_max - crop_image1.shape[1]) / 2):round((y_max - crop_image1.shape[1]) / 2) + crop_image1.shape[
        1]] = crop_image1.astype(bool)

    iou_tmp_img[
    round((x_max - crop_image2.shape[0]) / 2):round((x_max - crop_image2.shape[0]) / 2) + crop_image2.shape[0],
    round((y_max - crop_image2.shape[1]) / 2):round((y_max - crop_image2.shape[1]) / 2) + crop_image2.shape[
        1]] += crop_image2.astype(bool)
    return (iou_tmp_img[
            round((x_max - crop_image2.shape[0]) / 2):round((x_max - crop_image2.shape[0]) / 2) + crop_image2.shape[0],
            round((y_max - crop_image2.shape[1]) / 2):round((y_max - crop_image2.shape[1]) / 2) + crop_image2.shape[
                1]] == 2).sum() / (y_max * x_max)
from collections import defaultdict

if __name__ == "__main__":
    test_sample_1 = "../videos/sample_1.mp4"
    frames = load_video(test_sample_1)
    # for f in frames:
    #     hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    #     sensitivity = 110
    #     lower_white = np.array([0, 0, 255 - sensitivity])
    #     upper_white = np.array([255, sensitivity, 255])
    #     mask = cv2.inRange(hsv, lower_white, upper_white)
    #     res = cv2.bitwise_and(hsv,hsv, mask=mask)
    #     white_frame = f.copy()
    #     t, mostly_white = cv2.threshold(white_frame, 100, 255, cv2.THRESH_BINARY)
    #     white_frame[mostly_white.all(axis=2) == False] = 0
    #     cv2.imshow("", res)
    #     cv2.imshow("2", white_frame)
    #     cv2.waitKey()
    mostly_whites = []
    # for frame in frames:
    #     t, mostly_white = cv2.threshold(frame, 170, 255, cv2.THRESH_BINARY)
    #     white_frame = frame.copy()
    #     white_frame[mostly_white == 0] = 0
    #     mostly_whites.append(white_frame)
    table_mask = table_detection(frames)
    table_intersection_points, centroid = detect_intersection_points(table_mask)
    detector = cv2.SimpleBlobDetector()
    difference_segmentation = framewise_difference_segmentation(frames)
    whites_masks = []
    players_detections, players_masks = two_players_blob_detection(difference_segmentation, table_intersection_points)
    # BackgroundSubtractorMOG = cv2.bgsegm.createBackgroundSubtractorMOG()
    BackgroundSubtractorGMG = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=2, decisionThreshold=0.99)
    substr = []
    import time

    start = time.time()
    for i, frame in enumerate(frames):
        white_frame = frame.copy()
        t, mostly_white = cv2.threshold(white_frame, 100, 255, cv2.THRESH_BINARY)
        white_frame[mostly_white.all(axis=2) == False] = 0
        fore2 = BackgroundSubtractorGMG.apply(white_frame)
        substr.append((fore2))
        mostly_whites.append(white_frame)

    stop = time.time()
    print(stop - start)
    connections = []
    differences_intersection_previous = np.zeros_like(frames[0])
    _, previous_labels, _, previous_centroids = cv2.connectedComponentsWithStats(difference_segmentation[0][:, :, 0])
    curr_labs = [previous_labels]
    curr_centroids = [previous_centroids]
    prev_window = 4
    centroid_previous = []
    centroid_colors = [(int(random.random() * 256), int(random.random() * 256), int(random.random() * 256)) for _ in
                       range(prev_window)]
    previous_labels_number, previous_labels, previous_stats, previous_centroids = cv2.connectedComponentsWithStats(
        substr[- 1])
    chains = []
    for i, frame in enumerate(frames):
        if i == 0:
            continue

        both_subst = np.zeros_like(frame)
        both_subst[substr[i] > 0] = [0, 255, 0]
        both_subst[substr[i - 1] > 0] = [255, 0, 0]

        current_labels_number, current_labels, current_stats, current_centroids = cv2.connectedComponentsWithStats(
            substr[i])

        matched_centroids = []
        matched_inds = []
        prev_matches = defaultdict(list)
        all_matches = []



        for current_index, current_single_centroid in enumerate(current_centroids[1:], start=1):
            min_dist = 999999
            closest_centroid = [-1,-1]
            closest_index = -1
            bbox_current_blob = current_labels[
                                current_stats[current_index][1]:current_stats[current_index][1] +
                                                                current_stats[current_index][3],
                                current_stats[current_index][0]:current_stats[current_index][0] +
                                                                current_stats[current_index][2]]
            for previous_index, previous_single_centroid in enumerate(previous_centroids[1:], start=1):
                dist = math.hypot(current_single_centroid[0] - previous_single_centroid[0],
                                  current_single_centroid[1] - previous_single_centroid[1])
                bbox_prev_blob = previous_labels[
                                    previous_stats[previous_index][1]:previous_stats[previous_index][1] +
                                                                      previous_stats[previous_index][3],
                                    previous_stats[previous_index][0]:previous_stats[previous_index][0] +
                                                                      previous_stats[previous_index][2]]
                iou = cropped_iou(bbox_current_blob, bbox_prev_blob)
                if iou >0:
                    full_distance = (1/iou) * dist
                else:
                    full_distance = 999999
                if min_dist > full_distance:
                    closest_centroid = previous_single_centroid
                    min_dist = full_distance
                    closest_index = previous_index
            if closest_index!=-1:
                prev_matches[closest_index].append((current_single_centroid, closest_centroid, current_index, min_dist, closest_index))

        for key in prev_matches:
            closest = prev_matches[key][0]
            for element in prev_matches[key][1:]:
                if closest[3] > element[3]:
                    closest = element
            matched_centroids.append(closest)
            matched = False
            for chain in chains:
                if np.all(chain[-1][0] == closest[1]):
                    chain.append((closest[0], closest[2], closest[4]))
                    matched=True
                    break
            if not matched:
                chains.append([(closest[1], closest[2], closest[4]), (closest[0], closest[2], closest[4])])
        for start_point, end_point, curr_ind, _, prev_ind in matched_centroids:
            cv2.line(both_subst, (int(start_point[0]), int(start_point[1])),
                     (int(end_point[0]), int(end_point[1])), (255, 255, 255), 1)
            cv2.putText(both_subst, str(prev_ind),
                        (int(start_point[0]), int(start_point[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        1,
                        1)
            cv2.putText(both_subst, str(curr_ind),
                        (int(end_point[0]), int(end_point[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 255),
                        1,
                        1)

        for cent_number, centroid in enumerate(previous_centroids[1:], start=1):
            cv2.circle(both_subst, (int(centroid[0]), int(centroid[1])), 2, (0, 255, 255), -1)
            # cv2.circle(both_subst, (int(centroid[0]), int(centroid[1])), 20, (0, 255, 255), 1)
        for cent_number, centroid in enumerate(current_centroids[1:], start=1):
            cv2.circle(both_subst, (int(centroid[0]), int(centroid[1])), 2, (255, 0, 255), -1)
            # cv2.circle(both_subst, (int(centroid[0]), int(centroid[1])), 20, (255,0 , 255), 1)

        cv2.imshow("curr", frame)
        cv2.imshow("mostly_whites", mostly_whites[i])
        cv2.imshow("players_masks", players_masks[i])
        cv2.imshow("substr1", substr[i])
        cv2.imshow("both_subst", both_subst)
        previous_labels_number, previous_labels, previous_stats, previous_centroids = current_labels_number, current_labels, current_stats, current_centroids
        print(i)
        cv2.waitKey()
        if i == 61:
            print()

# bbox_curr_candidate = current_labels[
#                       current_stats[cosest_ind][1]:current_stats[cosest_ind][1] + current_stats[cosest_ind][3],
#                       current_stats[cosest_ind][0]:current_stats[cosest_ind][0] + current_stats[cosest_ind][2]]
# try:
#     bbox_prev_closest = previous_labels[
#                         previous_stats[matched_centroids[z][4]][1]:previous_stats[matched_centroids[z][4]][1] +
#                                                                    previous_stats[matched_centroids[z][4]][3],
#                         previous_stats[matched_centroids[z][4]][0]:previous_stats[matched_centroids[z][4]][0] +
#                                                                    previous_stats[matched_centroids[z][4]][2]]
# except:
#     print()
# bbox_prev = previous_labels[previous_stats[prev_ind][1]:previous_stats[prev_ind][1] + previous_stats[prev_ind][3],
#             previous_stats[prev_ind][0]:previous_stats[prev_ind][0] + previous_stats[prev_ind][2]]
