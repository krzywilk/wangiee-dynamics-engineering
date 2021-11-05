import cv2

from detections.player_detection import framewise_difference_segmentation, two_players_blob_detection, \
    player_backwards_bbox_tracking
from detections.table_detection import table_detection, detect_intersection_points
from video_utils import load_video, draw_players_bboxes, draw_table_contours

if __name__ == '__main__':
    test_sample_1 = "videos/sample_1.mp4"

    frames = load_video(test_sample_1)
    print("table detection")
    table_mask = table_detection(frames)
    table_intersection_points, centroid = detect_intersection_points(table_mask)
    print("players detection")
    difference_segmentation = framewise_difference_segmentation(frames)
    players_detections, diff_vals = two_players_blob_detection(difference_segmentation)

    processed_frames = draw_players_bboxes(frames, players_detections, diff_vals)
    processed_frames = draw_table_contours(processed_frames, table_intersection_points)
    for i, frame in enumerate(processed_frames):
        cv2.imshow("detections", frame)
        cv2.imshow("difference_segmentation", difference_segmentation[i])
        cv2.waitKey()
