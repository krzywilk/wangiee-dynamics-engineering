import cv2

from player_tracking.player_detection import framewise_difference_segmentation, blob_2_players_detection
from video_utils import load_video, draw_players_bboxes

if __name__ == '__main__':
    test_sample_1 = "videos/sample_1.mp4"

    frames = load_video(test_sample_1)
    print("loaded")
    difference_segmentation = framewise_difference_segmentation(frames)
    players_detections = blob_2_players_detection(difference_segmentation)
    processed_frames = draw_players_bboxes(frames, players_detections)
    for frame in processed_frames:
        cv2.imshow("players detection",frame)
        cv2.waitKey()
