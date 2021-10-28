import cv2


def load_video(path):
    res = []
    cap = cv2.VideoCapture(path)
    ret = True
    while ret:
        ret, frame = cap.read()
        if ret:
            res.append(frame)
    return res


def draw_players_bboxes(frames, players_detections):
    result_frames = []
    for i,f in enumerate(frames):
        tmp_frame = f.copy()
        for player_object in players_detections[i]:
            if player_object is None:
                continue
            stats, centroid = player_object
            x = stats[cv2.CC_STAT_LEFT]
            y = stats[cv2.CC_STAT_TOP]
            w = stats[cv2.CC_STAT_WIDTH]
            h = stats[cv2.CC_STAT_HEIGHT]
            area = stats[cv2.CC_STAT_AREA]
            (cX, cY) = centroid
            tmp_frame = cv2.rectangle(tmp_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        result_frames.append(tmp_frame)
    return result_frames