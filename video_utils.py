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
        for top,player_object in enumerate(players_detections[i]):
            if player_object is None:
                continue
            stats, centroid = player_object
            x = stats[cv2.CC_STAT_LEFT]
            y = stats[cv2.CC_STAT_TOP]
            w = stats[cv2.CC_STAT_WIDTH]
            h = stats[cv2.CC_STAT_HEIGHT]
            area = stats[cv2.CC_STAT_AREA]
            cv2.rectangle(tmp_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # if top == 0:
            #     if diff_vals[i]:
            #         tmp_frame = cv2.rectangle(tmp_frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
            #     else:
            #         tmp_frame = cv2.rectangle(tmp_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # if top == 1:
            #     if not diff_vals[i]:
            #         tmp_frame = cv2.rectangle(tmp_frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
            #     else:
            #         tmp_frame = cv2.rectangle(tmp_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        result_frames.append(tmp_frame)
    return result_frames


def draw_table_contours(frames, table_intersection_points):
    result = []
    for frame in frames:
        tmp = frame.copy()
        for inte in table_intersection_points:
            cv2.circle(tmp, (int(inte[0]), int(inte[1])), 4, (0, 0, 255), -1)
        cv2.line(tmp, (int(table_intersection_points[0][0]), int(table_intersection_points[0][1])), (int(table_intersection_points[1][0]), int(table_intersection_points[1][1])), (0, 0, 255), 2)
        cv2.line(tmp, (int(table_intersection_points[1][0]), int(table_intersection_points[1][1])), (int(table_intersection_points[2][0]), int(table_intersection_points[2][1])), (0, 0, 255), 2)
        cv2.line(tmp, (int(table_intersection_points[2][0]), int(table_intersection_points[2][1])), (int(table_intersection_points[3][0]), int(table_intersection_points[3][1])), (0, 0, 255), 2)
        cv2.line(tmp, (int(table_intersection_points[3][0]), int(table_intersection_points[3][1])), (int(table_intersection_points[0][0]), int(table_intersection_points[0][1])), (0, 0, 255), 2)
        result.append(tmp)
    return result