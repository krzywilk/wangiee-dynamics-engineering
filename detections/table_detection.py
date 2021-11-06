import math
import numpy as np
import cv2
from collections import defaultdict
from sklearn import preprocessing
from sklearn.cluster import KMeans, DBSCAN
import cv2
import numpy as np


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


def table_rectangle(pts):
    rect = np.zeros((4, 2), dtype="float32")
    net = np.zeros((2, 2), dtype="float32")
    s = pts.sum(axis=1)
    left_top_ind = np.argmin(s)
    rect[0] = pts[left_top_ind]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    left_bottom_ind = np.argmax(diff)
    rect[3] = pts[left_bottom_ind]
    net = pts[np.logical_and(pts[:, 1] > rect[0][1], pts[:, 1] < rect[3][1])]  ## to improve
    centroid = rect.mean(axis=0)
    return rect, centroid


def table_detection(frames):
    table = np.zeros_like(frames[0][:, :, 0])
    for i, frame in enumerate(frames):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_frame, 150, 170, apertureSize=3)
        lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]),
                                minLineLength=100, maxLineGap=5)
        a, b, c = lines.shape
        for i in range(a):
            cv2.line(table, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255), 3,
                     cv2.LINE_AA)
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(table)
    stats = stats[1:]
    centroids = centroids[1:]
    best = -1, float('inf')
    for i, c in enumerate(centroids):
        if best[1] > math.hypot(c[0] - frame.shape[1] // 2, c[1] - frame.shape[0] // 2):
            best = i, math.hypot(c[0] - frame.shape[1] // 2, c[1] - frame.shape[0] // 2)
    x = stats[best[0], cv2.CC_STAT_LEFT]
    y = stats[best[0], cv2.CC_STAT_TOP]
    w = stats[best[0], cv2.CC_STAT_WIDTH]
    h = stats[best[0], cv2.CC_STAT_HEIGHT]
    res = np.zeros_like(table)

    res[y: y + h, x: x + w] = table[y: y + h, x: x + w]

    return res


def detect_intersection_points(table):
    B = np.argwhere(table)
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
    table_trim = table[ystart:ystop, xstart:xstop]
    table_trim = cv2.erode(table_trim, np.ones((3, 3)), iterations=2)
    table_trim = cv2.dilate(table_trim, np.ones((1, 3)), iterations=2)
    lines = cv2.HoughLines(table_trim, 2, np.pi / 180, 200)
    lines = filter_close_lines(lines)
    line_clusters = cluster_lines(lines)
    intersections = find_intersections(line_clusters)
    centroids = find_intersections_centroids(intersections, max(table_trim.shape) * 0.033)
    intersections, centroids_sparse_matrix = organize_intersection_centroids(centroids, max(
        table_trim.shape) * 0.033)
    intersections, center = table_rectangle(intersections)
    for i, point in enumerate(intersections):
        intersections[i] = [point[0]+xstart, point[1]+ystart]
    # for l in lines:
    #     for rho, theta in l:
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         x1 = int(x0 + 1000 * (-b))
    #         y1 = int(y0 + 1000 * (a))
    #         x2 = int(x0 - 1000 * (-b))
    #         y2 = int(y0 - 1000 * (a))
    # # tt = np.zeros_like(table_trim)
    # # for inte in intersections:
    # #     table_trim=cv2.circle(table_trim, (int(inte[0]), int(inte[1])), 8, 255, -1)
    # cv2.imshow("asd", table_trim)
    # cv2.waitKey()
    return intersections, center
