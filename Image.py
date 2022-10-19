import cv2
import numpy as np

from matplotlib import pyplot as plt

# Image.find_order(contours)
from bisect import bisect

# Image.find_paths(contours)
from scipy.spatial.distance import cdist
from collections import defaultdict

from config import Config
from canny_edge_detector import CannyEdgeDetector


class Image(object):
    def __init__(self, image_location, shape=None, background=True):

        self.image = cv2.imread(image_location)

        if Config.CANNY_EDGE_DETECTOR:
            self.image = self.rgb2gray(self.image)
            if shape:
                ratio = max(self.image.shape) / max(shape)
                new_shape = tuple(int(ti / ratio) for ti in self.image.shape)
                self.image = cv2.resize(self.image, new_shape)
            if background:
                self.background_image = cv2.imread(Config.BACKGROUND_IMAGE)
                self.background_image = cv2.resize(self.background_image, new_shape)

        else:
            if shape:
                self.image = cv2.resize(self.image, shape)

    @staticmethod
    def rgb2gray(rgb):

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        return gray

    def get_background_image(self):
        return self.background_image

    def sort(self):
        contours = self.find_contours()
        if Config.DROP_CONTOURS:
            contours = tuple(
                contour for contour in contours if contour.shape[0] > self.image.size * Config.CONTOURS_THRESHOLD)

        show_contours = True
        if show_contours:
            # create an empty image for contours
            for c in range(len(contours)):
                img_contours = np.zeros(self.image.shape)
                # draw the contours on the empty image
                cv2.drawContours(img_contours, contours, c, (255, 255, 255), 1)
                plt.imshow(img_contours)
            plt.close()

        acc_list = []
        for idx, (start, end, stride) in self.find_order(contours):
            if start is None and end is None and stride == -1:
                acc_list.append(contours[idx][start::-1])
            elif start is None and end is None and stride == 1:
                acc_list.append(contours[idx])
            elif start is None and end is not None and stride == -1:
                acc_list.append(contours[idx][:end: -1])
            else:
                acc_list.append(contours[idx][start:end: stride])
        return np.vstack(acc_list)
        # return np.vstack([contours[idx][start::-1] if start is None and end is None and stride == -1
        #                   else contours[idx] if start is None and end is None and stride == 1
        # else contours[idx][:end:-1] if start is None and end is not None and stride == -1
        # else contours[idx][start:end:stride] for idx, (start, end, stride) in self.find_order(contours)])

    def find_contours(self):
        # https://wttech.blog/blog/2022/edge-detection-and-processing-using-canny-edge-detector-and-hough-transform/
        image = self.image

        if Config.CANNY_EDGE_DETECTOR:

            detector = CannyEdgeDetector(self.image, sigma=1.4, kernel_size=5, lowthreshold=0.09,
                                         highthreshold=0.17,
                                         weak_pixel=100)
            edges = detector.detect()

            edges = edges.astype(np.uint8)
        else:
            image = cv2.GaussianBlur(image, (5, 5), 0)
            edges = cv2.Canny(image, 100, 255)

        if Config.ERODE:
            kernel = np.ones((3, 3), np.uint8)
            image_erosion = cv2.erode(255 - edges, kernel, iterations=1)
            # plt.imshow(image_erosion)
            image_dilation = cv2.dilate(image_erosion, kernel, iterations=1)
            # plt.imshow(image_dilation)
            erode_dilate_edges = 255 - image_dilation

        ret, thresh = cv2.threshold(erode_dilate_edges, 127, 255, 0)
        contours, __ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # CHAIN_APPROX_NONE

        return contours

    def find_order(self, contours):
        # This function was written as recursive originally.
        # This function obtains a dictionary of connections from find_paths(contours)
        # and "recursively" goes through the dictionary to find the slice notations that connects all contours together.
        order = []
        stack = [(0, 0, 0)]
        paths = self.find_paths(contours)

        while stack:
            cur_contour, cur_pos, original_pos = stack.pop(-1)
            if paths[cur_contour]:
                pos = bisect([start for _, (start, _) in paths[cur_contour]], cur_pos)
                # Check connections to the left and then to the right
                next_contour, (start, end) = paths[cur_contour].pop(pos - 1 if pos > 0 else 0)
                # Order imitates slicing notation
                # For example, (cur_pos, start+1, 1) indicates a slice of cur_pos:start+1:1
                order.append((cur_contour, (cur_pos, start + 1, 1) if start + 1 > cur_pos else (
                    cur_pos, start - 1 if start > 0 else None, -1)))
                stack.append((cur_contour, start, original_pos))
                if next_contour in paths:
                    stack.append((next_contour, end, end))
                else:
                    order.append((next_contour, (end, None, -1)))
                    order.append((next_contour, (None, None, 1)))
                    order.append((next_contour, (None, end - 1 if end > 0 else None, -1)))
            else:
                order.append((cur_contour, (cur_pos, None, 1)))
                order.append((cur_contour, (None, original_pos - 1 if original_pos > 0 else None, -1)))

        return order

    def find_paths(self, contours):
        # This function first gets a distance matrix from cdist(points, points)
        # Then consider a "blob" that contains contours[0] (all the points of contours[0])
        # This function then uses that distance matrix to find the closest point to blob
        # And then adding said closest point into the blob because it is now connected
        # And then ignoring said closest point's distance to the blob and vice versa by setting the distance in the distance matrix to np.inf.
        # Finally construct a dictionary of connections.
        points = np.vstack(contours)
        points = points.reshape((points.shape[0], 2))
        dist = cdist(points, points)

        len_arr = np.array([len(contour) for contour in contours], dtype=np.int_)
        end_points = np.add.accumulate(len_arr)

        start = 0
        start_end = []
        for end in end_points:
            dist[start:end:, start:end:] = np.inf
            start_end.append((start, end))
            start = end

        paths = defaultdict(list)
        # temp_order keeps track of the order in temp_dist
        # temp_start_end keeps track of the starts and ends of each contour in temp_dist
        # temp_dist is a slice (in terms of rows) of the original distance matrix, mainly to reduce np.argmin calculations.
        temp_order = [0]
        temp_start_end = [start_end[0]]
        temp_dist = dist[start_end[0][0]:start_end[0][1]]

        # The first connection connects two contours, and the rest connects only one contour
        while len(temp_order) < end_points.size:

            row_min = np.argmin(temp_dist, axis=0)
            cols = np.indices(row_min.shape)
            col_min = np.argmin(temp_dist[row_min, cols])

            # row_min[col_min] gives the row min of temp_dist
            temp_row, temp_col = row_min[col_min], col_min
            temp_cur_contour = self.find_contour_index(temp_row, temp_start_end)
            cur_contour = temp_order[temp_cur_contour]
            # express row in terms of the index inside contours[cur_contour]
            row = temp_row - temp_start_end[temp_cur_contour][0]
            next_contour = self.find_contour_index(temp_col, start_end)
            col = temp_col - start_end[next_contour][0]

            paths[cur_contour].append((next_contour, (row, col)))
            # Ignore the distance from connected points to other connected points
            start, end = start_end[next_contour]
            for order in temp_order:
                new_start, new_end = start_end[order]
                dist[new_start:new_end:, start:end:] = np.inf
                dist[start:end:, new_start:new_end:] = np.inf

            temp_order.append(next_contour)
            temp_len_arr = np.array([len(contours[order]) for order in temp_order], dtype=np.int_)
            temp_end_points = np.add.accumulate(temp_len_arr)
            temp_start_end.append((temp_start_end[-1][-1], temp_start_end[-1][-1] + temp_len_arr[-1]))
            temp_dist = dist[np.hstack([np.arange(start_end[order][0], start_end[order][1]) for order in temp_order])]

        for contour in paths:
            paths[contour].sort(key=lambda x: x[1][0])
        return paths

    def find_contour_index(self, idx, start_end):
        for i, (start, end) in enumerate(start_end):
            if start <= idx < end:
                return i
        return len(start_end) - 1
