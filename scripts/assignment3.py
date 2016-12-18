from __future__ import division
import math
import cv2
from random import randint
import numpy as np
from matplotlib import pyplot as plt
import os

from numpy.core.umath import absolute


def depth_map(image1, image2, window_size):
    left_image = cv2.imread(image1, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    right_image = cv2.imread(image2, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    final_depth_image = left_image.copy()
    border_size = window_size // 2


    bordered_left_image = cv2.copyMakeBorder(left_image, border_size, border_size, border_size, border_size,
                                             cv2.BORDER_CONSTANT, value=0)
    bordered_right_image = cv2.copyMakeBorder(right_image, border_size, border_size, border_size, border_size,
                                              cv2.BORDER_CONSTANT, value=0)
    depth_image = bordered_left_image.copy()

    for row in xrange(border_size, bordered_left_image.shape[0] - border_size):
        for column in xrange(border_size, bordered_left_image.shape[1] - border_size):
            ssd = 0
            minimum_ssd = 999999999999999999
            corresponding_centered_row_index = 0
            corresponding_centered_column_index = 0
            for column_right in xrange(border_size, bordered_right_image.shape[1] - border_size):
                if window_size == 3:
                    ssd = (bordered_left_image[row - 1, column - 1] - bordered_right_image[row - 1, column_right - 1]) ** 2 \
                          + (bordered_left_image[row - 1, column] - bordered_right_image[row - 1, column_right]) ** 2 \
                          + (bordered_left_image[row - 1, column + 1] - bordered_right_image[row - 1, column_right + 1]) ** 2 \
                          + (bordered_left_image[row, column - 1] - bordered_right_image[row, column_right - 1]) ** 2 \
                          + (bordered_left_image[row, column] - bordered_right_image[row, column_right]) ** 2 \
                          + (bordered_left_image[row, column + 1] - bordered_right_image[row, column_right + 1]) ** 2 \
                          + (bordered_left_image[row + 1, column - 1] - bordered_right_image[row + 1, column_right - 1]) ** 2 \
                          + (bordered_left_image[row + 1, column] - bordered_right_image[row + 1, column_right]) ** 2 \
                          + (bordered_left_image[row + 1, column + 1] - bordered_right_image[row + 1, column_right + 1]) ** 2
                elif window_size == 7:
                    ssd = (bordered_left_image[row - 3, column - 3] - bordered_right_image[row - 3, column_right - 3]) ** 2 \
                          + (bordered_left_image[row - 3, column - 2] - bordered_right_image[row - 3, column_right - 2]) ** 2 \
                          + (bordered_left_image[row - 3, column - 1] - bordered_right_image[row - 3, column_right - 1]) ** 2 \
                          + (bordered_left_image[row - 3, column] - bordered_right_image[row - 3, column_right]) ** 2 \
                          + (bordered_left_image[row - 3, column + 1] - bordered_right_image[row - 3, column_right + 1]) ** 2 \
                          + (bordered_left_image[row - 3, column + 2] - bordered_right_image[row - 3, column_right + 2]) ** 2 \
                          + (bordered_left_image[row - 3, column + 3] - bordered_right_image[row - 3, column_right + 3]) ** 2 \
                          + (bordered_left_image[row - 2, column - 3] - bordered_right_image[row - 2, column_right - 3]) ** 2 \
                          + (bordered_left_image[row - 2, column - 2] - bordered_right_image[row - 2, column_right - 2]) ** 2 \
                          + (bordered_left_image[row - 2, column - 1] - bordered_right_image[row - 2, column_right - 1]) ** 2 \
                          + (bordered_left_image[row - 2, column] - bordered_right_image[row - 2, column_right]) ** 2 \
                          + (bordered_left_image[row - 2, column + 1] - bordered_right_image[row - 2, column_right + 1]) ** 2 \
                          + (bordered_left_image[row - 2, column + 2] - bordered_right_image[row - 2, column_right + 2]) ** 2 \
                          + (bordered_left_image[row - 2, column + 3] - bordered_right_image[row - 2, column_right + 3]) ** 2 \
                          + (bordered_left_image[row - 1, column - 3] - bordered_right_image[row - 1, column_right - 3]) ** 2 \
                          + (bordered_left_image[row - 1, column - 2] - bordered_right_image[row - 1, column_right - 2]) ** 2 \
                          + (bordered_left_image[row - 1, column - 1] - bordered_right_image[row - 1, column_right - 1]) ** 2 \
                          + (bordered_left_image[row - 1, column] - bordered_right_image[row - 1, column_right]) ** 2 \
                          + (bordered_left_image[row - 1, column + 1] - bordered_right_image[row - 1, column_right + 1]) ** 2 \
                          + (bordered_left_image[row - 1, column + 2] - bordered_right_image[row - 1, column_right + 2]) ** 2 \
                          + (bordered_left_image[row - 1, column + 3] - bordered_right_image[row - 1, column_right + 3]) ** 2 \
                          + (bordered_left_image[row, column - 3] - bordered_right_image[row, column_right - 3]) ** 2 \
                          + (bordered_left_image[row, column - 2] - bordered_right_image[row, column_right - 2]) ** 2 \
                          + (bordered_left_image[row, column - 1] - bordered_right_image[row, column_right - 1]) ** 2 \
                          + (bordered_left_image[row, column] - bordered_right_image[row, column_right]) ** 2 \
                          + (bordered_left_image[row, column + 1] - bordered_right_image[row, column_right + 1]) ** 2 \
                          + (bordered_left_image[row, column + 2] - bordered_right_image[row, column_right + 2]) ** 2 \
                          + (bordered_left_image[row, column + 3] - bordered_right_image[row, column_right + 3]) ** 2 \
                          + (bordered_left_image[row + 1, column - 3] - bordered_right_image[row + 1, column_right - 3]) ** 2 \
                          + (bordered_left_image[row + 1, column - 2] - bordered_right_image[row + 1, column_right - 2]) ** 2 \
                          + (bordered_left_image[row + 1, column - 1] - bordered_right_image[row + 1, column_right - 1]) ** 2 \
                          + (bordered_left_image[row + 1, column] - bordered_right_image[row + 1, column_right]) ** 2 \
                          + (bordered_left_image[row + 1, column + 1] - bordered_right_image[row + 1, column_right + 1]) ** 2 \
                          + (bordered_left_image[row + 1, column + 2] - bordered_right_image[row + 1, column_right + 2]) ** 2 \
                          + (bordered_left_image[row + 1, column + 3] - bordered_right_image[row + 1, column_right + 3]) ** 2 \
                          + (bordered_left_image[row + 2, column - 3] - bordered_right_image[row + 2, column_right - 3]) ** 2 \
                          + (bordered_left_image[row + 2, column - 2] - bordered_right_image[row + 2, column_right - 2]) ** 2 \
                          + (bordered_left_image[row + 2, column - 1] - bordered_right_image[row + 2, column_right - 1]) ** 2 \
                          + (bordered_left_image[row + 2, column] - bordered_right_image[row + 2, column_right]) ** 2 \
                          + (bordered_left_image[row + 2, column + 1] - bordered_right_image[row + 2, column_right + 1]) ** 2 \
                          + (bordered_left_image[row + 2, column + 2] - bordered_right_image[row + 2, column_right + 2]) ** 2 \
                          + (bordered_left_image[row + 2, column + 3] - bordered_right_image[row + 2, column_right + 3]) ** 2 \
                          + (bordered_left_image[row + 3, column - 3] - bordered_right_image[row + 3, column_right - 3]) ** 2 \
                          + (bordered_left_image[row + 3, column - 2] - bordered_right_image[row + 3, column_right - 2]) ** 2 \
                          + (bordered_left_image[row + 3, column - 1] - bordered_right_image[row + 3, column_right - 1]) ** 2 \
                          + (bordered_left_image[row + 3, column] - bordered_right_image[row + 3, column_right]) ** 2 \
                          + (bordered_left_image[row + 3, column + 1] - bordered_right_image[row + 3, column_right + 1]) ** 2 \
                          + (bordered_left_image[row + 3, column + 2] - bordered_right_image[row + 3, column_right + 2]) ** 2 \
                          + (bordered_left_image[row + 3, column + 3] - bordered_right_image[row + 3, column_right + 3]) ** 2
                else:
                    print "Sorry, only window sizes of 3 or 7 are supported"
                if ssd < minimum_ssd:
                    minimum_ssd = ssd
                    corresponding_centered_row_index = row
                    corresponding_centered_column_index = column_right
            disparity = corresponding_centered_column_index - column
            if disparity != 0:
                depth = 1 - (1//disparity)
                # if depth > 255:
                #     depth = 255
                # elif depth < 0:
                #     depth = 0
            else:
                depth = 0

            depth_image[row, column] = depth
            final_depth_image[row - 3, column - 3] = depth
    return final_depth_image
    # cv2.imwrite('..\\results\\depth_image.png', depth_image)
    # np.set_printoptions(threshold=np.nan)
    # f = open("..\\results\\Depth_3.txt", 'w')
    # f.write(str(depth_image))


def median_filter_9(image):
    output_image = image.copy()
    bordered_input = cv2.copyMakeBorder(image, 4, 4, 4, 4,
                                             cv2.BORDER_CONSTANT, value=0)

    for row in xrange(4, bordered_input.shape[0] - 4):
        for column in xrange(4, bordered_input.shape[1] - 4):
            median_sorting_list = []
            for row_window in xrange(row - 4, row + 4):
                for column_window in xrange(column - 4, column + 4):
                    median_sorting_list.append(bordered_input[row_window, column_window])

            median_sorting_list.sort()
            output_image[row - 4, column - 4] = median_sorting_list[40]

    return output_image

image1 = "..\\images\\image1.png"
image2 = "..\\images\\image2.png"

depth_3 = depth_map(image1, image2, 3)
output3 = median_filter_9(depth_3)
cv2.imwrite('..\\results\\Depth_3.jpg', output3)
np.set_printoptions(threshold=np.nan)
f = open("..\\report\\Depth_3.txt", 'w')
f.write(str(output3))

depth_7 = depth_map(image1, image2, 7)
output7 = median_filter_9(depth_7)
cv2.imwrite('..\\results\\Depth_7.jpg', output7)
np.set_printoptions(threshold=np.nan)
f = open("..\\report\\Depth_7.txt", 'w')
f.write(str(output7))
