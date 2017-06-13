#!/usr/bin/env python
import cv2 
import math
import numpy as np
import sys

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent):
    assert img.shape[2] == 3
    assert percent > 0 and percent < 100

    half_percent = percent / 25.0 # not sure how this tuning parameter works but 25 give a pretty good result

    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high percentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        assert len(flat.shape) == 1

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val  = flat[int(math.floor(n_cols * half_percent))]
        high_val = flat[int(math.ceil(n_cols * (1.0 - half_percent)))]

        print "Lowval: ", low_val
        print "Highval: ", high_val

        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1]) # add parameter for image src, example: python color_fix "c:\path\path\path.jpg"
    out = simplest_cb(img, 1)
    # cv2.imshow("before", img) # commented out for running in batch
    # cv2.imshow("after", out) # commented out for running in batch
    cv2.imwrite(str(sys.argv[1])[:-4] + "_fixed.jpg",out) # removes the last 4 characters of the path name and appends
                                                          # and appends _fixed.jpg 
    # cv2.waitKey(0) # commented out for running in batch