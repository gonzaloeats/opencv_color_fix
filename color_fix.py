#!/usr/bin/env python
import cv2 
import math
import numpy as np
import sys
from shutil import copyfile
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

    half_percent = percent / 100.0 # depending on the pod a range of 10.0-25.0 returns a pretty good result

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
        # normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 200, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    return cv2.merge(out_channels)

#def backup_img(img):
#	return copyfile(img, img[:-4] + "work_orig")

	
if __name__ == '__main__':
    img = cv2.imread(sys.argv[1]) # add parameter for image src, example: python color_fix "c:\path\path\path.jpg"
    #print img
    perc_val = int(raw_input("color value adj. perameter 8<x<20: "))
    #orig = simplest_cb(img,1) # no transformation
    copyfile(str(sys.argv[1]), str(sys.argv[1])[:-4] + "_original.bmp") #backup_img(img)
    out = simplest_cb(img, perc_val) # value changes intesity 8 to 20 seems to be a good range
    #cv2.imshow("before", img) # commented out for running in batch
    #cv2.imshow("after", out) # commented out for running in batch
    #cv2.imwrite(str(sys.argv[1])[:-4] + "_original.bmp",orig) # removes the last 4 characters of the path name
    cv2.imwrite(str(sys.argv[1])[:-4] + ".bmp",out) # removes the last 4 characters of the path name
                                                          # and appends _original.bmp ~ makes original copy 
                                                          # and appends _fixed.jpg 													  
    #cv2.waitKey(0) # commented out for running in batch
