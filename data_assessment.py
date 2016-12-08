#Setup

#For image visualization
import matplotlib.pyplot as plt
#Image Processing
import os
import cv2
import numpy as np
import copy
import time
#Interactive Widgets
import ipywidgets
#from __future__ import print_function
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output


def to_matrix(l, n):
    """
    to_matrix creates a 2-d array from a list.

    argumens:
    l - 1-d list of pixels
    n - the number of rows & columns

    outputs:
    A 2-d array with n rows
    """
    return [l[i:i+n] for i in xrange(0, len(l), n)]

def flatten(arr):
    """
    flatten takes a 2-d list or array and flattens it into one 1-d list

    arguments:
    arr - 2-d array or list

    outputs:
    1-d list
    """
    return [item for sublist in arr for item in sublist]
#------------------------------------------------------------------------------------------------------------------------
labels_dict = dict({'bridge': 17, 'container': 15, 'asphalt': 6, 'heap of sand': 7, 'cars': 13, 'pipes': 10,
 'rubble': 11, 'reinforcement': 8, 'water': 4, 'concrete': 1, 'trees': 14, 'bike lane': 12,
 'background': 0, 'wooden boards': 9, 'foundations': 2, 'grass': 5, 'heavy earthy equipment': 16,
 'concrete rings': 3})
#------------------------------------------------------------------------------------------------------------------------
def convert_images_v2(final_images, original_images, class_name,output_root):
    """
    convert_images_v2 takes the images and converts them into labeled pairs and saves them in a directory

    arguments:
    final_images - list of images (2-d arrays of pixels) that have been segmented
    original_images - the original images that were pushed through the segmentation
    class_name - the class that is being replicated
    output_root - the string of the address of the directory where the images need to go

    output:
    the list of string addresses pointing to the shape files (labels)
    """
    shape_links = []
    if len(final_images) != len(original_images):
        print("Image Arrays are not the same length")
        return None
    if os.path.exists(output_root+"original_images/"):
        existing = os.listdir(output_root+"original_images/")
        numbers = []
        for item in existing:
            temp = item.split("/")
            if str(temp[len(temp)-1].split("_")[0]) == class_name:
                numbers.append(int(temp[len(temp)-1].split("_")[1][:-4]))
        placeholder = max(numbers)
        print("There are "+str(placeholder)+" images in this class - adding onto the class")
    else:
        os.makedirs(output_root+"original_images/")
        placeholder = 0
    if not os.path.exists(output_root+"shapes/"):
        os.makedirs(output_root+"shapes/")
    for im_ind in xrange(len(final_images)):
        original_image =cv2.imread(original_images[im_ind])
        final_label = final_images[im_ind]
        image_path = "original_images/" + class_name + "_" + str(im_ind+placeholder) + ".png"
        shape_path = "shapes/" + class_name + "_" + str(im_ind+placeholder) + ".png" 
        shape_links.append(output_root+shape_path)
        cv2.imwrite(output_root+shape_path,final_label)
        cv2.imwrite(output_root+image_path,original_image)
        if (im_ind % 10 == 0):
            print("Converted and Saved "+str(im_ind)+" images")
    print("Images are saved in "+output_root)
    return shape_links