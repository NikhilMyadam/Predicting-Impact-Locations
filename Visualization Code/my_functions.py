# -*- coding: utf-8 -*-
"""
Created on January 5th, 2022

@author: Nikhil Myadam
"""

# project_path --> Project Data path
# augmdata_path --> Augmented Data path
# currdata_path --> EPOT Data/Training Data path

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from random import randint

project_path = 'C:\CAME International Academy\Winter Semester 2021-22\Computational Intelligence in Engineering\Project B_G15\Project Data_Sciebo'
augmdata_path = os.path.join(project_path, 'Augmented_Data')
currdata_path = os.path.join(project_path, 'EPOT_Data')
valaugdata_path = os.path.join(project_path, 'Validation_augmented_data')

# reading all the files - EPOT Training files and the Augmented Data files
def read_all():
    signal_li, filelist = [], []
    data = {}

    _, _, pr_files = next(os.walk(currdata_path))
    _, _, aug_files = next(os.walk(augmdata_path))

    # reading all the files present in the Training Data path
    for _, value in enumerate(pr_files):
        fname = os.path.join(currdata_path, value)
        mat_data = loadmat(fname)
        temp_signal = mat_data.get('num_data')
        data[value] = temp_signal
        signal_li.append(temp_signal)
        filelist.append(value)
        
    # reading all the files present in the Augmented Data path
    for _, value in enumerate(aug_files):
        fname = os.path.join(augmdata_path, value)
        mat_data = loadmat(fname)
        temp_signal = mat_data.get('Data')
        data[value] = temp_signal
        signal_li.append(temp_signal)
        filelist.append(value)

    # to convert the list i.e. 'signal_li' into a numpy array
    # to concatenate all the files/signals along the 3D axis i.e. depth-wise 
    signal_arr = np.dstack(signal_li)
    signal_arr = np.rollaxis(signal_arr, -1)

    return signal_arr, filelist, data

# reading all the Validation augmented data files
def read_valaugfiles():
    signal_valaug, valaug_filelist = [], []
    data_valaug = {}
    
    _, _, valaug_files = next(os.walk(valaugdata_path))
    
    for _, value in enumerate(valaug_files):
        fname = os.path.join(valaugdata_path, value)
        mat_data = loadmat(fname)
        temp_signal = mat_data.get('num_data')
        data_valaug[value] = temp_signal
        signal_valaug.append(temp_signal)
        valaug_filelist.append(value)
        
    valaugsignal_arr = np.dstack(signal_valaug)
    valaugsignal_arr = np.rollaxis(valaugsignal_arr, -1)
    
    return valaugsignal_arr, valaug_filelist, data_valaug

# reading a randomn file from the filelist corresponding to the randomn index generated
# def read_randomn_file(filelist):
#     idx = randint(0, len(filelist))
#     print('Index :', idx)
#     filename = filelist[idx]
#     print('Filename :', filename)
#     return idx, filename

# better to run the above block of code on the main script/code

# to find out the impact co-ordinates corresponding to the randomn file read
def find_impact_coordinates(filename):
    temp = filename.split("_", 3)
    x_coordinate = int(temp[1])
    y_coordinate = int(temp[2][:-4])
    
    return x_coordinate, y_coordinate

# to provide a label to the file/signal read - i.e. position of the impact location
def find_impact_loc_label(filename):

    x_coordinate, y_coordinate = find_impact_coordinates(filename)
    center_point = [250, 250]

    # axis lines coordinates
    x_axis = int(center_point[0])
    y_axis = int(center_point[1])

    # center point
    if (x_coordinate == x_axis) & (y_coordinate == y_axis):
        return ('Center')

    # horizontal axis
    elif (x_coordinate != x_axis) & (y_coordinate == y_axis):
        return ('X_Axis')

    # vertical axis
    elif (x_coordinate == x_axis) & (y_coordinate != y_axis):
        return ('Y_Axis')

    # first quadrant
    elif (x_coordinate > x_axis) & (y_coordinate > y_axis):
        return ('Q1')

    # second quadrant
    elif (x_coordinate < x_axis) & (y_coordinate > y_axis):
        return ('Q2')

    # third quadrant
    elif (x_coordinate < x_axis) & (y_coordinate < y_axis):
        return ('Q3')

    # fourth quadrant
    elif (x_coordinate > x_axis) & (y_coordinate < y_axis):
        return ('Q4')

# to find the parent/epot training data file coordinates corresponding to the randomn file read
def find_parent_coordinates(filename):

    x_coordinate, y_coordinate = find_impact_coordinates(filename)
    center_point = [250, 250]

    # axis lines coordinates
    x_axis = int(center_point[0])
    y_axis = int(center_point[1])

    # finding the parent file if the label of 'filename' is Q1
    if (x_coordinate > x_axis) & (y_coordinate > y_axis):
        x_parent_coordinate = x_coordinate
        y_parent_coordinate = y_coordinate
        return x_parent_coordinate, y_parent_coordinate  

    # finding the parent file if the label of 'filename' is Q2
    if (x_coordinate < x_axis) & (y_coordinate > y_axis):
        y_parent_coordinate = y_coordinate
        x_parent_coordinate = x_axis - (x_coordinate - x_axis)
        return x_parent_coordinate, y_parent_coordinate

    # finding the parent file if the label of 'filename' is Q3
    if (x_coordinate < x_axis) & (y_coordinate < y_axis):
        x_temp = x_coordinate
        y_temp = y_axis - (y_coordinate - y_axis)
        y_parent_coordinate = y_temp
        x_parent_coordinate = x_axis - (x_temp - x_axis)
        return x_parent_coordinate, y_parent_coordinate

    # finding the parent file if the label of 'filename' is Q4
    if (x_coordinate > x_axis) & (y_coordinate < y_axis):
        x_parent_coordinate = x_coordinate
        y_parent_coordinate = y_axis - (y_coordinate - y_axis)
        return x_parent_coordinate, y_parent_coordinate

# to find the parent/epot training data file from its respective coordinates
def find_parent_filename(filelist, filename):

    x_parent, y_parent = find_parent_coordinates(filename)
    parent_file = []

    for i in range(len(filelist)):
        temp = filelist[i].split('_', 3)
        x_coordinate = int(temp[1])
        y_coordinate = int(temp[2][:-4])
        if (x_parent == x_coordinate) & (y_parent == y_coordinate):
            parent_file = filelist[i]

    return parent_file

# to get the validation augmented data files
def get_valaug_signal(label, data_valaug):

    # first quadrant
    if (label == 'Q1'):
        return data_valaug['EPOT_275_265.mat']

    # second quadrant
    elif (label == 'Q2'):
        return data_valaug['EPOT_225_265.mat']

    # third quadrant
    elif (label == 'Q3'):
        return data_valaug['EPOT_225_235.mat']
    
    # fourth quadrant
    elif (label == 'Q4'):
        return data_valaug['EPOT_275_235.mat']

# to generate subplots
def create_subplots_channelwise(signal1, signal2, signal3, signal4, channel):

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize = (15,8))
    fig.suptitle('Signal Subplots')
    ax1.plot(signal1[:,channel], 'tab:blue')
    ax1.set_title('Validation Augmented Parent Signal')
    ax2.plot(signal2[:,channel], 'tab:blue')
    ax2.set_title('Validation Augmented Randomn Signal')
    ax3.plot(signal3[:,channel], 'tab:blue')
    ax3.set_title('EPOT Data Parent Signal')
    ax4.plot(signal4[:,channel], 'tab:blue')
    ax4.set_title('Augmented Data Randomn Signal')
    plt.show()



