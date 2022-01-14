# -*- coding: utf-8 -*-
"""
Created on January 9th 2022

@author: Nikhil Myadam
"""

import os
import my_functions as mf
from random import randint


if __name__=='__main__':

	# Generating the Index and corresponding randomn file name
	signal_array, file_list, data = mf.read_all()
	index = randint(0, len(file_list))
	file_name = file_list[index]
	print('Index: ', index)
	print('Randomn File: ', file_name)

	# Generating the Parent file name corresponding to the randomn file name
	# x_parent, y_parent = mf.find_parent_coordinates(file_name)
	parent_file_name = mf.find_parent_filename(file_list, file_name)

	# Generating the Validation augmented signal for comparision
	label = mf.find_impact_loc_label(file_name)
	valaugsignal_arr, valaug_filelist, data_valaug = mf.read_valaugfiles()
	Valaug_signal = mf.get_valaug_signal(label, data_valaug)

	# Generating all the required signals
	rlabel_valaugfile = mf.find_impact_loc_label(file_name)
	plabel_valaugfile = mf.find_impact_loc_label(parent_file_name)

	signal1 = mf.get_valaug_signal(plabel_valaugfile, data_valaug)
	signal2 = mf.get_valaug_signal(rlabel_valaugfile, data_valaug)
	signal3 = data[parent_file_name]
	signal4 = data[file_name]

	# Generating the subplots using all four signals
	channel = int(input(' Input channel -->  ')) 
	mf.create_subplots_channelwise(signal1, signal2, signal3, signal4, channel)
	print('okay bye')