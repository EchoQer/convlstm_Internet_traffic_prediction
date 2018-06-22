#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 13:14:32 2018

@author: echo
"""

import os
import numpy
import math
import numpy
import json



def read_file(path):
    '''
    Read the entire dataset text from the given path.
    
    Args:
        path:the path of dataset 
        
    Returns:
        data_set:the list of data set after preprocessing
        
    '''
    data_set = []
    
    # check weather the input String is a directory or file
    if not os.path.isdir(path) and not os.path.isfile(path):
        return False
    if os.path.isfile(path):
        
             # open the file with the mode'r'
            fp = open(path)
            while 1 :
                next_line = fp.readline()              
                # check whether it is the end of the text
                if not next_line :
                        break       
                # split next_line with character '\t' and put the results into a list
                
                next_line = next_line[0:-1]
                next_line_list = next_line.split('\t')
                               
#==============================================================================
#                 square_id = next_line_list[0]
#                 time_interval = next_line_list[1]
#                 country_code = next_line_list[2]
#                 sms_in = next_line_list[3]
#                 sms_out = next_line_list[4]
#                 call_in = next_line_list[5]
#                 call_out = next_line_list[6]
#                 internet_traffic = next_line_list[7]                               
#==============================================================================
                # check whether there is internet traffic activity data
                if(next_line_list[7] != ''):
                    
                    square_id = int(next_line_list[0])
                    time_interval = int(next_line_list[1])
                    internet_traffic = float(next_line_list[7])
                    
                    temp_list = [square_id,time_interval,internet_traffic]
                    
                    if(len(data_set) != 0):
                        if(data_set[-1][0] == square_id and data_set[-1][1] == time_interval):                          
                            data_set[-1][2] = data_set[-1][2] + internet_traffic
                        else:
                            data_set.append(temp_list)
                    else:
                        data_set.append(temp_list)
            fp.close()
    return data_set


def data_preprocess(data_set):
    '''
    Preprocess the  dataset  from the given list.
    
    Args:
        data_set:the list of dataset 
        
    Returns:
        dict_data:the dictionary of dataset organised by time interval 
        
    '''
    # initialise the dictionary of dataset organised by time interval 
    data_array = list(numpy.array(data_set).T)
    time_interval_list = list(set(data_array[1])) 
    
    dict_data = {}
    for i in time_interval_list:
        dict_data[i] = []
    
    for i in data_set:      
        dict_data[i[1]].append([i[0],i[2]])

    for i in time_interval_list:
        
        # initialise grid traffic activity data
        grid_traffic = [([0] * 100) for i in range(100)]
        
        for j in dict_data[i]:
            a = (j[0]-1)//100
            b = j[0]%100
            grid_traffic[99-a][b-1] = j[1]

        dict_data[i] = grid_traffic
    return dict_data
    
def write_grid_internet_activity(dict_data,file_path):
    with open(file_path,"w") as f:
        json.dump(dict_data,f)
        print("Write file succeed..." + file_path)



month = [11,12] 
day = [x for x in range(1,32)]
index = 1
for i in month:
    for j in day:
        i = str(i) 
        j = str(j) 
        if len(j) == 1:
            j = '0' + j 
        read_path = 'original dataset/sms-call-internet-mi-2013-' + i + '-' + j + '.txt'
        write_path = 'dataset/' + str(index) +  '.json'
        data_set = read_file(read_path)
        
        if data_set != False:
            dict_data = data_preprocess(data_set)
            write_grid_internet_activity(dict_data,write_path)
            index = index + 1

read_path_1 = 'original dataset/sms-call-internet-mi-2014-01-01.txt'
write_path_1 = 'dataset/62.json'
data_set_1 = read_file(read_path_1)
if data_set_1 != False:
    dict_data_1 = data_preprocess(data_set_1)
    write_grid_internet_activity(dict_data_1,write_path_1)