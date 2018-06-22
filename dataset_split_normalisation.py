#dataset split and normalization

import numpy as np
import json
import math

TRAIN_PARTITION = 0.8
VAL_PARTITION = 0.1

data_set = {}
training_data = {}
validation_data = {}
test_data = {}

# =============================================================================

def read_file(path):
    
    with open(path,'r') as load_f:
        load_dict = json.load(load_f)
        print("Load file succeed..." + path)
    return load_dict

# =============================================================================
    
def write_grid_internet_activity(dict_data,file_path):
    with open(file_path,"w") as f:
        json.dump(dict_data,f)
        print("Write file succeed..." + file_path)

# =============================================================================
        
def deta_normalization(data_dic):
    
    temp = list(data_dic.values())
    print(np.array(temp).shape)
    mean = np.array(temp).mean()
    std = np.array(temp).std()
    print(mean)
    print(std)
    for value in data_dic.values():       
        value = [[(value[i][j] - mean)/std for j in range(len(value[i]))] for i in range(len(value))]
    return data_dic

# =============================================================================



for i in range(1,63):
    
    path = 'dataset/'  + str(i) + '.json'
    if i == 1 :
        dataset = read_file(path)
    else:
        dataset = dict(dataset, **read_file(path))

        
dataset = deta_normalization(dataset)
length = len(dataset)

train_size = math.floor( TRAIN_PARTITION * len(dataset))
val_size = math.floor( VAL_PARTITION * len(dataset))

index = 1
for key in dataset.keys():
    
    if index in range(1, train_size + 1):
        training_data[key] = dataset[key]
       
    else:
        if index in range(train_size + 1, train_size + val_size + 1):
            validation_data[key] = dataset[key]
            
        else:
            test_data[key] = dataset[key]
            
    index = index + 1           
  
print(len(training_data))    
print(len(validation_data))    
print(len(test_data))    
 
write_grid_internet_activity(training_data,'dataset/training/training.json')
write_grid_internet_activity(validation_data,'dataset/validation/validation.json')
write_grid_internet_activity(test_data,'dataset/test/test.json')
 
