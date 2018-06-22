#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 23:04:53 2018

@author: echo
"""


import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import json







LEARNING_RATE = 0.001
EPOCH = 50
INPUT_SIZE = 15
CHANNEL = 1
TIME_STEPS = 5
BATCH_SIZE = 20
#------------------------------------Generate Data-----------------------------------------------#
train_set = []
test_set = []

#generate data
def generate(seq):
    X=[]
    y=[]
    for i in range(len(seq) - 2 * TIME_STEPS + 1):
        X.append( seq[i : i + TIME_STEPS] )
        y.append( seq[i + TIME_STEPS : i + 2 * TIME_STEPS] )
    return np.array(X, dtype = np.float32), np.array(y, dtype = np.float32)

def load_data(path):
    
    seq = []
    with open(path,'r') as load_f:
        load_dict = json.load(load_f)
    
    for value in load_dict.values():
        value =np.array(value)[-1][0:INPUT_SIZE]
        value = np.array(value).reshape((INPUT_SIZE,1))
        seq.append(value)
    return seq
# =============================================================================


day = [x for x in range(1,31)]

for i in day:
    
    i = str(i) 
    if len(i) == 1:
        i = '0' + i 
    path = 'preprocessed dataset/grid-internet-mi-2013-11-' + i + '.json'
#    print(i)
    if i == '01':
        train_set = np.array(load_data(path))
#        print(train_set.shape)
    else:
        train_set  = np.concatenate((train_set,load_data(path)),axis=0)
#        print(train_set.shape)
        
X_train,y_train = generate(train_set)
TRAIN_EXAMPLES =  X_train.shape[0]
day = [x for x in range(1,6)]
for i in day:
    
    i = str(i) 
    if len(i) == 1:
        i = '0' + i 
    path = 'preprocessed dataset/grid-internet-mi-2013-12-' + i + '.json'
#    print(i)
    if i == '01':
        test_set = np.array(load_data(path))
#        print(train_set.shape)
    else:
        test_set  = np.concatenate((test_set,load_data(path)),axis=0)
#        print(train_set.shape)
X_test,y_test=generate(test_set)    
TEST_EXAMPLES =  X_test.shape[0]



#--------------------------------------Define Graph---------------------------------------------------#
graph = tf.Graph()
with graph.as_default():

    # placehoder
    encoder_inputs = tf.placeholder(tf.float32, [TIME_STEPS, None, INPUT_SIZE, CHANNEL])
    decoder_target_outputs = tf.placeholder(tf.float32, [TIME_STEPS, None, INPUT_SIZE, CHANNEL])
    _, batch_size, _ , _= tf.unstack(tf.shape(encoder_inputs))

    #------------------------------------Encoder------------------------------------------#
    #Convlstm instance
    encoder_cell_1 = rnn.ConvLSTMCell(
            conv_ndims = 1,
            input_shape = [INPUT_SIZE, CHANNEL],
            output_channels = 16,
            kernel_shape = [6,],
            use_bias = True,
            skip_connection = False,
            forget_bias = 1.0,
            initializers = None,
            name = 'encode')

    encoder_cell_2 = rnn.ConvLSTMCell(
            conv_ndims = 1,
            input_shape = [INPUT_SIZE, 16],
            output_channels = 16,
            kernel_shape = [6,],
            use_bias = True,
            skip_connection = False,
            forget_bias = 1.0,
            initializers = None,
            name = 'encoder')
    
    encoder_cells = rnn.MultiRNNCell(cells = [encoder_cell_1, encoder_cell_2])
     
    #initialize to zero
    init_state = encoder_cells.zero_state(batch_size, dtype=tf.float32)

    
    
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
            cell = encoder_cells,
            inputs = encoder_inputs,
            initial_state = init_state,
            time_major = True,
            scope = "encoder")
    print('encoder_final_state')
    print(encoder_final_state)

    #---------------------------------------Decoder-----------------------------------------------#
    
    decoder_cell_1 = rnn.ConvLSTMCell(
            conv_ndims = 1,
            input_shape = [INPUT_SIZE, 16],
            output_channels = 16,
            kernel_shape = [6,],
            use_bias = True,
            skip_connection = False,
            forget_bias = 1.0,
            initializers = None,
            name = "decoder")
    decoder_cell_2 = rnn.ConvLSTMCell(
            conv_ndims = 1,
            input_shape = [INPUT_SIZE, 16],
            output_channels = 16,
            kernel_shape = [6,],
            use_bias = True,
            skip_connection = False,
            forget_bias = 1.0,
            initializers = None,
            name = "decoder")
    
    decoder_cells = rnn.MultiRNNCell(cells = [decoder_cell_1, decoder_cell_2])
    
    
    # decoder_init_input
    decoder_init_input = tf.zeros([batch_size, INPUT_SIZE, 16], dtype=tf.float32, name='decoder_init_input')
    
   
    # Initial call at time=0 to provide initial cell_state and input to convlstm.
    def loop_fn_initial():
        
        initial_elements_finished = (0 >= TIME_STEPS)  # all False at the initial step
        initial_input = decoder_init_input

        initial_cell_state = encoder_final_state
        initial_cell_output = None
        initial_loop_state = None  
        return (initial_elements_finished,
                initial_input,
                initial_cell_state,
                initial_cell_output,
                initial_loop_state)
    # transition call for all following timesteps 
    def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

        def get_next_input():

            return previous_output
        # this operation produces boolean tensor of [batch_size], defining if corresponding sequence has ended
        elements_finished = (time >= TIME_STEPS) 
                                                 
        # -> boolean scalar
        finished = tf.reduce_all(elements_finished) 
        next_input = tf.cond(finished, lambda: decoder_init_input, get_next_input)
        state = previous_state
        output = previous_output
        loop_state = None

        return (elements_finished, 
                next_input,
                state,
                output,
                loop_state)
        
    def loop_fn(time, previous_output, previous_state, previous_loop_state):
        if previous_state is None:    # time == 0
            assert previous_output is None and previous_state is None
            return loop_fn_initial()
        else:
            return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

    decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cells, loop_fn)

    decoder_outputs = decoder_outputs_ta.stack()#time_step,batch_size
    print('decoder_outputs')
    print(decoder_outputs)
    decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, INPUT_SIZE,16))
 
    decoder_predictions = tf.layers.conv1d(inputs = decoder_outputs_flat,
                                          filters=1,
                                          kernel_size=8,
                                          strides=(1,),
                                          padding = 'same')
    decoder_predictions =  tf.reshape(decoder_predictions, (TIME_STEPS, batch_size, INPUT_SIZE, 1))
    
   
    #---------------------------------define loss and optimizer----------------------------------#

    losses = tf.losses.mean_squared_error(labels = decoder_target_outputs, predictions = decoder_predictions)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=losses)
    init = tf.global_variables_initializer()

    

    
    
with tf.Session(graph=graph) as sess:   
    
    draw_1 = []
    draw_2 = []
    average_test_loss = []
    
    sess.run(init)
    for epoch in range(1,EPOCH+1):
        target_output = []
        results = []
        target_outputs = []
        train_losses=[]
        test_losses=[]
        print("epoch:",epoch)

        for j in range(TRAIN_EXAMPLES//BATCH_SIZE):
            _,train_loss = sess.run(fetches=(optimizer,losses),
                                    feed_dict = {
                                            encoder_inputs : X_train[j * BATCH_SIZE :(j + 1) * BATCH_SIZE].transpose((1,0,2,3)),
                                            decoder_target_outputs : 
                                                np.array(y_train[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]).transpose((1,0,2,3))
                                                }
                                        )
            train_losses.append(train_loss)
        print("average training loss:", sum(train_losses) / len(train_losses))

        for j in range(TEST_EXAMPLES//BATCH_SIZE):
            result, test_loss = sess.run(fetches = (decoder_predictions,losses),
                                          feed_dict = {
                                                  encoder_inputs : X_test[j * BATCH_SIZE : (j + 1) * BATCH_SIZE].transpose((1,0,2,3)),
                                                  decoder_target_outputs : 
                                                      np.array(y_test[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]).transpose((1,0,2,3))
                                                      }
                                              )
            test_losses.append(test_loss)
# =============================================================================
#             
#             result = result.reshape(-1)
#             target_output = np.array(y_test[j * BATCH_SIZE : (j + 1) * BATCH_SIZE]).transpose((1,0,2,3))
# 
#             target_output = target_output.reshape(-1)
#              
#             if j == 0:
#                 results = result
#                 target_outputs = target_output
#             else:
#                 results = np.concatenate((results,result),axis=0)
#                 target_outputs = np.concatenate((target_outputs,target_output),axis=0)
#              
# =============================================================================
        print("average test loss:", sum(test_losses) / len(test_losses))
        average_test_loss.append(sum(test_losses) / len(test_losses))
     
            
    plt.plot(range(1,EPOCH+1), average_test_loss, label="average test loss")
    plt.xlabel('Epoch Number')
    plt.ylabel('Test loss')
    plt.title('1D convolution on temporal data')
    plt.legend()
    plt.show()
 

# #  =============================================================================
# # =============================================================================
# 
# =============================================================================
