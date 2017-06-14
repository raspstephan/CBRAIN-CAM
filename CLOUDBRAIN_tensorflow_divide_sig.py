# -*- coding: utf-8 -*-
"""
Created on Mon May 22, 2017

@author: gentine
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf
import netCDF4 as nc 
from netCDF4 import Dataset
import argparse
import sys
import time 
import datetime

from folderDefs import *

print("Starting")
slim = tf.contrib.slim

# Parameters
fraction_data   = 1./100. # fraction of data used for the training
training_epochs = 10 
learning_rate   = 1e-3
batch_size      = 4096
display_step    = 256
Ntimedata       = 1. # number of times through the same dataset

# Network Parameters
n_hidden_1      = 5 # 1st layer number of features
n_hidden_2      = 5 # 2nd layer number of features

LogDir  = LogDirMain + '/layer1_' + str(n_hidden_1)
if n_hidden_2>0:
   LogDir = LogDir +  '_layer2_' + str(n_hidden_2)
LogDir   = LogDir +  '_learn_rate_'+ str(learning_rate)
LogDir   = LogDir +  '_sig'
filename = LogDir + filename

print('LogDir is ' + LogDir)
counter = 0
# number of loops through entire dataset - randomly sampled
Nloop = int(float(Ntimedata)/fraction_data)                 
print('Nloop=',Nloop)

# need to retrieve mean and standard deviation of the full dataset first
print("Reading Netcdf for Normalization")
fh        = Dataset(nc_norm_file, mode='r')
mean_in   = fh.variables['mean'][:]
std_in    = fh.variables['std'][:]
fh.close()
print("End Reading Netcdf for Normalization")
                  
# NOW ACTUAL TENSORFLOW                  

n_input = mean_in.shape[0]
fh = Dataset(nc_file, mode='r')
OUTPUT   = fh.variables['SPDQ'][:]
n_outputs=OUTPUT.shape[0]
print(n_outputs)
del OUTPUT


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.sigmoid):
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights)
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases)
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.summary.histogram('pre_activations', preactivate)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations

#with tf.name_scope('dropout'):
#    keep_prob = tf.placeholder(tf.float32)
#    tf.summary.scalar('dropout_keep_probability', keep_prob)
#    dropped = tf.nn.dropout(hidden1, keep_prob)

# starting neural network:
# tf Graph input 
x = tf.placeholder(tf.float32, [None, n_input], name='x-input')
y = tf.placeholder(tf.float32, [None, n_outputs], name='y-output')

hidden1 = nn_layer(x, n_input, n_hidden_1, 'layer1')
if n_hidden_2>0:
    hidden2 = nn_layer(hidden1, n_hidden_1, n_hidden_2, 'layer2')
    pred = nn_layer(hidden2, n_hidden_2, n_outputs, 'layerout', act=tf.identity)    
else:
    pred = nn_layer(hidden1, n_hidden_1, n_outputs, 'layerout', act=tf.identity)

# Add ops to save and restore all the variables.
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(pred - y))
tf.summary.scalar('loss', loss)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('train'):
    optimizer  = tf.train.AdamOptimizer(learning_rate).minimize(loss)
saver = tf.train.Saver()
tf.contrib.layers.summarize_tensor(pred)

print("Running tensorflow")
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
   merged       = tf.summary.merge_all()
   train_writer = tf.summary.FileWriter(LogDir + '/train', sess.graph)
   test_writer  = tf.summary.FileWriter(LogDir + '/test')
   model        = tf.global_variables_initializer()
   
   sess.run(model)

   # Define loss and optimizer
   with tf.device('/gpu:0'):
      for index in range(0,Nloop):
         print("Loop number ", index)
         #fields = {'PS','QAP','TAP','OMEGA','SHFLX','LHFLX'};
         #input_names    = {'PS','QAP','TAP','OMEGA','SHFLX','LHFLX'};
         #target_names   = {'SPDQ'};%,'CLOUD','SODT','SPDQ','SPDQC','SPDQI','SPMC','SPMCDN','SPMCUDN','SPMCUP','SPMCUUP'
    
         print("Reading Netcdf")
         # read netcdf file
         fh = Dataset(nc_file, mode='r')
         PS       = fh.variables['PS'][:]
         N        = PS.shape[0]
         Ndata    = np.int_(fraction_data*N)
         batchlarge = np.int_(N*np.random.rand(Ndata))
         PS       = PS[batchlarge]
         QAP      = fh.variables['QAP'][:]
         QAP      = QAP[:,batchlarge]
         TAP      = fh.variables['TAP'][:]
         TAP      = TAP[:,batchlarge]
         OMEGA    = fh.variables['OMEGA'][:]
         OMEGA    = OMEGA[:,batchlarge]
         SHFLX    = fh.variables['SHFLX'][:]
         SHFLX    = SHFLX[batchlarge]
         LHFLX    = fh.variables['LHFLX'][:]
         LHFLX    = LHFLX[batchlarge]
         y_data   = fh.variables['SPDQ'][:]
         y_data   = y_data[:,batchlarge]
         fh.close()
         print("End Reading Netcdf")
         print(PS[None,:].shape)
         print(QAP.shape)

         inX = np.append(PS[None,:], QAP, axis=0)
#         del QAP
#         del PS
         inX = np.append(inX, TAP, axis=0)
#         del TAP
         inX = np.append(inX, OMEGA, axis=0)
#         del OMEGA
         inX = np.append(inX, SHFLX[None,:], axis=0)
#         del SHFLX
         inX = np.append(inX, LHFLX[None,:], axis=0)
         inX = np.transpose(inX)
        
         inX       = (inX - mean_in)/std_in
         y_data      = np.transpose(y_data)
     
         # Launch the graph
         summary_writer = tf.summary.FileWriter(LogDir, sess.graph)
         total_batch = int(Ndata/batch_size)
         print("total batch size", total_batch)
               
         # https://www.tensorflow.org/programmers_guide/threading_and_queues
         # Loop over all small batches
         for epoch in range(total_batch):
            batch = np.int_(Ndata*np.random.rand(batch_size))
            batch_x = inX[batch,:]      	# input sample
            batch_y = y_data[batch,:]  		# output sample
            # Run optimization op (backprop) and cost op (to get loss value)
            _, summary, l = sess.run([optimizer, merged, loss], feed_dict={x: batch_x, y: batch_y})
	    # Record a summary
            counter = counter + 1
            summary_writer.add_summary(summary, counter)
            # Display logs per epoch step
            if epoch % display_step == 0:
               	  print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(l))
                  # Save the variables to disk.
               	  run_metadata = tf.RunMetadata()
                  train_writer.add_run_metadata(run_metadata, 'step%03d' % (counter))
                  save_path = saver.save(sess, filename)
                  print("Model saved in file: %s" % save_path)
   print("Large Batch Optimization Finished!")
print("Final Optimization Finished!")
        
