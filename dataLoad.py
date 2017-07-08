import numpy as np
import shutil, time, math, itertools, os
import h5py
import netCDF4 as nc 
from netCDF4 import Dataset
from tqdm import tqdm
import tensorflow as tf
import threading
import random
from colorama import Fore, Back, Style
import sys
from folderDefs import *

class DataLoader:
    def __init__(self, folderPath, config):
        self.config = config
        self.filePath = folderPath+self.config.dataset
        self.batchSize = config.batch_size
        self.varname  = config.varname
        self.nSampleFetching = 1024
        self.reload()

    def reload(self, finishedEpoch = 0):
        # need to retrieve mean and standard deviation of the full dataset first
        print("Reading Netcdf for Normalization")
        fh = Dataset(nc_norm_file, mode='r')
        self.mean_in   = fh.variables['mean'][:]
        self.std_in    = fh.variables['std'][:]
        fh.close()
        print("End Reading Netcdf for Normalization")
        try:
            self.f.close()
        except:
            pass
        print("opening "+self.filePath)
        self.f = Dataset(nc_file, mode='r')
        print("batchSize = ", self.batchSize)

        self.Nsamples = self.f.variables['PS'][:].shape[0]
        print('Nsamples', self.Nsamples)
        self.n_input = self.mean_in.shape[0]
        self.n_output = self.f.variables[self.varname][:].shape[0]

        self.NumBatch = self.Nsamples // self.config.batch_size
        self.NumBatchTrain = int(self.Nsamples * self.config.frac_train) // self.batchSize
        self.indexValidation = self.NumBatchTrain * self.batchSize
        self.NumBatchValid = (self.Nsamples * (1.0 - self.config.frac_train)) // self.config.batch_size
        print('NumBatch', self.NumBatch)
        print('NumBatchTrain', self.NumBatchTrain)
        print('indexValidation', self.indexValidation)
        print('NumBatchValid', self.NumBatchValid)

        self.samplesTrain = range(0, self.indexValidation, self.nSampleFetching)
        self.randSamplesTrain = list(self.samplesTrain)
        random.shuffle(self.randSamplesTrain)
        self.samplesValid = range(self.indexValidation, self.Nsamples, self.nSampleFetching)
        self.randSamplesValid = list(self.samplesValid)
        random.shuffle(self.randSamplesValid)
        self.numFetchesTrain = len(self.randSamplesTrain)
        self.numFetchesValid = len(self.randSamplesValid)
        print('randSamplesTrain', self.randSamplesTrain[:16], self.numFetchesTrain)
        print('randSamplesValid', self.randSamplesValid[:16], self.numFetchesValid)
        self.posTrain = 0
        self.posValid = 0

        print('n_input', self.n_input)
        print('n_output', self.n_output)
        self.Xshape = [self.n_input]
        self.Yshape = [self.n_output]

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.f.close()
        except:
            pass

    def accessData(self, s, l):
#        print("Reading Netcdf")
        fh = self.f
        PS       = fh.variables['PS'][s:s+l]
        QAP      = fh.variables['QAP'][:,s:s+l]
        TAP      = fh.variables['TAP'][:,s:s+l]
        OMEGA    = fh.variables['OMEGA'][:,s:s+l]
        SHFLX    = fh.variables['SHFLX'][s:s+l]
        LHFLX    = fh.variables['LHFLX'][s:s+l]
        y_data   = fh.variables[self.varname][:,s:s+l]
#        print("End Reading Netcdf")

#        print('PS.shape', PS.shape)
#        print('QAP.shape', QAP.shape)
#        print('TAP.shape', TAP.shape)
#        print('OMEGA.shape', OMEGA.shape)
#        print('SHFLX.shape', SHFLX.shape)
#        print('LHFLX.shape', LHFLX.shape)
#        print('y_data.shape', y_data.shape)

        inX = np.append(PS[None,:], QAP, axis=0)
        #del QAP
        #del PS
        inX = np.append(inX, TAP, axis=0)
        #del TAP
        inX = np.append(inX, OMEGA, axis=0)
        #del OMEGA
        inX = np.append(inX, SHFLX[None,:], axis=0)
        #del SHFLX
        inX = np.append(inX, LHFLX[None,:], axis=0)
        inX = np.transpose(inX)

        inX    = (inX - self.mean_in) / self.std_in
        y_data = np.transpose(y_data)

        return inX, y_data

    def sampleTrain(self):
        s = self.randSamplesTrain[self.posTrain]
        #print(self.posTrain, s)
        self.posTrain = (self.posTrain+1) % self.numFetchesTrain
        x,y = self.accessData(s, self.nSampleFetching)
        return x,y

    def sampleValid(self):
        s = self.randSamplesValid[self.posValid]
        self.posValid = (self.posValid+1) % self.numFetchesValid
        x,y = self.accessData(s, self.nSampleFetching)
        return x,y

    def data_iterator(self):
        """ A simple data iterator """
        while True:
            sampX, sampY = self.sampleTrain() if self.config.is_train else self.sampleValid()
            yield sampX, sampY

    def prepareQueue(self):
        with tf.name_scope('prepareQueue'):
            self.dataX = tf.placeholder(dtype=tf.float32, shape=[None]+self.Xshape)
            self.dataY = tf.placeholder(dtype=tf.float32, shape=[None]+self.Yshape)

            self.capacityTrain = max(self.nSampleFetching * 32, self.batchSize * 8)
            self.queue = tf.RandomShuffleQueue(shapes=[self.Xshape, self.Yshape],
                                               dtypes=[tf.float32, tf.float32],
                                               capacity=self.capacityTrain,
                                               min_after_dequeue=self.capacityTrain // 2
                                               )
            self.enqueue_op = self.queue.enqueue_many([self.dataX, self.dataY])
            self.size_op = self.queue.size()

    def get_inputs(self):
        with tf.name_scope('dequeue'):
            train0Valid1 = tf.placeholder_with_default(1, [], name='train0Valid1')
            b_X, b_Y = self.queue.dequeue_many(self.batchSize)
            print("b_X",b_X.get_shape(), "b_Y",b_Y.get_shape())
            return b_X, b_Y

    def thread_main(self, sess):
        for dtX, dtY in self.data_iterator():
            sess.run(self.enqueue_op, feed_dict={self.dataX:dtX, self.dataY:dtY})

    def start_threads(self, sess, n_threads=1):
        """ Start background threads to feed queue """
        threads = []
        print("starting %d data threads for training" % n_threads)
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        # Make sure the queueu is filled with some examples (n = 500)
        num_samples_in_queue = 0
        while num_samples_in_queue < self.capacityTrain:
            num_samples_in_queue = sess.run(self.size_op)
            print("Initializing queue, current size = %i/%i" % (num_samples_in_queue, self.capacityTrain))
            time.sleep(2)
        return threads

