import numpy as np
import shutil, time, math, itertools, os
import h5py
#import netCDF4 as nc 
#from netCDF4 import Dataset
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
        self.batchSize = config.batch_size
        self.nSampleFetching = 1024
        self.varname = config.dataset
        self.fileReader = []
        self.lock = threading.Lock()
        self.reload()

    def reload(self, finishedEpoch = 0):
        # need to retrieve mean and standard deviation of the full dataset first
        print("Reading Netcdf for Normalization")
        fh = h5py.File(nc_norm_file, mode='r')
        self.mean_in   = fh['mean'][:][None]   # (93, 1)
        self.std_in    = fh['std'][:][None]    # (93, 1)
        print('self.mean_in', self.mean_in.shape)
        print('self.std_in', self.std_in.shape)
        fh.close()
        print("End Reading Netcdf for Normalization")
        try:
            for i in range(len(self.fileReader)):
                self.fileReader[i].close()
        except:
            pass
        print("batchSize = ", self.batchSize)

        fh = h5py.File(nc_file, mode='r')
        self.Nsamples = fh['PS'][:].shape[0]
        print('Nsamples', self.Nsamples)
        self.n_input = self.mean_in.shape[1]
        self.n_output = fh[self.varname][:].shape[0]
        fh.close()

        self.NumBatch = self.Nsamples // self.config.batch_size
        self.NumBatchTrain = int(self.Nsamples * self.config.frac_train) // self.batchSize
        self.indexValidation = self.NumBatchTrain * self.batchSize
        self.NumBatchValid = int(self.Nsamples * (1.0 - self.config.frac_train)) // self.config.batch_size
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
            for i in range(len(self.fileReader)):
                self.fileReader[i].close()
        except:
            pass

    def accessData(self, s, l, ithFileReader):
        fh = self.fileReader[ithFileReader]

        QAP      = fh['QAP'][:,s:s+l]       # QAP    kg/kg   30   Specific humidity (after physics)
        TAP      = fh['TAP'][:,s:s+l]       # TAP    K       30   Temperature (after physics)
        OMEGA    = fh['OMEGA'][:,s:s+l]     # OMEGA  Pa/s    30   Vertical velocity (pressure)
        PS       = fh['PS'][s:s+l][None]    # PS     Pa      1    Surface pressure
        SHFLX    = fh['SHFLX'][s:s+l][None] # SHFLX  W/m2    1    Surface sensible heat flux
        LHFLX    = fh['LHFLX'][s:s+l][None] # LHFLX  W/m2    1    Surface latent heat flux

        y_data   = fh[self.varname][:,s:s+l]      # SPDT   K/s     30   dT/dt

#        print('PS.shape', PS.shape)
#        print('PS.shape[None,:]', PS.shape)
#        print('QAP.shape', QAP.shape)
#        print('TAP.shape', TAP.shape)
#        print('OMEGA.shape', OMEGA.shape)
#        print('SHFLX.shape', SHFLX.shape)
#        print('LHFLX.shape', LHFLX.shape)
#        print('y_data.shape', y_data.shape)

        inX = np.concatenate([PS, QAP, TAP, OMEGA, SHFLX, LHFLX], axis=0)
        inX = np.transpose(inX)
#        print('inX.shape', inX.shape)

#        inX    = (inX - self.mean_in) / self.std_in
        y_data = np.transpose(y_data)
        y_data *= 1e4

        return inX, y_data

    def sampleTrain(self, ithFileReader):
#        self.lock.acquire()
        s = self.randSamplesTrain[self.posTrain]
        #print(ithFileReader, self.posTrain, s)
        self.posTrain += 1
        self.posTrain %= self.numFetchesTrain
#        self.lock.release()
        x,y = self.accessData(s, self.nSampleFetching, ithFileReader)
        return x,y

    def sampleValid(self, ithFileReader):
        s = self.randSamplesValid[self.posValid]
        self.posValid += 1
        self.posValid %= self.numFetchesValid
        x,y = self.accessData(s, self.nSampleFetching, ithFileReader)
        return x,y

    def data_iterator(self, ithFileReader):
        """ A simple data iterator """
        print('data_iterator', ithFileReader, threading.current_thread())
        while True:
            sampX, sampY = self.sampleTrain(ithFileReader) if self.config.is_train else self.sampleValid(ithFileReader)
            yield sampX, sampY

    def prepareQueue(self):
        with tf.name_scope('prepareQueue'):
            self.dataX = tf.placeholder(dtype=tf.float32, shape=[None]+self.Xshape)
            self.dataY = tf.placeholder(dtype=tf.float32, shape=[None]+self.Yshape)

            self.capacityTrain = max(self.nSampleFetching * 32, self.batchSize * 8) if self.config.is_train else self.batchSize
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

    def thread_main(self, sess, ithFileReader):
        print('thread_main', ithFileReader, threading.current_thread())
        while len(self.fileReader) <= ithFileReader + 1:
            self.fileReader += [h5py.File(nc_file, mode='r')]
        for dtX, dtY in self.data_iterator(ithFileReader):
            sess.run(self.enqueue_op, feed_dict={self.dataX:dtX, self.dataY:dtY})

    def start_threads(self, sess, n_threads=4):
        """ Start background threads to feed queue """
        threads = []
        print("starting %d data threads for training" % n_threads)
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess,0,))
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

