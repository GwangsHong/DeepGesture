#from __future__ import print_function
#import matplotlib.pyplot as plt
#import numpy as np
#import time
#import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation #, Dropout
from keras.layers.recurrent import LSTM #, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution2D,Dropout,Flatten ,Reshape, Bidirectional, Lambda

import keras.backend as K
import argparse
import os
from data import *
import datetime
from models import *
import random
import signal_processing as sp
import csv
parser = argparse.ArgumentParser(description='Tensorflow  backend Gesture Classification Training on several datasets')
parser.add_argument('--data', default = "./train", type= str,
                    help='path to dataset')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run (default: 90')
parser.add_argument('--lr', default=0.00001, type=float,
                    help='initial learning rate')
parser.add_argument('--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--n_input', default=6, type=int,
                   help='number of input (default: 6)')
parser.add_argument('--n_hidden', default=256, type=int,
                    help='number of hidden (default: 256)')
parser.add_argument('--n_layers', default=4, type=int,
                    help='number of layers (default: 4)')
parser.add_argument('--n_steps', type=int, default=128)
parser.add_argument('-n', '--n_classes', default=5, type=int,
                    help='number of classes (default: 2)')
parser.add_argument('--mode', default='train')
parser.add_argument('--pretrained', default = "", type= str,
                    help='path to pre-trained model')
parser.add_argument('--input_type', default = "raw", type= str,
                    help='select input type')
parser.add_argument('--m', default='ox' ,type = str,help='model')
args = parser.parse_args()


# Creating training data

x_train_signals_paths = [
        os.path.join(args.data, signal + ".csv") for signal in INPUT_SIGNAL_TYPES
    ]
y_path = os.path.join(args.data, "y.csv")
x_train, x_test, y_train, y_test = load_dataset(x_train_signals_paths, y_path)

x_train = x_train.reshape(-1, args.n_input, args.n_steps, 1)
x_test = x_test.reshape(-1, args.n_input, args.n_steps, 1)

print("input train shape:{}".format(x_train.shape))
print("input test shape:{}".format(x_test.shape))
model, checkpoint ,save_path= create_model(args)

with open(os.path.join(save_path,"log.csv"),'w') as save_log_file:
    csvwiter = csv.writer(save_log_file)
    csvwiter.writerow(['epoch','loss','acc','val_loss','val_acc' ,'test_loss', 'test_acc'])

for i in range(args.epochs):
    hist = model.fit(x_train, y_train, batch_size=args.batch_size, verbose=1 ,callbacks=[checkpoint], epochs=1, validation_split=0.2)

    score = model.evaluate(x_test,y_test,batch_size=args.batch_size,verbose=1)
    print('[{}]Test loss :{} Test acc :{}'.format(i,score[0],score[1]))
    with open(os.path.join(save_path, "log.csv"), 'a') as save_log_file:
        csvwiter = csv.writer(save_log_file)
        csvwiter.writerow([i, hist.history['loss'][0], hist.history['acc'][0], hist.history['val_loss'][0],
                      hist.history['val_acc'][0],score[0],score[1]])
