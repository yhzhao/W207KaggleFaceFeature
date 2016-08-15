import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

import lasagne

FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'


def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


X, y = load()
print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))

import theano
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

from scipy import fftpack
import matplotlib.pyplot as pyplot

def dct2(xIn, xDim = 96, yDim = 96):
    #dctback = fftpack.idct(dct)

    return fftpack.dct(fftpack.dct(xIn.reshape(xDim,yDim), norm = 'ortho').T, norm = 'ortho')

def idct2(xIn2Dim):
    return fftpack.idct(fftpack.idct(xIn2Dim, norm = 'ortho').T, norm = 'ortho')

def primHPF(xIn2Dim, level = 5):
    shape0 = xIn2Dim.shape[0]
    shape1 = xIn2Dim.shape[1]
    
    for i in range(shape0 - level, shape0):
        for j in range(shape1 - level, shape1):
            xIn2Dim[i,j] = 0
    return xIn2Dim

def primLPF(xIn2Dim, level = 5):
    for i in range(level):
        for j in range(level):
            xIn2Dim[i,j] = 0
    return xIn2Dim

def load2dHPFFilter(l = 5):
    X, y = load()
    newX = np.empty_like(X)
    for i in range(X.shape[0]):
        newX[i] = idct2(primHPF(dct2(X[i]), level=l)).reshape(96 * 96)
    newX = newX.reshape(-1, 1, 96, 96)
    
    return newX, y

def load2dLPFFilter(l = 5):
    X, y = load()
    newX = np.empty_like(X)
    for i in range(X.shape[0]):
        newX[i] = idct2(primLPF(dct2(X[i]), level=l)).reshape(96 * 96)
    newX = newX.reshape(-1, 1, 96, 96)
    
    return newX, y

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)



net6dctstack = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),  # !
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),  # !
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),  # !
        ('hidden4', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.1,  # !
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.2,  # !
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.3,  # !
    hidden4_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    #batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=750 * 5,
    verbose=1,
    )

import sys
sys.setrecursionlimit(10000)

X, y = load2d()
#net6.fit(X, y)

hpf10, y = load2dHPFFilter(l=10)
hpf5, y = load2dHPFFilter(l=5)
lpf5, y = load2dLPFFilter(l=5)
lpf10, y = load2dLPFFilter(l=10)
net6dctstack.fit(hpf10, y)
net6dctstack.fit(hpf5, y)
net6dctstack.fit(lpf5, y)
net6dctstack.fit(lpf10, y)
net6dctstack.fit(X, y)

import cPickle as pickle
with open('net6dctstack.pickle', 'wb') as f:
    pickle.dump(net6dctstack, f, -1)
