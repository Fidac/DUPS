
from tensorflow.keras.layers import Input, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Flatten, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_recommenders as tfrs
import tensorflow as tf
import keras
from collections import OrderedDict

