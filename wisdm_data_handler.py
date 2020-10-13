from __future__ import print_function
# your import statements
from keras.preprocessing.image import ImageDataGenerator
import logging
import os
import sys
import numpy as np
import pandas as pd
from keras.utils import np_utils
import warnings 
warnings.filterwarnings('ignore') # to ignore the warnings by numpy
from ibmfl.data.data_handler import DataHandler

logger = logging.getLogger(__name__)
# imports from ibmfl lib

#import all neccessary libraries
#%matplotlib inline he output of plotting commands is displayed inline within frontends

from scipy import stats
from sklearn import preprocessing

class WISDMDataHandler(DataHandler):
    """
    Data handler for your dataset.
    """
    def __init__(self, data_config=None, channels_first=False):
        super().__init__()
        self.file_name = None
        if data_config is not None:
            if 'txt_file' in data_config:
                self.file_name = data_config['txt_file']
                print(self.file_name)
            # extract other additional parameters from `info` if any.
        self.channels_first = channels_first
        # load and preprocess the training and testing data
        # self.load_and_preprocess_data()

    def read_data(self, file_path):
        column_names = ['user-id',
                        'activity',
                        'timestamp',
                        'x-axis',
                        'y-axis',
                        'z-axis']
    
        df = pd.read_csv(file_path,
                        header=None,
                        names=column_names)
        # Last column has a ";" character which must be removed ...
        df['z-axis'].replace(regex=True,
        inplace=True,
        to_replace=r';',
        value=r'')
        # ... and then this column must be transformed to float explicitly
        df['z-axis'] = df['z-axis'].apply(self.convert_to_float)
        # This is very important otherwise the model will not fit and loss
        # will show up as NAN
        df.dropna(axis=0, how='any', inplace=True)

        return df

    def create_segments_and_labels(self, df, time_steps, step, label_name):

        # x, y, z acceleration as features
        N_FEATURES = 3
        # Number of steps to advance in each iteration (for me, it should always
        # be equal to the time_steps in order to have no overlap between segments)
        # step = time_steps
        segments = []
        labels = []
        for i in range(0, len(df) - time_steps, step):
            xs = df['x-axis'].values[i: i + time_steps]
            ys = df['y-axis'].values[i: i + time_steps]
            zs = df['z-axis'].values[i: i + time_steps]
            # Retrieve the most often used label in this segment
            label = stats.mode(df[label_name][i: i + time_steps])[0][0]
            segments.append([xs, ys, zs])
            labels.append(label)

        # Bring the segments into a better shape
        reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
        labels = np.asarray(labels)

        return reshaped_segments, labels

    def convert_to_float(self, x):
        try:
            return np.float(x)
        except:
            return np.nan

    def load_and_preprocess_data(self):
        """
        Loads and pre-processeses local datasets, 
        and updates self.x_train, self.y_train, self.x_test, self.y_test.
        """
        logger.info('Loaded training data from ' + str(self.file_name))
        # "datasets/" + 
        df = self.read_data(self.file_name)

        # Define column name of the label vector
        LABEL = 'ActivityEncoded'
        # Transform the labels from String to Integer via LabelEncoder
        le = preprocessing.LabelEncoder()
        # Add a new column to the existing DataFrame with the encoded values
        df[LABEL] = le.fit_transform(df['activity'].values.ravel())

        #keep users with ID 1 to 28 for training the model and users with ID greater than 28 for the test set.
        df_test = df[df['user-id'] > 28]
        df_train = df[df['user-id'] <= 28]

        # Normalize features for training data set (values between 0 and 1)
        # Surpress warning for next 3 operation
        pd.options.mode.chained_assignment = None  # default='warn'
        df_train['x-axis'] = df_train['x-axis'] / df_train['x-axis'].max()
        df_train['y-axis'] = df_train['y-axis'] / df_train['y-axis'].max()
        df_train['z-axis'] = df_train['z-axis'] / df_train['z-axis'].max()
        
        #Normalizing Testing
        df_test['x-axis'] = df_test['x-axis'] / df_test['x-axis'].max()
        df_test['y-axis'] = df_test['y-axis'] / df_test['y-axis'].max()
        df_test['z-axis'] = df_test['z-axis'] / df_test['z-axis'].max()        
        
        # Round numbers
        df_train = df_train.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})
        df_test = df_test.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})

        # LABELS = ['Downstairs',
        #         'Jogging',
        #         'Sitting',
        #         'Standing',
        #         'Upstairs',
        #         'Walking']
        # The number of steps within one time segment
        TIME_PERIODS = 80
        # The steps to take from one segment to the next; if this value is equal to
        # TIME_PERIODS, then there is no overlap between the segments
        STEP_DISTANCE = 40
        LABEL = 'ActivityEncoded'

        x_train, y_train = self.create_segments_and_labels(df_train,
                                                            TIME_PERIODS, 
                                                            STEP_DISTANCE,
                                                            LABEL)
        x_test, y_test = self.create_segments_and_labels(df_test,
                                                            TIME_PERIODS,
                                                            STEP_DISTANCE,
                                                            LABEL)

        print('x_train shape: ', x_train.shape)
        print(x_train.shape[0], 'training samples')
        print('y_train shape: ', y_train.shape)

        # Set input & output dimensions
        # num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
        num_classes = le.classes_.size
        print(list(le.classes_))

        # input_shape = (num_time_periods, num_sensors)
        # x_train = x_train.reshape(x_train.shape[0], input_shape)
        # print('x_train shape:', x_train.shape)
        # print('input_shape:', input_shape)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')

        num_classes = 6

        #execute the following line only once
        y_train_hot = np_utils.to_categorical(y_train, num_classes)
        print('New y_train shape: ', y_train_hot.shape)
        print(x_train[0])
        print(y_train[0])
        print(x_test[0])
        print(y_test[0])
        return (x_train, y_train), (x_test, y_test)  


    def get_data(self):
        """
        Gets the prepared training and testing data.
        
        :return: ((x_train, y_train), (x_test, y_test)) # most build-in training modules expect data is returned in this format
        :rtype: `tuple` 
        """

        if '.npz' in self.file_name:
            npz = np.load(self.file_name) 
            print(f"y_train shape: {npz['y_train'].shape}")
            return (npz['x_train'], npz['y_train']), (npz['x_test'], npz['y_test'])

        print("file_name: {}".format(self.file_name))
        (x_train, y_train), (x_test, y_test) = self.load_and_preprocess_data()
        return (x_train, y_train), (x_test, y_test)
