#!/usr/bin/env python
# coding: utf-8

# In[1]:


import errno
import hashlib
import os
import shutil
import tarfile
import time
import urllib
import warnings
from zipfile import ZipFile

import numpy as np
import torch
import torchaudio
from torch.utils.model_zoo import tqdm  # type: ignore # tqdm exists in model_zoo
from torchvision import transforms


import ai8x
import re


# In[ ]:


class WD:

    class_dict = {
        'bicepcurl': 0, 'tricepextension': 1, 'lateralraise': 2
    }
    
    dataset_dict = {
        'WD': ('Bicep Curl','Tricep Extension', 'Lateral Raise')
    }

    TRAIN = np.uint(0)
    TEST = np.uint(1)
    VALIDATION = np.uint(2)
    BENCHMARK = np.uint(3)

    def __len__(self):
        return self.dataset_len

    def __init__(self, root, classes, d_type, dataset_len, exp_len=200, desired_probs=None,
                 transform=None, quantize=False, download=False):
        
        self.root = root
        self.classes = classes
        self.d_type = d_type
        self.transform = transform
        self.dataset_len = dataset_len
        self.desired_probs = desired_probs
        self.exp_len = exp_len

        self.workout_train_folder = os.path.join(self.root, 'workout_train')
        self.workout_test_folder = os.path.join(self.root, 'workout_test')

        self.data, self.targets, self.data_type= self.__gen_datasets()

        # rms values for each sample to be returned
        #self.rms = np.zeros(self.dataset_len)
    @staticmethod
    def pad_sequences(sequences, max_len=None, padding_value=0):
        # Find the maximum length of sequences if not provided
        if max_len is None:
            max_len = max(len(seq) for seq in sequences)
        
        # Determine the feature size (assuming all sequences have the same feature size)
        feature_size = len(sequences[0][0]) if sequences else 0
        # Initialize the padded array
        padded_sequences = np.full((len(sequences), max_len, feature_size), padding_value, dtype=float)

        # Pad each sequence
        for i, seq in enumerate(sequences):
            for j, item in enumerate(seq):
                if j < max_len:
                    padded_sequences[i, j, :len(item)] = item       
        return padded_sequences
        
    @staticmethod
    def extract_numeric_values(text):
        # Regular expression to match all numbers (including decimals)
        pattern = re.compile(r'[-+]?\d*\.\d+|\d+')
        return [float(match) for match in pattern.findall(text)]
    
    @staticmethod
    def read_and_process_file(file_path):
        with open(file_path, 'r') as file:
            data = file.read()

        # Split data into lines if needed
        lines = data.split('\n')
        all_values = []
        accel_x = []
        accel_y = []
        accel_z = []
        gyro_x = []
        gyro_y = []
        gyro_z = []
        
        # Extract numerical values from each line
        for line in lines:
            if line.strip():  # Ensure the line is not empty
                values = WD.extract_numeric_values(line)
                accel_x.append(values[0])
                accel_y.append(values[1])
                accel_z.append(values[2])
                gyro_x.append(values[3])
                gyro_y.append(values[4])
                gyro_z.append(values[5])

        all_values.append(accel_x)
        all_values.append(accel_y)
        all_values.append(accel_z)
        all_values.append(gyro_x)
        all_values.append(gyro_y)
        all_values.append(gyro_z)

        return np.array(all_values)

    def __getitem__(self, index):
        rnd_num = np.random.choice(range(len(self.data)))
        rnd_num_2 =np.random.choice(range(len(self.data[rnd_num])))
        rec_len = len(self.data[rnd_num][rnd_num_2])
        max_start_idx = rec_len - 2
        start_idx = 0
        end_idx = start_idx + 2

        rand_inp = self.data[index]
        inp = torch.tensor(rand_inp)

        target = int(self.targets[index])

        return inp, target

        
    @staticmethod
    def calculate_rms(data):
        acc_data = data[:, 0, :]  # Accelerometer data
        gyro_data = data[:, 1, :]  # Gyroscope data
        combined_data = np.hstack((acc_data, gyro_data))
        flattened_data = combined_data.flatten()
        squared = np.square(flattened_data)
        mean_squared = np.mean(squared, axis=0)
        rms = np.sqrt(mean_squared)
        return rms    
    @staticmethod
    def normalize(data):

        """        acc_data = data[]
        acc_data = data[:, 0, :]  # Accelerometer data
        gyro_data = data[:, 1, :]  # Gyroscope data"""

        # Find min and max for accelerometer data
        acc_mean = np.mean(data[:3])
        gyro_mean = np.mean(data[3:])

        acc_std = np.std(data[:3])
        gyro_std = np.std(data[3:])       
        """        min_acc = np.min(acc_data, axis=0)
                max_acc = np.max(acc_data, axis=0)"""

        norm_acc_data = (data[:3] - acc_mean) / acc_std
        norm_gyro_data = (data[3:] - gyro_mean) / gyro_std
        
        # Find min and max for gyroscope data
        """ min_gyro = np.min(gyro_data, axis=0)
        max_gyro = np.max(gyro_data, axis=0)
        
        norm_acc_data = (acc_data - min_acc) / (max_acc - min_acc)
        norm_gyro_data = (gyro_data - min_gyro) / (max_gyro - min_gyro)"""
    # Combine normalized data
        normalized_data = np.stack((norm_acc_data, norm_gyro_data), axis=0)
        normalized_data = normalized_data.reshape(6,200)
        return normalized_data

        
                
        
    def __gen_datasets(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')

            labels = list(self.classes)
            print(os.getcwd())
            train_list = sorted(os.listdir(self.workout_train_folder))
            test_list = sorted(os.listdir(self.workout_test_folder))


            if self.d_type == 'train':
                workout_folder = [self.workout_train_folder]
            elif self.d_type == 'test':
                workout_folder = [self.workout_test_folder]

            if self.desired_probs is None or len(self.desired_probs) != len(labels):
                self.desired_probs = []
                print('Each class will be selected using the same probability!')
                label_count = len(labels)
                for i in range(label_count):
                    self.desired_probs.append(1/label_count)

            elif np.sum(self.desired_probs) != 1:
                print('Sum of the probabilities is not 1!\n')
                print('Carrying out the normal probability distribution.')
                self.desired_probs = self.desired_probs / np.sum(self.desired_probs)

            self.data_class_count = {}
            data_in = []
            data_type = []
            data_class = []
            rms_val = []

            for i, label in enumerate(labels):
                count = 0
                for folder in workout_folder:
                    for record_name in sorted(os.listdir(folder)):
                        if record_name.split('_')[0] in label:
                            record_path = os.path.join(folder, record_name)
                            record = self.read_and_process_file(record_path)
                            #data_in.append(self.normalize(record))
                            data_in.append(record)
                            
                            if folder == self.workout_train_folder:
                                data_type.append(0)  # train + val
                            elif folder == self.workout_test_folder:
                                data_type.append(1)  # test
                            data_class.append(i)
                            count += 1
                self.data_class_count[label] = count
            workout_dataset = (data_in, data_class, data_type)
            final_probs = np.zeros(len(data_in))
            idx = 0
            for i, label in enumerate(labels):
                for _ in range(self.data_class_count[label]):
                    final_probs[idx] = self.desired_probs[i]/self.data_class_count[label]
                    idx += 1
            self.final_probs = final_probs
        return workout_dataset                



# In[ ]:


def WD_get_datasets(data, desired_probs=None, train_len=192, test_len=53,
                                     load_train=True, load_test=True):
    (data_dir, args) = data

    classes = ['BicepCurl','TricepExtension', 'LateralRaise']

    transform = None
    quantize = False

    if load_train:
        train_dataset = WD(root="../ai8x-training/data", classes=classes, d_type='train',
                                dataset_len=train_len, desired_probs=desired_probs,
                                transform=transform,
                                quantize=quantize, download=False)
    else:
        train_dataset = None

    if load_test:
        test_dataset = WD(root="../ai8x-training/data", classes=classes, d_type='test',
                                dataset_len=test_len, desired_probs=desired_probs,
                                transform=transform,
                                quantize=quantize, download=False)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


# In[ ]:


datasets = [
    {
        'name': 'WD',
        'input': (6, 200),
        'output': (0, 1, 2),
        'loader': WD_get_datasets,
    }
]

