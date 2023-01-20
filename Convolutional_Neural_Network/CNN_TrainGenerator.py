"""
Data Generator for Convolutional Neural Network Training
"""

import numpy as np
import tensorflow as tf
import os
import random


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
                 root,
                 train_label_dir,
                 batch_size,
                 spectrogram_dimension=(512, 300),
                 shuffle=True):

        self.root = root
        self.train_label_dir = train_label_dir
        self.speaker_label_train = np.load(self.train_label_dir, allow_pickle=True)[()]
        self.batch_size = batch_size
        self.spectrogram_dimension = spectrogram_dimension
        self.shuffle = shuffle
        self.speaker_indexes = list()
        self.file_indexes = list()
        self.number_of_files = 0
        self.number_of_classes = 0
        self.label_to_index_range = {}
        self.speaker_to_index = {}

        class_index = 0

        for speaker_index, file_index in self.speaker_label_train.items():
            self.speaker_indexes.append(speaker_index)
            self.speaker_to_index[speaker_index] = class_index
            for index in file_index:
                self.file_indexes.append(index)

            class_index += 1

        print('\n\tNumber of train datapoints: {}'.format(len(self.file_indexes)))
        print('\tRoot location: {}\n'.format(self.root))

        self.number_of_files = len(self.file_indexes)
        self.number_of_classes = len(self.speaker_indexes)

        self.indexes = np.arange(len(self.file_indexes))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.file_indexes) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        file_names = [self.file_indexes[file_name] for file_name in indexes]
        spectrogram, labels = self.__data_generation(file_names)

        return spectrogram, labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, file_names):
        anchor_batch = np.zeros(shape=[self.batch_size, *self.spectrogram_dimension, 1], dtype=np.float32)
        anchor_label = np.zeros(shape=[self.batch_size, 1], dtype=np.float32)

        for idx, file_id in enumerate(file_names):
            anchor_filepath = os.path.join(self.root, str(file_id) + '.npy')
            anchor_batch[idx, ] = self.__data_preprocessing(np.load(anchor_filepath))
            temp_id = self.get_speaker_id(file_id)
            anchor_label[idx, ] = self.speaker_to_index[temp_id]

        assert anchor_batch.shape[0] == self.batch_size
        assert anchor_batch.shape[1] == 512
        assert anchor_batch.shape[2] == 300
        assert anchor_batch.shape[3] == 1

        anchor = tf.constant(anchor_batch, dtype=tf.float32)
        anchor_label = tf.keras.utils.to_categorical(np.array(anchor_label), num_classes=self.number_of_classes)

        return anchor, anchor_label

    def __data_preprocessing(self, spectrogram):
        temporary_spectrogram = spectrogram[:, :, 0]
        random_columns = random.sample(range(20, 280), 50)
        for column in random_columns:
            temporary_spectrogram[:, column] = 0

        return temporary_spectrogram.reshape(*self.spectrogram_dimension, 1)

    def get_speaker_id(self, filename):
        for speaker_id, file_name in self.speaker_label_train.items():
            if filename in file_name:
                return speaker_id
