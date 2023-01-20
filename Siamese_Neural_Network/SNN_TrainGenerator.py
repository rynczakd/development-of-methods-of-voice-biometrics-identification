"""
Data Generator for Siamese Neural Network Training
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
                 spectrogram_dimension=(512, 300)):

        self.root = root
        self.train_label_dir = train_label_dir
        self.speaker_label_train = np.load(self.train_label_dir, allow_pickle=True)[()]
        self.batch_size = batch_size
        self.spectrogram_dimension = spectrogram_dimension
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
     
    def __len__(self):
        return int(np.floor(len(self.file_indexes) / self.batch_size))

    def __getitem__(self, index):
        speaker_labels = np.random.choice(self.speaker_indexes, size=self.batch_size + 1, replace=False)
        spectrogram, labels = self.__data_generation(speaker_labels=speaker_labels)
        return spectrogram, labels

    def __data_generation(self, speaker_labels):

        anchor_batch = np.zeros(shape=[self.batch_size + 1, *self.spectrogram_dimension, 1], dtype=np.float32)
        positive_batch = np.zeros(shape=[self.batch_size + 1, *self.spectrogram_dimension, 1], dtype=np.float32)
        negative_batch = np.zeros(shape=[self.batch_size + 1, *self.spectrogram_dimension, 1], dtype=np.float32)

        anchor_label = np.zeros(shape=[self.batch_size + 1, 1], dtype=np.float32)
        positive_label = np.zeros(shape=[self.batch_size + 1, 1], dtype=np.float32)
        negative_label = np.zeros(shape=[self.batch_size + 1, 1], dtype=np.float32)

        for idx, speaker_id in enumerate(speaker_labels):
            spectre_indexes = np.random.choice(self.speaker_label_train[speaker_id], size=2, replace=False)
            anchor_filepath = os.path.join(self.root, str(spectre_indexes[0]) + '.npy')
            positive_filepath = os.path.join(self.root, str(spectre_indexes[1]) + '.npy')
            anchor_batch[idx, ] = self.__data_preprocessing(np.load(anchor_filepath))
            positive_batch[idx, ] = self.__data_preprocessing(np.load(positive_filepath))

            anchor_label[idx, ] = self.speaker_to_index[speaker_id]
            positive_label[idx, ] = self.speaker_to_index[speaker_id]

        negative_batch = anchor_batch[1:]
        negative_label = anchor_label[1:]

        anchor_batch = anchor_batch[:-1]
        anchor_label = anchor_label[:-1]
        positive_batch = positive_batch[:-1]
        positive_label = positive_label[:-1]

        labels = np.array([anchor_label, positive_label, negative_label])

        assert anchor_batch.shape[0] == self.batch_size
        assert anchor_batch.shape[1] == 512
        assert anchor_batch.shape[2] == 300
        assert anchor_batch.shape[3] == 1

        assert positive_batch.shape[0] == self.batch_size
        assert positive_batch.shape[1] == 512
        assert positive_batch.shape[2] == 300
        assert positive_batch.shape[3] == 1

        assert negative_batch.shape[0] == self.batch_size
        assert negative_batch.shape[1] == 512
        assert negative_batch.shape[2] == 300
        assert negative_batch.shape[3] == 1

        anchor = tf.constant(anchor_batch, dtype=tf.float32)
        positive = tf.constant(positive_batch, dtype=tf.float32)
        negative = tf.constant(negative_batch, dtype=tf.float32)

        return [anchor, positive, negative], labels

    def __data_preprocessing(self, spectrogram):
        temporary_spectrogram = spectrogram[:, :, 0]
        random_columns = random.sample(range(20, 280), 50)
        for column in random_columns:
            temporary_spectrogram[:, column] = 0

        return temporary_spectrogram.reshape(*self.spectrogram_dimension, 1)
