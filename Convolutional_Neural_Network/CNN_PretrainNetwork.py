"""
Bibliografia:
    [1]. Nagrani A., Chung J. S., Xie W., Zisserman A., VoxCeleb:Large-scale speaker verification in the wild,
        South Korea, Computer Speech & Language
    [2]. Nagrani A., CHung J. S., Zisserman A., VoxCeleb: A large-scale speaker identification Dataset, UK 2017
    [3]. Schroff F., Kalenichenko D., FaceNet: A Unified Embedding for Face Recognition Clustering, 2015
"""

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class SiameseNetworkPreTrain:

    def __init__(self,
                 input_shape,
                 num_epochs,
                 num_steps_per_epoch,
                 batch_size):

        self.input_shape = input_shape
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self.batch_size = batch_size
        self.logs_path = "/Logs_filepath/"
        self.model_filepath = "/Model_filepath/Model.hdf5"
        self.model_filepath_10 = "/Model_filepath/Model_epochs.hdf5"
        self.history_path = "/History_filepath/History.csv"
        self.model = None

    @staticmethod
    def custom_schedule_learning_rate(epoch, learning_rate):
        if epoch > 0:
            return learning_rate * 0.85
        return learning_rate

    def build_model(self):

        # INPUT ========================================================================================================
        model_input = tf.keras.layers.Input(shape=self.input_shape, name='Model_Input')

        # CONVOLUTIONAL BLOCK I ========================================================================================
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(model_input)
        x = tf.keras.layers.Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='Conv_1')(x)
        x = tf.keras.layers.BatchNormalization(name='Batch_Norm_1')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        # CONVOLUTIONAL BLOCK II =======================================================================================
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='Conv_2')(x)
        x = tf.keras.layers.BatchNormalization(name='Batch_Norm_2')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        # CONVOLUTIONAL BLOCK III ======================================================================================
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='Conv_3')(x)
        x = tf.keras.layers.BatchNormalization(name='Batch_Norm_3')(x)
        x = tf.keras.layers.Activation('relu')(x)

        # CONVOLUTIONAL BLOCK IV =======================================================================================
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='Conv_4')(x)
        x = tf.keras.layers.BatchNormalization(name='Batch_Norm_4')(x)
        x = tf.keras.layers.Activation('relu')(x)

        # CONVOLUTIONAL BLOCK V ========================================================================================
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='Conv_5')(x)
        x = tf.keras.layers.BatchNormalization(name='Batch_Norm_5')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(5, 3), strides=(3, 2))(x)

        # FULLY CONNECTED VI ===========================================================================================
        x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
        x = tf.keras.layers.Conv2D(filters=4096, kernel_size=(9, 1), strides=(1, 1), padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='FC_6')(x)
        x = tf.keras.layers.BatchNormalization(name='Batch_Norm_6')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name='Global_Avg_Pooling')(x)
        x = tf.keras.layers.Reshape((1, 1, 4096))(x)

        # FULLY CONNECTED VII ==========================================================================================
        x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
        x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='FC_7')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # LAST LAYERS ==================================================================================================
        x = tf.keras.layers.Conv2D(filters=1211, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), name="FC_8")(x)
        x = tf.keras.layers.Flatten()(x)
        model_output = tf.keras.layers.Activation('softmax')(x)

        pretrain_model = tf.keras.models.Model(inputs=model_input, outputs=model_output, name='Pretrain_CNN')
        print(pretrain_model.summary())
        self.model = pretrain_model

    def compile_model(self):
        opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        self.model.compile(optimizer=opt, loss="categorical_crossentropy",
                           metrics=["categorical_accuracy", "top_k_categorical_accuracy"])

    def train_model(self, train_generator, validation_generator):

        lr_schedule = tf.keras.callbacks.LearningRateScheduler(self.custom_schedule_learning_rate,
                                                               verbose=1)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',
                                                          patience=20)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.model_filepath,
                                                        monitor='val_categorical_accuracy',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        mode='max')

        checkpoint_10 = tf.keras.callbacks.ModelCheckpoint(self.model_filepath_10,
                                                           monitor='val_categorical_accuracy',
                                                           verbose=1,
                                                           period=5)

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.logs_path,
                                                     histogram_freq=1,
                                                     embeddings_freq=1)

        csv_logger = tf.keras.callbacks.CSVLogger(self.history_path, separator=',', append=True)

        callbacks_list = [tensorboard, lr_schedule, early_stopping, checkpoint, checkpoint_10, csv_logger]

        self.build_model()
        self.compile_model()

        train_history = self.model.fit(train_generator,
                                       steps_per_epoch=self.num_steps_per_epoch,
                                       validation_data=validation_generator,
                                       verbose=1,
                                       epochs=self.num_epochs,
                                       use_multiprocessing=True,
                                       workers=6,
                                       callbacks=callbacks_list)
