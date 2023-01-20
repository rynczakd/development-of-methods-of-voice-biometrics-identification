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


class SiameseNeuralNetwork:

    def __init__(self,
                 input_shape,
                 output_shape,
                 num_epochs,
                 num_steps_per_epoch,
                 batch_size,
                 loss_margin):

        self.siamese_model = None
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_epochs = num_epochs
        self.num_steps_per_epoch = num_steps_per_epoch
        self.batch_size = batch_size
        self.loss_margin = loss_margin
        self.cnn_model = "/Path_to_pretrained_CNN_Model/CNN_Model.hdf5"
        self.logs_path = "/Logs_path/"
        self.model_filepath = "/Model_filepath/Model.hdf5"
        self.model_filepath_10 = "/Model_filepath/Model_epochs.hdf5"
        self.history_path = "/History_path/History_SNN.csv"
        self.weights_filepath = '/Weights_filepath/Weights.h5'
        self.weights_5_filepath = '/Weights_filepath/Weights_epoch.h5'

    @staticmethod
    def custom_schedule_learning_rate(epoch, learning_rate):
        if epoch > 0:
            return learning_rate * 0.85
        return learning_rate

    def triplet_loss(self, inputs, distance='euclidean', margin=0.2):
        anchor, positive, negative = inputs
        positive_distance = tf.keras.backend.square(anchor - positive)
        negative_distance = tf.keras.backend.square(anchor - negative)

        if distance == 'euclidean':
            positive_distance = tf.keras.backend.sqrt(tf.keras.backend.sum(positive_distance, axis=-1, keepdims=True))
            negative_distance = tf.keras.backend.sqrt(tf.keras.backend.sum(negative_distance, axis=-1, keepdims=True))

        loss = positive_distance - negative_distance

        if margin > 0:
            loss = tf.keras.backend.maximum(0.0, loss + self.loss_margin)

        return tf.keras.backend.mean(loss)

    def build_model(self):

        cnn_model = tf.keras.models.load_model(self.cnn_model)
        cnn_out = cnn_model.get_layer("activation_6")

        x = tf.keras.layers.Conv2D(filters=1000, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), name="Triplet")(cnn_out.output)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=1000, activation=None)(x)
        model_output = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1))(x)

        siamese_embedding_model = tf.keras.Model(cnn_model.input, outputs=model_output, name='Embedding_Model')

        anchor_input = tf.keras.layers.Input(self.input_shape, name='Anchor')
        positive_input = tf.keras.layers.Input(self.input_shape, name='Positive')
        negative_input = tf.keras.layers.Input(self.input_shape, name='Negative')

        anchor_embedding = siamese_embedding_model(anchor_input)
        positive_embedding = siamese_embedding_model(positive_input)
        negative_embedding = siamese_embedding_model(negative_input)

        siamese_inputs = [anchor_input, positive_input, negative_input]
        siamese_outputs = [anchor_embedding, positive_embedding, negative_embedding]

        siamese_network_model = tf.keras.Model(siamese_inputs, siamese_outputs)
        siamese_network_model.add_loss(tf.keras.backend.mean(self.triplet_loss(siamese_outputs)))

        self.siamese_model = siamese_network_model
        print(self.siamese_model.summary())

    def compile_model(self):
        opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        self.siamese_model.compile(optimizer=opt, loss=None)

    def train_model(self, train_generator, validation_generator):

        lr_schedule = tf.keras.callbacks.LearningRateScheduler(self.custom_schedule_learning_rate,
                                                               verbose=1)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          mode='min',
                                                          patience=10)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.model_filepath,
                                                        monitor='loss',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        mode='min')

        checkpoint_10 = tf.keras.callbacks.ModelCheckpoint(self.model_filepath_10,
                                                           monitor='loss',
                                                           verbose=1,
                                                           period=5)

        tensorboard = tf.keras.callbacks.TensorBoard(log_dir=self.logs_path,
                                                     histogram_freq=1,
                                                     embeddings_freq=1)
        
        weight_saver = tf.keras.callbacks.ModelCheckpoint(self.weights_filepath,
                                                          monitor='loss',
                                                          verbose=1,
                                                          save_best_only=True,
                                                          save_weights_only=True,
                                                          mode='min')

        weight_saver_5 = tf.keras.callbacks.ModelCheckpoint(self.weights_5_filepath,
                                                            monitor='loss',
                                                            verbose=1,
                                                            save_weights_only=True,
                                                            period=5)

        csv_logger = tf.keras.callbacks.CSVLogger(self.history_path, separator=',', append=True)

        callbacks_list = [tensorboard, lr_schedule, checkpoint, checkpoint_10,
                          csv_logger, weight_saver, weight_saver_5, early_stopping]

        self.build_model()
        self.compile_model()

        train_history = self.siamese_model.fit(train_generator,
                                               steps_per_epoch=self.num_steps_per_epoch,
                                               validation_data=validation_generator,
                                               verbose=1,
                                               epochs=self.num_epochs,
                                               use_multiprocessing=True,
                                               workers=4,
                                               callbacks=callbacks_list)


