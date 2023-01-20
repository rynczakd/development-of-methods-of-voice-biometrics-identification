"""
Bibliografia:
    [1]. Nagrani A., Chung J. S., Xie W., Zisserman A., VoxCeleb:Large-scale speaker verification in the wild,
        South Korea, Computer Speech & Language
    [2]. Nagrani A., CHung J. S., Zisserman A., VoxCeleb: A large-scale speaker identification Dataset, UK 2017
    [3]. Schroff F., Kalenichenko D., FaceNet: A Unified Embedding for Face Recognition Clustering, 2015

Embedding Model for generating feature vectors
"""
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


class EmbeddingModel:

    def __init__(self,
                 weights_dir,
                 database_dir):

        self.weights_dir = weights_dir
        self.database_dir = database_dir
        self.input_shape = (512, 300, 1)
        self.siamese_model = None
        self.embedding_model = None
        self.embedding_dimension = (1, 1000)

    def triplet_loss(self, inputs, distance='euclidean', margin=0.2):
        anchor, positive, negative = inputs
        positive_distance = tf.keras.backend.square(anchor - positive)
        negative_distance = tf.keras.backend.square(anchor - negative)

        if distance == 'euclidean':
            positive_distance = tf.keras.backend.sqrt(tf.keras.backend.sum(positive_distance, axis=-1, keepdims=True))
            negative_distance = tf.keras.backend.sqrt(tf.keras.backend.sum(negative_distance, axis=-1, keepdims=True))

        loss = positive_distance - negative_distance

        if margin > 0:
            loss = tf.keras.backend.maximum(0.0, loss + margin)

        return tf.keras.backend.mean(loss)

    def get_model(self):

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
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

        # CONVOLUTIONAL BLOCK III ======================================================================================
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='Conv_3')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # CONVOLUTIONAL BLOCK IV =======================================================================================
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='Conv_4')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # CONVOLUTIONAL BLOCK V ========================================================================================
        x = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='Conv_5')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(5, 3), strides=(3, 2))(x)

        # FULLY CONNECTED VI ===========================================================================================
        x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
        x = tf.keras.layers.Conv2D(filters=4096, kernel_size=(9, 1), strides=(1, 1), padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='FC_6')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name='Global_Avg_Pooling')(x)
        x = tf.keras.layers.Reshape((1, 1, 4096))(x)

        # FULLY CONNECTED VII ==========================================================================================
        x = tf.keras.layers.ZeroPadding2D(padding=(0, 0))(x)
        x = tf.keras.layers.Conv2D(filters=1024, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), name='FC_7')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.Conv2D(filters=1000, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                                   kernel_regularizer=tf.keras.regularizers.l2(5e-4), name="Triplet")(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(units=1000, activation=None)(x)
        model_output = tf.keras.layers.Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1))(x)
        siamese_embedding_model = tf.keras.Model(inputs=model_input, outputs=model_output, name='Embedding_Model')

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
        self.siamese_model.load_weights(self.weights_dir)

        snn_embedding_model = self.siamese_model.layers[3]
        self.embedding_model = snn_embedding_model


if __name__ == '__main__':

    em = EmbeddingModel(weights_dir="/SNN_Model_Weights/SNN_Weights.h5",
                        database_dir="/Database_path/")
    em.get_model()



