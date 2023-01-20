from SNN_TrainGenerator import DataGenerator
from SNN_ValidationGenerator import ValidationGenerator
from SNN_Model import SiameseNeuralNetwork

def train_SNN():

    train_generator = DataGenerator(root="/Database_path/",
                                    train_label_dir="/Training_labels_filepath/",
                                    batch_size=32)

    validation_generator = ValidationGenerator(root="/Database_path/",
                                               train_label_dir="/Validation_labels_filepath/",
                                               batch_size=32)

    SNN = SiameseNeuralNetwork(input_shape=(512, 300, 1),
                               output_shape=1000,
                               num_epochs=80,
                               num_steps_per_epoch=1200,
                               batch_size=32,
                               loss_margin=0.2)

    SNN.train_model(train_generator, validation_generator)


if __name__ == '__main__':
    train_SNN()
