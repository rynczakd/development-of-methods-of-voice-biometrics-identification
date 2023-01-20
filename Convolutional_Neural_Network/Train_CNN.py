from CNN_TrainGenerator import DataGenerator
from CNN_ValidationGenerator import ValidationGenerator
from CNN_PretrainNetwork import SiameseNetworkPreTrain


def train_CNN():

    train_generator = DataGenerator(root="/Path_to_database/",
                                    train_label_dir="/Path_to_training_labels/",
                                    batch_size=32,
                                    shuffle=True)

    validation_generator = ValidationGenerator(root="/Path_to_database/",
                                               train_label_dir="Path_to_validation_labels/",
                                               batch_size=32,
                                               shuffle=True)

    CNN_pretrain = SiameseNetworkPreTrain(batch_size=32,
                                          input_shape=(512, 300, 1),
                                          num_steps_per_epoch=1200,
                                          num_epochs=150)

    CNN_pretrain.train_model(train_generator, validation_generator)


if __name__ == '__main__':
    train_CNN()





