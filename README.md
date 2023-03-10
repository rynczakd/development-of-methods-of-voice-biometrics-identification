# Development of methods of voice biometrics identification
This repository contains an implementation of the *VGGVox* Convolutional Neural Network, which was used to classify speakers from the VoxCeleb dataset. 
This repository also includes an implementation of Siamese Neural Network with a Triplet Loss function, which was created based on a pre-trained *VGGVox* CNN. 
SNN was used to generate embeddings for speakers that enabled them to be identified.

## Project structure
1. Convolutional_Neural_Network:
   - CNN_PretrainNetwork.py - implementation of the CNN model
   - CNN_TrainGenerator.py - custom data generator for CNN training
   - CNN_ValidationGenerator.py - custom data generator for CNN validation process
   - Train_CNN.py - lauching the learning process

2. Data_Preparation_Tool:
   - DataPreparationTool.py - data pre-processing and spectrograms computation
   - SpectrogramGenerator.py - filtering and STFT computation

3. Siamese_Neural_Network:
   - SNN_Embedding_Model.py - implementation of the single SNN model for embeddings generation
   - SNN_Model.py - implementation of the SNN model with Triplet Loss function
   - SNN_TrainGenerator.py - custom data generator for SNN training
   - SNN_ValidationGenerator.py - custom data generator for SNN validation process
   - Train_SNN.py - lauching the learning process


*SNN_Embedding_Model.py* can be used to generate embedding for speakers.

### How to use:     
1. Train_CNN.py - this file allows you to start the CNN learning process after specifying the file paths in scripts from the Convolutional_Neural_Network directory in advance.     
2. SNN_Model.py - this file allows you to start the SNN learning process after specifying the file paths in scripts from the Siamese_Neural_Network directory in advance.     
3. SNN_Embedding_Model.py - this file allows you to generate embeddings based on (512x300x1 size) spectrograms after loading the model weights.

### Requirements:
    - Numpy 1.19.5
    - SciPy - Signal module 1.6.2
    - Scikit-Learn 0.24.2
    - TensorFlow 2.2.0
    - Keras 2.4.3

## Bibliography:
    [1]. Nagrani A., Chung J. S., Xie W., Zisserman A., VoxCeleb:Large-scale speaker verification in the wild,
        South Korea, Computer Speech & Language
    [2]. Nagrani A., CHung J. S., Zisserman A., VoxCeleb: A large-scale speaker identification Dataset, UK 2017
    [3]. Schroff F., Kalenichenko D., FaceNet: A Unified Embedding for Face Recognition Clustering, 2015
    [4]. Sainburg T.:
    https://timsainburg.com/python-mel-compression-inversion.html#python-mel-compression-inversion
