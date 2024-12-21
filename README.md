# Convolutional Neural Netwrok(CNN) for Image Classification

## Overview
The goal of this project is classify images by using Convoltional Neural Network. The dataset of this project can be found from here: https://www.kaggle.com/datasets/rrebirrth/animals-with-attributes-2. 

In this project, I have worked on 10 classes from that dataset: 
* Collie
* Dolphin
* Elephant
* Fox
* Moose
* Rabbit
* Sheep
* Squirrel
* Giant panda
* Polar bear

## Dataset
I downloaded corresponding images for these classes. The goal is create model with first 650 image for each of class. I manually deleted the rest of images. 
The initial code file is ImagePredictionCNN.ipynb. All of initial code transformation, model creation, training are done in this file. 

To detect classes and images automatically, I used Pytorch ImageFolder function which results this automatic load of dataset. Since the all data comes from same dataset, 
I used sklearn train_test_split function to divide dataset into train and test.
For this project, I used 80% of data for training and 20% of data for test. 

After this split, I transformed data with using Pytorch transforms. To result in better model performance and prevent overfitting, I used a series of transformation for images. 
The image size is 128x128 because insufficient memory of gpu.

## The CNN Model

The CNN model consists of the following layers:
* Convolutional layers with ReLU activation
* MaxPooling layers for down-sampling
* Fully connected layers
* Dropout layers to reduce overfitting

## Training model

For model training I used these hyperparameters: 
* Learning Rate : 0.001
* Optimizer: Adam 
* Batch size : 64
* Epoch : 100
* Loss Function : Cross Entropy Loss

## Model Evaluation 
I used function to show overall accuracy and accuracy for each class. And created confusion matrix to see how model made prediction for each class.

## Saving model and data
To train further, I saved the model as "cnn_trained.pth" and saved the datasets. 

## Testing model with different light adjusted images and gray world color algorithm
To show the model performance in different lights, I created new file called "TestWithManipulatedTestData.ipynb". In this file I showed what are possible drawbacks of model and how to increase performance of it.

## Further work
As base model perform well on test data, it can be lead poor result if it is testing with good captured image. It should be trained with different style of images and various image types for 
each class should be given to model.

 

