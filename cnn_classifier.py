import os
# dont use GPU, causes error on my pc
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import Adam
from keras.losses import *
import keras

import matplotlib.pyplot as plt

# importing automatically loads dataset
import open_dataset
import random
import numpy as np


def show_off_predictions(model, x, y):
    """
    For each feature and label in dataset, print predicted speed, actual speed and display the sign
    :param model: a keras model to use for predictions
    :param x: a list of numpy flattened images (35x35x3)
    :param y: a list of one hot encoded correct classifications
    """
    for im, classification in zip(x, y):
        prediction = model.predict(np.array([im]), verbose=0)[0].tolist()

        im_shaped = np.array(im).reshape((35, 35, 3))
        plt.imshow(im_shaped)

        prediction_name = list(open_dataset.category_mappings.keys())[prediction.index(max(prediction))]
        print("Prediction:", prediction_name)
        actual_name = list(open_dataset.category_mappings.keys())[classification.index(1)]
        print("Actual:", actual_name)

        plt.show()


# If a model name is given, it is loaded and used to predict, else a model is trained on the dataset
MODEL_NAME = "best_model_0.96"
# fraction out of 1 of the dataset to use in training
TRAIN_FRACTION = 0.5

# Prepare dataset for training
print("Preparing Dataset...")

random.shuffle(open_dataset.dataset)

# split into test and training data
train_amount = int(len(open_dataset.dataset)*TRAIN_FRACTION)
train_data, test_data = open_dataset.dataset[:train_amount], open_dataset.dataset[train_amount:]
train_data_x, train_data_y = zip(*train_data)
test_data_x, test_data_y = zip(*test_data)

print("Prepared Dataset")

if MODEL_NAME:
    # if model name is given, simply make predictions on all of the dataset and exit
    nn = keras.models.load_model(MODEL_NAME)

    print("LOADED MODEL")

    all_x = train_data_x+test_data_x
    all_y = train_data_y+test_data_y

    score = nn.evaluate(np.array(all_x), np.array(all_y))

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    show_off_predictions(nn, all_x, all_y)
    exit()

# =============== TRAINING NEW NETWORK ===============

# CREATE CONVOLUTIONAL NEURAL NETWORK
nn = Sequential()
nn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(35, 35, 3)))
nn.add(MaxPooling2D((2, 2)))
nn.add(Conv2D(64, (3, 3), activation="relu", input_shape=(35, 35, 3)))
nn.add(MaxPooling2D((2, 2)))
nn.add(Conv2D(128, (3, 3), activation="relu", input_shape=(35, 35, 3)))
nn.add(Flatten())

nn.add(Dense(64, activation="relu"))
nn.add(Dense(32, activation="relu"))
nn.add(Dense(16, activation="relu"))
nn.add(Dense(5, activation="relu"))

nn.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss=BinaryCrossentropy(),
    metrics=["accuracy"]
)

nn.summary()

# TRAIN NEURAL NETWORK ON DATASET
nn.fit(np.array(train_data_x), np.array(train_data_y), epochs=100)

# EVALUATE AND SAVE NEURAL NETWORK
score = nn.evaluate(np.array(test_data_x), np.array(test_data_y))

print('Test loss:', score[0])
print('Test accuracy:', score[1])

nn.save("model_" + str(round(score[1], 2)))

show_off_predictions(nn, test_data_x, test_data_y)
