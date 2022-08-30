Dataset in SignDataset directory. Images are RGB (35, 35, 3). The images should be checked to be correct dimensions, as some images haven't been resized.

This repo is used to classify road speed limit signs using a convolutional neural network.

The dataset is generated from blender renders, which can be found in BlenderDatasetGenerator (.blend file downloaded through link)

The dataset also features images from https://github.com/Ekeany/Detection-and-Classification-of-Speed-Limit-Signs .
They have been resized to (35, 35, 3).

Included model has accuracy of about 0.98. It can be loaded by running cnn_classifier.py with the MODEL_NAME variable set.

To train a new model, run cnn_classifier with MODEL_NAME set to an empty string.
