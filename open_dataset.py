from PIL import Image
import os
import numpy as np

ROOT = "SignDataset/"
files = os.walk(ROOT)

category_mappings = {
    "20Kph": 0,
    "30Kph": 1,
    "50Kph": 2,
    "80Kph": 3,
    "100Kph": 4
}


def one_hot(max_index, index):
    # one hot encode
    # e.g. with max_index 4, and index 2
    # returns: [0, 0, 1, 0, 0]

    empty = [0] * (max_index+1)
    empty[index] = 1
    return empty


dataset = []
for directory, subdirectories, file_names in files:
    for file_name in file_names:
        # name of directory (20Kph, etc.)
        category_name = os.path.split(directory)[-1]

        try:
            category = category_mappings[category_name]
        except:
            # dont use this folder if not in category_mappings
            continue

        im: Image.Image = Image.open(os.path.join(directory, file_name))

        if im.size != (35, 35):
            # not sized correctly
            continue

        im_data = np.asarray(im)

        category_one_hot_encoded = one_hot(len(category_mappings)-1, category)

        # add to dataset, with the image scaled to 0-1
        dataset.append(((im_data/255).tolist(), category_one_hot_encoded))
