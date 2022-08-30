"""
Resize all images in a directory, to (35, 35), and save with a suffix '_resized'
"""

import os
from PIL import Image

ROOT = "SignDataset/"

files = os.walk(ROOT)

for directory, contained_directories, directory_files in files:
    for file_name in directory_files:
        if file_name.endswith("jpg") and not file_name.endswith("_resized.jpg"):
            file_path = os.path.join(directory, file_name)
            image: Image.Image = Image.open(file_path)
            if image.size != (35, 35):
                resized = image.resize((35, 35), resample=Image.Resampling.NEAREST)
                new_name = os.path.join(directory, file_name.split(".")[0] + "_resized.jpg")
                resized.save(os.path.join(directory, new_name))
