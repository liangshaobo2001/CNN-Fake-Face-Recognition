## Truncated images cannot be loaded. Please use this script to find them and replace with another good image.

import os
from PIL import Image

# The folder from which the images are obtained.
folder_from = "./Dataset_TpDne/val/TpDne"

for count, filename in enumerate(os.listdir(folder_from)):
    image = Image.open(f"{folder_from}/{filename}")
    try:
        # Try to open that image and load the data.
        image.load()
    except Exception as e:
        # Print image name if the image cannot be loaded.
        print(filename)

    # Progress bar
    if count%100 == 0: 
        print(count)