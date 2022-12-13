## Create a copy of resized images at the destination folder

import os
from PIL import Image


folder_from = "./Dataset_TpDne/val/TpDne/"
folder_to = "./Dataset_TpDne400/val/TpDne/"

# Target size
size = (400, 400)

for count, filename in enumerate(os.listdir(folder_from)):
    image = Image.open(f"{folder_from}/{filename}")
    try:
        # Try to change the size of the image. If successful, save it to the destination folder
        image.thumbnail(size)
        image.save(f"{folder_to}/{filename}")
    except Exception as e:
        print(filename)

    # Progress report
    if (count + 1) % 1000 == 0: 
        print(f"{count + 1}/{len(os.listdir(folder_from))}")