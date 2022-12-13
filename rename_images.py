## Rename all images in a given folder.

import os

folder = "./Dataset_Mixed/val/TpDne"

format = "jpg"

for count, filename in enumerate(os.listdir(folder)):
    dst = f"td{str(count)}.{format}"  # New name.
    src =f"{folder}/{filename}"  # Previous name.
    dst =f"{folder}/{dst}"

    os.rename(src, dst)