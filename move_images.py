## Move images to the train/val/test folders used in the mode..

import os

total_count = 30000;

source = "./Image_TpDne/"
train_destination = "./Dataset_TpDne/train/TpDne/"
val_destination = "./Dataset_TpDne/val/TpDne/"
test_destination = "./Dataset_TpDne_Test/test/TpDne/"
image_type = "jpg"

for count, filename in enumerate(os.listdir(source)):
    if count < int(total_count * 0.7):
        os.rename(source + f"{str(count)}.{image_type}", train_destination + f"{str(count)}.{image_type}")
    elif count < int(total_count * 0.9):
        os.rename(source + f"{str(count)}.{image_type}", val_destination + f"{str(count)}.{image_type}")
    elif count < int(total_count):
        os.rename(source + f"{str(count)}.{image_type}", test_destination + f"{str(count)}.{image_type}")