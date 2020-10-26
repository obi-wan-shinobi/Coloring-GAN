import os
from PIL import Image, ImageOps
import numpy as np

IMG_SIZE = 720
IMG_CHANNELS = 1
DIR = 'Data/'

cntr = 0
image_list = os.listdir(DIR)
while(cntr<11):
    training_data = []
    print(f'Reading batch {cntr}')
    for images in image_list[cntr*2000:(cntr+1)*2000]:
        path = os.path.join(DIR,images)
        image = Image.open(path).resize((IMG_SIZE,IMG_SIZE), Image.ANTIALIAS)
        image = ImageOps.grayscale(image)
        training_data.append(np.array(image))

    print(f'Resizing batch {cntr}')
    training_data = np.reshape(
        training_data, (-1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS))

    training_data = training_data / 127.5 - 1

    print(f'Saving batch {cntr}')
    np.save(f'training/sunset_greyscale_batch_{cntr}.npy', training_data)
    del training_data
    cntr+=1

training_data = []
print(f'Reading batch {cntr}')
for images in image_list[cntr*2000:]:
    path = os.path.join(DIR,images)
    image = Image.open(path).resize((IMG_SIZE,IMG_SIZE), Image.ANTIALIAS)
    image = ImageOps.grayscale(image)
    training_data.append(np.array(image))

print(f'Resizing batch {cntr}')
training_data = np.reshape(
    training_data, (-1, IMG_SIZE, IMG_SIZE, IMG_CHANNELS))

training_data = training_data / 127.5 - 1

print(f'Saving batch {cntr}')
np.save(f'training/sunset_greyscale_batch_{cntr}.npy', training_data)
