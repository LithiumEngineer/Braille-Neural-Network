import os
import numpy as np
from PIL import Image
import string

def load_dataset(dir):
    images = []
    labels = []
    for filename in os.listdir(dir):
        if 'rot' in filename:
            continue

        label = filename[0].lower()
        
        img_path = os.path.join(dir, filename)
        img = Image.open(img_path).convert('L') # Don't need resize, images are guaranteed to be 28x28
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_flat = img_arr.flatten()

        images.append(img_flat)
        labels.append(ord(label) - ord('a'))

    X = np.array(images).T
    y_indices = np.array(labels)

    Y = np.zeros((26, len(y_indices)))
    Y[y_indices, np.arange(len(y_indices))] = 1

    return X, Y