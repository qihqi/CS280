import os
import sys
import numpy as np
import scipy.misc
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils.np_utils import to_categorical

val_path = './val/val'

def get_mini_batch(size):
    buf = []
    count = 0
    for i in sorted(os.listdir(val_path)):
        full = os.path.join(val_path, i)
        img = img_to_array(load_img(full, target_size=(128,128)))
        img -= np.mean(img, axis=0, keepdims=True) 
        buf.append(img.reshape((1, 3, 128, 128)))
        if  (count + 1) % size == 0:
            yield np.vstack(buf).astype(np.float32)
            buf = []
        count += 1

model = load_model(sys.argv[1])
print model.summary()
correct = 0
data = np.load('../../validation_fc6.npy')
ytrue = data[()]['y']
for i, batch in enumerate(get_mini_batch(32)):
    y = model.predict(batch)
    start = i * 32
    end = start + 32
    correct += sum(np.argmax(y, axis=1) == ytrue[start:end])
print correct

