from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import DirectoryIterator
import cv2, numpy as np
import sys

TRAIN_ROOT = './images/train2'

def cnn2():
    model = Sequential()
    model.add(Convolution2D(96, 5, 5, input_shape=(3, 128, 128), activation='relu'))
    model.add(Convolution2D(96, 5, 5, activation='relu'))
    model.add(MaxPooling2D((3,3), strides=(2,2)))
    model.add(Convolution2D(256, 5, 5, activation='relu'))
    model.add(MaxPooling2D((3,3), strides=(2,2)))

    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((3,3), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(100, activation='softmax'))
    return model


if __name__ == "__main__":
    images = sys.argv[1] if len(sys.argv) >=2 else TRAIN_ROOT
    gen = ImageDataGenerator()
    data = gen.flow_from_directory(
            images, target_size=(128, 128), 
            class_mode='categorical', batch_size=16)

    model = cnn2()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    print model.summary()

    for i, (X, y) in enumerate(data):
        print model.train_on_batch(X, y)
        if i % 10000 == 9999:
            model.save('vgg_weights.h5')

