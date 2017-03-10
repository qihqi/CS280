import os
from keras.models import Sequential, load_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import cv2, numpy as np
import sys

TRAIN_ROOT = './images/train2'

def cnn2():
    model = Sequential()
    model.add(Convolution2D(36, 7, 7, input_shape=(3, 128, 128), subsample=(2, 2), border_mode='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
#
    model.add(Convolution2D(150, 2, 2, activation='relu'))
    model.add(Convolution2D(100, 2, 2, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(900, activation='relu'))
    model.add(Dense(900, activation='relu'))
    model.add(Dense(100, activation='softmax'))
    return model


if __name__ == "__main__":
    images = sys.argv[1] if len(sys.argv) >=2 else TRAIN_ROOT
    gen = ImageDataGenerator(
            samplewise_center=True, 
            rotation_range=10, 
            width_shift_range=0.1, 
            height_shift_range=0.1,
            vertical_flip=True)
    data = gen.flow_from_directory(
            images, target_size=(128, 128), 
            class_mode='categorical', batch_size=32,
            )

    if os.path.exists('train_weights_snapshotnew.h5'):
        model = load_model('train_weights_snapshotnew.h5')
        print 'loaded model'
    else:
        model = cnn2()
        model.compile(optimizer='adam', loss='categorical_crossentropy', 
            metrics=['categorical_crossentropy', 'categorical_accuracy'])
        print 'new model'
#    print model.summary()
#    print model.evaluate_generator(data, val_samples=10000)
#    print model.layers
    for i in range(8):
        print i
        model.fit_generator(data, samples_per_epoch=100000, nb_epoch=4)
        model.save('train_weights_snapshotnew.h5')

#    for i, (X, y) in enumerate(data):
#        metrics = model.train_on_batch(X, y)
#        if (i + 1) % 1000 == 0:
#            for j, m in enumerate(model.metrics_names):
#                print i, m, metrics[j]
#        if (i + 1) % 10000 == 0:
#            model.save('train_weights_snapshot_{}.h5'.format(i))


