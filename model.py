import csv
import cv2
import numpy as np
from PIL import Image



image_path = './train_data/IMG/'
correction = 0.3
lines = []

with open('./train_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
          
    
images = []
measurements = []

for line in lines:
    # center image
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = image_path + filename
    image = cv2.imread(current_path)
    images.append(image)

    measurement = float(line[3])
    measurements.append(measurement)

    # left image
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = image_path + filename
    left = cv2.imread(current_path)
    images.append(left)

    measurement = float(line[3])
    measurements.append(measurement + correction)

    # right image
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = image_path + filename
    right = cv2.imread(current_path)
    images.append(right)

    measurement = float(line[3])
    measurements.append(measurement - correction)

    
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement*-1.0)



X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


# preprocessing the data
# includes normalizing

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D



model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0)) ))
# leNet architecture
"""
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

"""
# nVidia architecture
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split= 0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')
