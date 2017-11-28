import csv
import cv2
import numpy as np
from PIL import Image
lines = []
car_images = []
steering_angles = []
def process_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
with open('./train_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        lines.append(row)



"""
for lines in lines:
    for i in range(3):
        source_path = lines[0]
        filename = source_path.split('/')[-1]
        current_path = './train_data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(lines[3])
        correction =  0.2
        measurements.append(measurement + correction)
        measurements.append(measurement - correction)           
        measurements.append(measurement)

"""
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
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers import MaxPooling2D



model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

# leNet architecture

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
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
"""
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split= 0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')
