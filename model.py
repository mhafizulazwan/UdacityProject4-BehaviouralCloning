import glob
import csv
from scipy import ndimage
import numpy as np

#See download_script file to download the data
#Data 1 – Two laps of center lane driving
#Data 2 – One lap driving counter-clockwise
#Data 3 – One lap of recovery from the sides
#Data 4 – One lap focusing on driving smoothly around curves

#open training data and store them in a list
lines = []
for filepath in sorted(glob.glob('/opt/Data/Data*/driving_log.csv')):
    with open(filepath) as csvfile:
        reader = csv.reader(csvfile)
        index = 0
        for line in reader:
            if index == 0: #remove first row (because it's just label)
                index = index + 1 
            else:
                lines.append(line)

#store images and measurements 
images = []
measurements = []
tobe_augmented_images = []
tobe_augmented_measurements = []
for line in lines:
    for i in range(3):
        source_path = line[i] #path of the centre, left, and right images
        filename1 = source_path.split('/')[-3]
        filename2 = source_path.split('/')[-2] 
        filename3 = source_path.split('/')[-1] 
        current_path = '/opt/Data/'
        image = ndimage.imread(current_path+filename1+'/'+filename2+'/'+filename3)
        images.append(image)
        measurement = float(line[3]) #take steering measurement
        measurements.append(measurement)
        if filename1 == 'Data': #perform augmentation only on the first data i.e., Data
            tobe_augmented_images.append(image)
            tobe_augmented_measurements.append(measurement)

#flip images and measurements (data augmentation purpose)
for image, measurement in zip(tobe_augmented_images, tobe_augmented_measurements):
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    images.append(image_flipped)
    measurements.append(measurement_flipped)

#store in standard array format
x_train = np.array(images)    
y_train = np.array(measurements)

#import keras libraries
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
import matplotlib.pyplot as plt

#build model and layer (utilized network from Autonomous Vehicle team in NVIDIA)
model = Sequential()
model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((66,23),(0,0))))
model.add(Conv2D(24,(5,5),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(36,(5,5),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(48,(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#compile and train
model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit(x_train,y_train, validation_split = 0.2, shuffle=True, epochs = 3, verbose = 1)

#plot training and validation loss for each epochs
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss_plot.png')
#plt.show()

#save the model
model.save('model.h5')