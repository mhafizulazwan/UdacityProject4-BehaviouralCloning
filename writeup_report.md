# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_lane_driving.jpg "Center lane driving"
[image2]: ./examples/counter_clockwise.jpg "Conter-clockwise driving"
[image3]: ./examples/from_right.jpg "Going to the right edge"
[image4]: ./examples/from_right_center.jpg "Recovering"
[image5]: ./examples/to_center.jpg "Being in the center"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 65-72) 

The model includes RELU layers to introduce nonlinearity (code line 65-72), and the data is normalized in the model using a Keras lambda layer (code line 63). 

#### 2. Attempts to reduce overfitting in the model

The model contains MaxPooling layers in order to reduce overfitting (model.py lines 66-72). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 14-22). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 81).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination two laps of center lane driving, one lap couter-clockwise driving, one lap recovering from the left and right sides of the road, and one lap focusing on driving smoothly around curves. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use enough data, use appropriate architecture, preprocess the data, and playing around with the number of epochs.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it can predict images pretty well.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that the convolution and fully connected layers are increased.

Then I normalized and cropped the images.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially at the curves. To improve the driving behavior in these cases, I collect one lap of data focusing on the curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 62-78) consisted of a convolution neural network with the following layers and layer sizes:
1. Lambda layer.
2. Cropping layer.
3. Four convolution layers with 24, 36, 48, and 64 depths.
4. Four fully connected layers.
5. Five fully connected layers with 1164, 100, 50, 10, and 1 units.

The above architecture is based on the architecture from Autonomous Vehicles team's from NVIDIA.

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

After that, I collected one lap of counter-clockwise driving as shown in the image below:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to the center. These images show what a recovery looks like starting from right to the center:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data set, I also flipped images and angles thinking that this would combat overfitting.

After the collection process, I had X number of data points. I then preprocessed this data by normalization and cropping.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the training and validation losses. I used an adam optimizer so that manually training the learning rate wasn't necessary.
