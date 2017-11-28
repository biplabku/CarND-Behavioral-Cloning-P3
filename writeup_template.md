# **Behavioral Cloning** 

## Writeup Template
Behavioral Cloning for Self-Driving Cars

The goal of this project is to train a deep neural network to clone driving behavior by using images to predict steering angles. Once the model is trained, it will be used to drive a car autonomously around a test track in Udacity's driving simulator.

![Alt text](/examples/nVidia_model.png?raw=true "Graphic model")

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


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

I tried initially with a simple LeNet Architecture. It involves the following layers.
1. Convolutional layer.
2. 6 filter of size 5 x 5 size 
3. Relu Activation function
4. Maxpooling layers - 2
5. Fully connected layers - 2

Another architecture
1. Convolutional layers - 5
2. Filters - 24, 36,48, 64, 64.  size - 5 x 5
3. Activation function - relu
4. Dropout layers - 3
5. fully connected layers - 4



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. Other ways used during training are as follows - 
1. Experimented with different Epochs.
2. Trained the model by driving the car near the edges and driving a reverse lap
3. Using fewer convulational layers

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the roa. 

For details about how I created the training data, see the next section. 
1. Driving the vehicle in the center of the road 
2. Driving the vehicle near the edges
3. Travelling in counter clockwise direction
4. Collecting the data from the second track
5. Taking couple of laps

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive the car through the lap without getting out of the road. My first try was to use LeNet architecture as i used it during our last project but the results were not satisfactory. The car went straight to the lake after following the road for few steps. I tried to change the Epochs and increase convulational layers but it still was not able to follow properly.

The second step was to use a more powerfull model: nVidia Autonomous Car Group The only modification was to add a new layer at the end to have a single output as it was required. This time the car did its first complete track without any fail apart from drifting in some areas. Since the training was not completly done on the seocnd track i collected a full lap of training data for the second track and finally the car was able to drive properly without any fault.

The final step was to run the simulator to see how well the car was driving around track one.At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the left turn and right side repsectively.

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help in better training of the nodel to overcome such situations. 

After the collection process, I had X number of data points. I then preprocessed this data by Cropping and normalizing the data. 


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by results. I used an adam optimizer so that manually training the learning rate wasn't necessary.


## Results 

The above strategies resulted in a model that performed well on both test and challenge tracks. The training and validation sets are extracted in the ratio of 80% to 20%. The validation data for the model is pulled from the training set.

## Conclusion and Discussion

This project along with all the other assignments provided me with an opportunity to play with the data's and looking at how the training improved the model.

The constant imoprovement with the different layers and experimenting with various parameters not only improved my understanding regarding the functionalities but provided me an improved insight into the self driving world.

Training the car to drive itself, with relatively little or no effort was extremely rewarding.
