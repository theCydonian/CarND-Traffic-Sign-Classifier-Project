# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/right33.jpg "Traffic Sign 1"
[image5]: ./examples/road25.jpg "Traffic Sign 2"
[image6]: ./examples/row11.jpg "Traffic Sign 3"
[image7]: ./examples/speed4.jpg "Traffic Sign 4"
[image8]: ./examples/yield13.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it!

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a line graph showing how the validation acuracy improves over each epoch.

[graph]: ./results.png "Training Results"

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

For preprocessing, I decided to normalize the data so that all values lie between -1.00 and 1.00. This ensures the data has a mean of zero and should allow the network to converge faster.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_6 (Conv2D)            (None, 30, 30, 32)        896       
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 15, 15, 32)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 13, 13, 64)        18496     
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 6, 6, 64)          0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 4, 4, 64)          36928     
_________________________________________________________________
flatten_2 (Flatten)          (None, 1024)              0         
_________________________________________________________________
dense_4 (Dense)              (None, 64)                65600     
_________________________________________________________________
dense_5 (Dense)              (None, 43)                2795      
=================================================================
Total params: 124,715
Trainable params: 124,715
Non-trainable params: 0
_________________________________________________________________
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used ten epochs as the network was quickly converging. I also used sparse catagorical crossentropy for my loss function for greater efficiency.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.39%
* validation set accuracy of 97.01%
* test set accuracy of 94.56%

I started with a basic small CNN so that I could test more quickly. The initial model overfit a bit. I adjusted to compensate for overfitting by adding dropout layers. I then tuned the dropout percent.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

[image8]: ./examples/yield13.jpg "Traffic Sign 1"
[image7]: ./examples/speed4.jpg "Traffic Sign 2"
[image6]: ./examples/row11.jpg "Traffic Sign 3"
[image5]: ./examples/road25.jpg "Traffic Sign 4"
[image4]: ./examples/right33.jpg "Traffic Sign 5"


The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 70 km/h     			| 70 km/h 										|
| Right of Way  		| Right of Way									|
| Yield         		| Yield      									| 
| Roadwork	      		| Roadwork          			 				|
| Right Turn   			| Right Turn        							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.56%, but, as the test set contains more images, it is a more relevant metric.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a speed limit of 70 km/h sign, and the image does contain a stop sign. The top four soft max predictions were 70 kh/h(100%), 20 kh/h(0%), 30 kh/h(0%), and 80 kh/h(0%).
For the second image, the top four soft max predictions were Right of Way(100%), Ice(0%), Priority Road(0%), and Animal Crossing(0%).
For the third image, the top four soft max predictions were yield(100%), Priority Road(0%), Speed Limit 120 km/h(0%), and no entry(0%).
For the fourth image, the top four soft max predictions were road work(99.77%), ice(0.23%), bicycles(0%), and dangerous right turn(0%).
For the fifth image, the top four soft max predictions were right turn(90.00%), ahead only(9.94%), keep left(0.06%), and mandatory roundabout(0%). 


