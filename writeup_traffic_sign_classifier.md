# **Traffic Sign Recognition** 

## Writeup



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

[image1]: ./examples/Training_set.png "Distribution of training set"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./Test images/00001.jpg "Traffic Sign 1"
[image4]: ./Test images/00002.png "Traffic Sign 2"
[image5]: ./Test images/00003.png "Traffic Sign 3"
[image6]: ./Test images/00004.jpg "Traffic Sign 4"
[image7]: ./Test images/00005.jpg "Traffic Sign 5"
[image8]: ./Test images/00006.jpg "Traffic Sign 6"
[image9]: ./Test images/00007.jpg "Traffic Sign 7"
[image10]: ./Test images/00008.jpg "Traffic Sign 8"
[image11]: ./Test images/00009.jpg "Traffic Sign 9"
[image12]: ./Test images/00001.png "Traffic Sign 10"




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/GlinZhu/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the trainig data set. It is a bar chart showing how the data is distributed among the traffic sign classes, as we can see from the chart, the traffic sign of 41 (End of no passing) has the most examples in the training set while the Speed limit (120km/h) appears to be the minimum of examples.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale, since dimension reduction helps the training faster and make the training problem more tractable.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
      
As a last step, I normalized the image data to (-1,1) using (x-np.mean(x))/(np.std(x, axis=0)), beacause original image data has the range of 0-255 which is not zero mean oriented and they have different data distribution, normalization centers the mean at 0 and ensures that the input parameter has the similar data distribution which makes the convergence faster while training the network. This data normalization is done by subtracting the mean of each pixel of image, and then dividing by the standard deviation.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							|
| Pre-process data		| 32x32x1 grayscaled and normalized image		|
| Convolution 5x5x6   	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs  16x16x6 				|
| Convolution 5x5x16	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs  2x2x16  				|
| Fully connected		| 120 outputs  									|
| RELU					|												|
| Fully connected		| 84 outputs  									|
| RELU					|												|
| Fully connected		| 43 outputs  									|
| Softmax				|            									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an adam optimizer which works better in this case, and I also used the batch size of 64 and epochs of 50 with an expontially decay learning rate of 0.0009.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98.8% at epoch 50
* test set accuracy of 91%

I initially sued Lenet-5 model which was implemented in the class for detecting hand-written digit, and this model works perfet for my initial attempt. However, it only has accuracy of 89% while I was trying it on the validation set, and the reason causing this issue might be due to the overfitting of the training data, therefore, I used learning rate decay and L2 regularization method to avoid the overfitting issue. 
The L2 regularization implemented was for fully connected layer with penalizing all four parameters (weights,biases, etc.) and it turns out that the model shows higher accuracy on the validation set which reaches to about 91% when I implemented both learning rate decay and L2 regularization

During the model training, I tuned all hyperparameters, for example, decreasing both learning rate and batch size will significantly improve the validation accuracy, that is, I will lose some accuracy from having the bigger batch-size or learning rate, however, too small learning rate or batch size will significantly increase the trainig time.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11] ![alt text][image12]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The final accuracy of detecting those new images is 60% which mean we detected 6 traffic signs correctly. 

Here are the results of the prediction:

| Image			                 |     Prediction	        					| 
|:------------------------------:|:--------------------------------------------:| 
| 18: General Caution	         | General Caution   							| 
| 38: Keep right                 | Keep right 									|
| 34: Turn left ahead	         | Turn left ahead								|
| 19: Dangerous cure to the left | Dangerous cure to the left	 				|
| 25: Road work	                 | Speed limit(30kph)      						|
| 36: Go straight or right		 | End of no passing      						|
| 0: Speed limit(20kph)		     | Speed limit(20kph)      						|
| 40: Roundabout mandatory		 | End of no passing by vehicel over 3.5    	|
| 3: speed limit(60kph)		     | Speed limit (50km/h)    						|
| 1: Speed limit(30kph)		     | Slippery Road      							|

The model was able to correctly guess 6 of the 10 traffic signs, which gives an accuracy of 60%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 71th and 72th cell of the Ipython notebook.
The challenge of detecting those new images is that there are too much background in the images, 

Here are the 5 results of the image prediction:

Top 3 labels for image General caution:
* General caution with prob =100%
* Traffic signals with prob =2.2e-20%
* Pedestrians with prob =2.19e-23%

Top 3 labels for image Keep right:
* keep right with prob =99.5%
* Dangerous curve to the right with prob =0.49%
* Go straight or right with prob =0.0046%

Top 3 labels for image Turn left ahead:
* Turn left ahead with prob =100%
* Keep right with prob =1.88e-18%
* Go straight or right with prob =1.67e-19%

Top 3 labels for image Dangerous cure to the left:
* Dangerous cure to the left with prob =99.99%
* Road work with prob =0.0008%
* Slippery road with prob =1.06e-12%

Top 3 labels for image Road work:
* Speed limit(30kph) with prob =99.96%
* Right-of-way at the next intersection with prob =0.038%
* Speed limit (20kph) with prob =0.00094%

The model was able to detect 6 out of 10 images which give a relatively high accuracy, the four images detected incorrectly all have too much background or the images are transformed so increasing the number of training set by using data augmentation method is able to expose the neural network model to a wide varienty of variations. 
