#**Traffic Sign Recognition** 

#ReadMe
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

[image1]: new_signs.png "New Signs"
[image10]: bar_chart.png "Bar Chart"
[image11]: preprocessing.png "Preprocessing"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Data summary

The code for this step is contained in the third code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Data visualization

The code for this step is contained in the fourth and fifth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is categorized.

![alt text][image10]

**Example:**  
Based on the chart, about 2000 images in the training set are cetgorized as traffic sign #2

###Design and Test a Model Architecture

####1. Data preprocessing

The code for this step is contained in the sixth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because a grayscale image has 1 channel for a network to traverse opposed to 3 channels, making it more efficient.

Next, I applied a Gaussian Blur to the image to smooth out major edges and overlayed this blurred image on top of the original grayscale image to sharpen the image and increase detail.

Lastly, I ran the image through two histogram equalizers for normalization. The first normalizer is from the OpenCV library and the second is imported from sklearn.exposure

It is also important to note that the very last preprocessing step done is shuffling the data. This is crucial for improving the validity of the model.

Here is an example of a traffic sign image before and after these preprocessing steps.

![alt text][image11]

####2. Data splitting

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook.  

The number of images in each set are:  

* Training set: 34799 images 
* Test set: 12630 images 
* Validation set: 4410 images

For the time being, I decided not to augment the data. This is a step that will be taken in the future, however my goal here is to acheive a high accuracy without the need to augment and generate new data.


####3. Network Architecture

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers with a mu of 0 and sigma of 0.1:

| Layer         		|     Description	      | 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image | 
| Convolution 5x5   | 1x1 stride, valid padding, outputs 28x28x6     |
| RELU					|						      |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 |
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16 |
| RELU			    	|        		      	      |
| Max pooling    	| 2x2 stride, outputs 5x5x16 |
| Flatten           | outputs 400			      |
| Fully Connected	| outputs 120			      |
| RELU              |                         |
| Dropout           |                         |
| Fully Connected   | outputs 84              |
| RELU					| 						      |
| Dropout				|					 	      |
| Fully Connected   | outputs 10              | 


####4. Model training

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an Adam Optimizer and prefaced that with L2 loss regularization with a beta of 0.001.

The model was also built with 60 epochs, a batch size of 128, and a learning rate of 0.0008

The epochs, batch size, and beta are defined in the seventh cell of the ipython notebook.

After numerous tests, I landed upon an epoch count of 60 because this led to the most satisfying validation accuracy while still computing in an acceptable time and not requiring architectural changes to the network.

I chose a learning rate of 0.0008 because any higher and the accuracy tended to bounce around a bit and any lower would result in a slow increase in accuracy.

####5. Final Solution

The code for calculating the accuracy of the model is located in the eighth cell for training and vaildation and the ninth cell for testing.

My final model results were:   
 
* training set accuracy of 99.0
* validation set accuracy of 95.3
* test set accuracy of 93.1


If an iterative approach was chosen:  

* **Architecture -** The architecture started as an exact copy of the LeNet architecture from Yann Lecun's 1998 paper on Gradient-Based learning
* **Improvements -** While the initial architecture actually worked quite well once the data was pre-processed, it was missing some features to increase robustsness and alleviate overfitting.
* **Modifications -** To address these issues, I made two major adjustments. I added two dropout layers to the fully connected layer with a probability of keeping an activation (keep_prob) at 0.5 for training. Secondly, I added L2 regularization to the architecture with a beta value of 0.001. This value seemed to provide solid results for my model.
* **Hyperparameters -** I tuned the learning rate most heavily to decrease likelihood of bouncing accuracy rate. I also started keep_prob at 0.7 but found this to be too high as too many activations were being accepted and the model was still overfitting so I decreased it to 0.5.
* **Design -** The two most important design choices for my method were keeping the convolutional layers because of their excellent ability to scan the data and learn important features as well as adding a dropout layer. Dropout was essential in the architecture in order to alleviate overfitting the training set.
 

###Test a Model on New Images

####1. New signs.

Here are nine German traffic signs that I found on the web:

![alt text][image1]

Each of these images are generally clearer than those in the original data set. This may result in issues in preiciton since the model is not accustomed to as clear of images.

####2. New sign predictions.

The code for making predictions on my final model is located in the twelfth and thirteenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction 1	        					| Prediction 2 | Prediction 3 |
|:---------------------:|:---------------------------------------------:| :---------------------------------------------:| :---------------------------------------------:|
| 120km/h | 30km/h (82%)| 20km/h (17%)| 120km/h (0%)| 
| 50km/h | 50km/h (97%)| 80km/h (1%)| 30km/h (1%) |
| Keep right| Keep right (100%)| End all speed and passing limits (0%)| Turn left ahead (0%)| 
| No entry | No entry (100%) | Stop (0%) | End of no passing (0%)|
| No passing| No passing (99%)| No passing for large vehicles (1%) | Ahead only (0%)|
|RoW at next intersection|RoW at next intersection(100%)|Beware of ice/snow (0%)|Pedestrians(0%)|
|Road work| Road work (62%)|Children crossing (11%)|Dangerous curve to the right (10%)|
|Priority Road| Priority Road (100%)|Roundabout mandatory (0%)| End of no passing (0%)|
|Yield| Yield(100%)|Ahead only (0%)| Priority road(0%)|


The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 88.9%. This compares favorably to the accuracy on the test set of 93.1%

####3. Five softmax probabilites

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

**First Image:**  

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .82         			| 30 km/h | 
| .17     				| 20 km/h |
| 0.0						| 120 km/h|
| 0.0      			    | 70km/h	|
| 0.0			          | 80km/h  |


**Second Image:**    

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .97        			| 50 km/h | 
| .01    				| 80 km/h |
| .01					| 30 km/h|
| 0.0   			    | 60km/h	|
| 0.0			       | 80km/h  |

**Third Image:**    

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| Keep right | 
| 0.0    				| End of all speed and passing limits |
| 0.0					| Turn left ahead|
| 0.0   			    | Yield	|
| 0.0			       | General Caution  |

**Fourth Image:**    

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0        			| No entry | 
| 0.0    				| Stop |
| 0.0					| End of no passing|
| 0.0   			    | Roundabout mandatory|
| 0.0			       | Turn right ahead|

**Fifth Image:**    

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99        			| No passing | 
| 0.01    			| No passing for large vehicles |
| 0.0					| Ahead only|
| 0.0   			    | Large vehicles prohibited|
| 0.0			       | No vehicles|

**Sixth Image:**    

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0       			| RoW at next intersection | 
| 0.0    			| Beware ice/snow |
| 0.0					| Pedestrians|
| 0.0   			    | General Caution|
| 0.0			       | Roundabout mandatory|

**Seventh Image:**    

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .62       			| Road work | 
| .11   			| Children crossing |
| .10					| Dangerous curve to the right|
| 0.06  			    | Beware ice/snow|
| 0.04			       | Bicycles crossing|

**Eighth Image:**    

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0       			| Priority Road | 
| 0.0    			| Roundabout mandatory |
| 0.0					| End of no passing|
| 0.0   			    | Stop|
| 0.0			       | Yield|

**Ninth Image:**    

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0       			| Yield | 
| 0.0    			| Ahead only |
| 0.0					| Priority Road|
| 0.0   			    | Straight or right |
| 0.0			       | 60km/h |