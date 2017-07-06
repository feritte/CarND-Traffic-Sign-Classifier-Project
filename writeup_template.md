**Traffic Sign Recognition** 



The goal is to classify traffic signs into one of 43 classes using deep learning.

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

[image1]: ./examples/HistogramOfTheDataSet.png "Visualization"
[image2]: ./examples/GrayScaleImages.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/5RandomImages.png "Traffic Sign 1"
[image5]: ./examples/randomTrafficSign02.png "Traffic Sign 2"
[image6]: ./examples/randomTrafficSign03.png "Traffic Sign 3"
[image7]: ./examples/randomTrafficSign04.png "Traffic Sign 4"
[image8]: ./examples/randomTrafficSign4InOne.png "Traffic Sign 5"
[image9]: ./examples/lenet.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.


You're reading it! and here is a link to my [project code](https://github.com/feritte/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]
Sample images from training data set.
![alt text][image8]
###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I implemented histogram equalization and then decided to convert the images to grayscale.

Here is an example of a traffic sign image after grayscaling.

![alt text][image2]

 I also tried to normalize the image data  however I havent got good validation accuracy with that. Therefore I have removed it for now.



####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

 

    1.  Convolution layer 1 with Input = 32x32x1. Output = 28x28x6.
    2.  RELU Activation function 1.
    3.  Pooling layer 1 of size 2x2, with Input = 28x28x6. Output = 14x14x6.
    4.  Convolution layer 2 with Output = 10x10x16.
    5.  RELU Activation function 2.
    6.  Pooling layer 2 with Input = 10x10x16. Output = 5x5x16.
    7.  Flatten Layer 1
    8.  Dropout 1
    9.  Fully Connected layer 1 with Input = 400. Output = 120.
    10. RELU Activation function 2.
    11. Droptout 2
    12. Fully Connected layer 2 with Input = 120. Output = 84.
    13. RELU Activation function 3.
    14. Fully Connected layer 3 eith Input = 84. Output = 43.
####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate. It is a lenet. 
![alt text][image9]
1. EPOCH Size: 55
2. Batch Size: 128
3. Learning Rate: 0.0008
4. Keep Prob: 0.5 (As suggested in the lessons)
To train the model, I used an Adam optimizer. 

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.953 
* test set accuracy of 0.936

If an iterative approach was chosen:
* I have adapted the lenet from the lenet lab.
* What were some problems with the initial architecture?
* I have added dropouts before every fully connected layer.
* I have choosen lower learnin rate in order to have better accuracy as it is advised in the classes.
* I need to investigate why my normalization step is not improving the performance.

If a well known architecture was chosen:
* lenet architecture was chosen.
* I have found that there are already published works that are using the lenet for this kind of problem
* Even with a limited number of EPOCH I achieved good results
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 

I obtained 100 % accuracy. Only the speed limit 40 sign I labeled it as 60 because there is no defined label for this.  
####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The top 5 softmax probabilities are given as:
* Top five:  TopKV2(values=array([[  9.85944390e-01,   7.15442793e-03,   2.10586167e-03, 1.16283540e-03,   6.79316465e-04]], dtype=float32), indices=array([[37, 40, 26, 20, 17]]))
* Top five:  TopKV2(values=array([[  9.99999762e-01,   9.45029157e-08,   6.50893668e-08, 9.58167412e-09,   9.02625175e-09]], dtype=float32), indices=array([[12, 40, 35, 38, 17]]))
* Top five:  TopKV2(values=array([[  9.99997139e-01,   1.76795936e-06,   2.72849235e-07, 1.36836675e-07,   1.35440615e-07]], dtype=float32), indices=array([[13, 15, 12, 35,  3]]))
* Top five:  TopKV2(values=array([[ 0.15786375,  0.13445772,  0.12347692,  0.05046315,  0.04843397]], dtype=float32), indices=array([[ 3, 38, 12,  9, 10]]))
* Top five:  TopKV2(values=array([[ 0.64425236,  0.12954569,  0.08598296,  0.05502813,  0.05331091]], dtype=float32), indices=array([[ 0,  1, 40,  6,  8]]))

I see that all probabilities are so high except of the speed limit 40 sign because there is no true label for that and it is assigned to the closest one. 
The accuracy on the captured images is 100% while it was 93.6% on the testing set thus It seems the model is underfitting.
