# Table of Content
* [Introduction](#introduction)
* [Project Proposal](#project-proposal)
   * [Abstract](#abstract)
   * [System Overview](#system-overview)
   * [Development Environment](#development-environment)
   * [Responsibility of Team members](#responsibility-of-team-members)
   * [Future Goals](#future-goals)
   
* [How to set up the development environment](#how-to-set-up-the-development-environment)
* [References](#references)


## Introduction

This project is for our [CSE 4340 - Fundamentals of Wireless class](http://wsslab.org/vpnguyen/teaching.html) taught by [Dr. VP Nguyen](http://wsslab.org/vpnguyen/)
semester project at [The University of Texas at Arlington](https://www.uta.edu). We will be following some standard documentation procedure along the way. If you have any questions, feel free to communicate with any of the contributors.

## Project Proposal

Here's how our porject proposal looks like. You can even find a pdf version of it [here](https://github.com/nisargushah/face-mask-detection/blob/main/Project%20Proposal.pdf)

### Abstract

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;With the current pandemic of Covid-19, many governing authorities and/or corporate
institutions have made the wearing of face masks mandatory on their premises. Face masks
can help in reducing the spread of the novel virus, from something as normal as a person
speaking to a person coughing. However, some people have refused to adhere to mask
policies in stores and other public places or simply forget to carry one with them from time to
time. That is why, we have come up with a project idea to develop a real-time system which
detects if the person is wearing a face mask or not. This can help automate the detection of
face masks in stores like Walmart, Target who are advocating the policy of mandatory face
covering to enter their premises. For the purpose of this project, we will be using the camera
sensor in web-cameras coupled with machine learning algorithms to detect the Face Masks in
real time. This algorithm can then be extended to be used in CCTV cameras or mobile
cameras, for real time face mask detection in the institutions previously mentioned.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We plan to display live statistics on the face of the person being tested. A red frame
around the face will be displayed if the person is not wearing a mask, and a green frame will
be displayed if the mask is detected. The frames will be supplemented by their accuracy
percentages, right next to it, which will give us an indicator of how confident our app is of
detecting the face masks. This will serve as the basic functionality for our application

### System Overview

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;For this project, we will try to utilize the webcam that is already present in our phones
or computers and using various python libraries, we will train neural network to detect if a
person is wearing a face mask or not or if they are wearing it wrong. We will use Kaggle
dataset as our training data to train our Convolution Neural Network. We will use this dataset 
and use tensorflow’s advanced object detection feature to isolate face images and then gather
all such images and then train our network on them.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The following diagram demonstrates the workflow and basic components that will be
used in the project. As mentioned previously, the algorithm will be a two phase process, first,
having to train the model to detect face masks using the dataset, and then applying this
training to analyse further new images - to load the detector, to detect the face, and then to
detect the mask.




### Development Environment

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We will be using python3 and the IDE of our choice will be PyCharm. We will be
demonstrating our project with the camera sensor embedded in webcams in our laptops, so
that each member has access to it for development. The dataset that we will be using is
available freely to be used for open-source on Kaggle Dataset to get training data for our
model.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;We will also be using various open-source libraries that are available. Here’s a list of possible
libraries that we may use.

➢ Keras and Tensorflow for training our model.

➢ NumPy for converting our images into arrays of pixel density and it is also a requirement
for opencv2

➢ OpenCV2 to help us utilize the webcam and capture videos and images for us to
implement our pre-train model into a live scenario.

➢ Imutils and tkinter for some useful functions and GUI effects, if needed.

### Responsibility of Team members

We have decided that for the purpose of this project it is necessary to divide worklaod from the begining. Heres' and overview of that ... 

[Nisarg Shah](https://nisargushah.com) - Development lead on both the phases, and administrative head (maintains the GitHub repo,
YouTube link, document submissions, and collaboration among group members)

[Hemantha Govindu](www.uta.edu) - Primary developer for Phase 1 (Training the Face Mask Detector), in charge of project
progress report

[Nihar Gupte](https://www.linkedin.com/in/nihargupte/) - Primary developer for Phase 2 (Applying the Face Mask Detector), in charge of final project
report

### Future Goals

If time permits, we would also like to implement a social distancing detector
using some of the same libraries.


## How to set up the development environment

For setting up the development environment,wewill be using the Anaconda toolkit which can be download from [here](https://www.anaconda.com/products/individual) for free. After downloading the environment, we can go to the desired folder where we want to copy the repoitory. After that open Ananconda bash shell for Windows user and terminal for Mac/Linux user and write the following:


```concole

foo@bar~$ conda env create -f environment.yml
```

This should create all the neccessary environment, after that we can clone the repository there and continue with the steps.


## Please note that this is still an under-development project, so things will keep on changing but we will try to keep updating it. Thanks!



### References

