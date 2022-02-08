# Developing an AI Application
Going forward, AI algorithms will be incorporated into several applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, an image classifier is trained to recognize different species of flowers. The idea in this project can be deployed as a phone app that can use the phone's camera to identify the specie of a flower. In practice you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories, you can see a few examples below.

![image](https://user-images.githubusercontent.com/85171486/152971138-c82eab9f-f3a0-492d-986e-96727587e620.png)

The project is broken down into multiple steps:

Load and preprocess the image dataset
Train the image classifier on your dataset
Use the trained classifier to predict image content
We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
