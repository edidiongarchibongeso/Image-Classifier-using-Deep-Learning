#!/usr/bin/env python
# coding: utf-8

# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Imports here
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import json
import time
from PIL import Image
import random
from pathlib import Path
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from collections import OrderedDict

get_ipython().run_line_magic('matplotlib', 'inline')

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[3]:


# Define your transforms for the training, validation, and testing sets
""" The torchvision.transforms library is used to transform the dataset. 
The .ToTensor function converts the images into numbers. The transform.Normalize 
function subtracts the mean from each value and then divides it by the standard deviation"""

mean = [0.485, 0.456, 0.406] # Three values to match each RGB picture
std = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([transforms.RandomRotation(30), # Randomly rotate images
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(), transforms.Normalize((mean), (std))])

valid_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                      transforms.ToTensor(), transforms.Normalize((mean), (std))])

test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                      transforms.ToTensor(), transforms.Normalize((mean), (std))])

# TODO: Load the datasets with ImageFolder
training_set = datasets.ImageFolder(train_dir, transform=train_transform)
validation_set = datasets.ImageFolder(valid_dir, transform=valid_transform)
test_set = datasets.ImageFolder(test_dir, transform=test_transform)

# TODO: Using the image datasets and the trainforms, define the dataloaders
training_loader = DataLoader(training_set, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

print("Training set size: {}".format(len(training_set)))
print("Validation set size: {}".format(len(validation_set)))
print("Test set size: {}".format(len(test_set)))


# In[4]:


image_classes = training_set.classes
print(image_classes)


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[5]:


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print(len(cat_to_name))


# In[6]:


# Shows a batch of images after transform has been applied
def show_images(image):
    if isinstance(image, torch.Tensor):
        image = image.numpy().transpose((1, 2, 0))
    else:
        image = np.array(image).transpose((1, 2, 0))
    
    # Remove the normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    # Plot images
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    plt.imshow(image)
    ax.axis('off')

images, _ = next(iter(training_loader))
out = torchvision.utils.make_grid(images, nrow=8)
show_images(out)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# **Note for Workspace users:** If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[7]:


# TODO: Build and train your network

# Load a pre-trained network using torchvision.models as models library
model = models.vgg16(pretrained=True)

# Freezing parameters to prevent backpropagation through them
for param in model.parameters():
    param.requires_grad = False
    
model


# A new classifier is developed to replace the pre-trained classifier. To parameters that should be inputted into the neural network should be the same as the model's default classifier. The number of output should match the number of images in the dataset. 
# 
# nn.Sequential helps to group multiple modules together.
# 
# nn.Linear specifies the interaction between layers.
# 
# nn.ReLU is the activation function for the hidden layers. 
# 
# nn.LogSoftmax is the activation function for the output layer. To specify that the output layer is a column, dimension is set to 1

# In[8]:


def classifier(model):
    """ This defines a new, untrained feed-forward network as a classifier, using ReLU activations and dropout"""
        
    input_size = 25088
    hidden_size = [512, 256]
    num_labels = len(cat_to_name)
    

    classifier = nn.Sequential(nn.Linear(input_size, hidden_size[0]),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(hidden_size[0], hidden_size[1]),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(hidden_size[1], num_labels),
                           nn.LogSoftmax(dim=1))
    
    return classifier


# In[9]:


# Replace the default classifier with the new classifier
model.classifier = classifier(model)
model


# In[10]:


# Learning rate
learning_rate = 0.001

# Set up the loss function to evaluate the error in the model
criterion = nn.NLLLoss()

# Update the weights and biases to reduce the error
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

optimizer


# In[11]:


# Find the device available 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the model to the available device specified above
model.to(device);


# In[12]:


epochs = 10
counter = 0
running_loss = 0
print_every = 40
training_losses, validation_losses = [], []

for epoch in range(epochs):
    # Set model to train mode
    model.train()
    for data in training_loader:
        images, labels = data
        
       # Print the progress of the training
        counter += 1
        
       # Move images and labels to device
        images, labels = images.to(device), labels.to(device)
        
       # Set the gradients of all optimized torch to zero 
        optimizer.zero_grad()
        
       # Use the forward function to get output from the input
        output = model.forward(images)
       
       # Use the output to compute the loss
        loss = criterion(output, labels)
        
       # Backward pass to calculate the gradients
        loss.backward()
       
       # Iterate over all parameters and update the weights
        optimizer.step()
     
       # Add the loss to the training sets running loss
        running_loss += loss.item()
        
       # Check the models performance on the validation set
        if counter % print_every == 0:
            
        # Set the model to evaluation mode 
            model.eval()
            validation_loss = 0
            accuracy = 0
              
            # Disable gradient calculation
            with torch.no_grad():
                for data in validation_loader:
                    images, labels = data
                    
                    # Move to device
                    images, labels = images.to(device), labels.to(device)
                
                    # Pass in images from the validation set to get the logps
                    output = model(images)
                
                    # Calculate the loss
                    batch_loss = criterion(output, labels)
                
                    # Add the batch_loss to the validation set running
                    # to keep track of the validation loss
                    validation_loss += batch_loss.item()
                    
                    # Calculate the accuracy
                    # Get the probabilities
                    proba = torch.exp(output) # Gives the first largest value in our probabilities
                
                    # Check for equality with the labels
                    equality = (labels.data == proba.max(dim=1)[1])
                
                    # With the equality tensor update the accuracy
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
            
            training_losses.append(running_loss/len(training_loader))
            validation_losses.append(validation_loss/len(validation_loader))
            
            # Print the training and evaluation information
            print("Epoch: {}/{} ".format(epoch+1, epochs),
                 "Training Loss: {:.3f} ".format(running_loss/print_every), # Takes the average of the training loss
                 "Validation Loss: {:.3f} ".format(validation_loss/len(validation_loader)), # Outputs loss for each batch
                 "Validation Accuracy: {:.3f} ".format(accuracy/len(validation_loader)))
            
            # Set running loss back to zero
            running_loss = 0
        
            # After evaluation, set model to training mode      
            model.train()


# In[13]:


plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.legend()
plt.show()


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[14]:


def test_model(model, test_loader):
    test_accuracy = 0
    test_loss = 0
    
    # Set to evaluation mode
    model.eval()
    model.to(device)
    
    for data in test_loader:
        images, labels = data
                    
        # Move to device
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            # Complete a forward pass of the test data
            output = model.forward(images)
                    
            # Return a new tensor with the exponential of the elements of the input tensor
            proba = torch.exp(output)
                
            # See how many classes were correct
            equality = (labels.data == proba.max(dim=1)[1])
                
            # Calculate the batch mean and input it to the running accuracy for this epoch
            test_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
    
    print("The test accuracy of the model is {:.2f}%".format(test_accuracy/len(test_loader) * 100))
    


# In[15]:


test_model(model, test_loader)


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[16]:


# Save the checkpoint 
model.class_to_idx = training_set.class_to_idx

checkpoint = {'epochs': epochs,  
                 'classifier': model.classifier,
                 'class_to_idx': model.class_to_idx,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}

# Saving checkpoint
torch.save(checkpoint, 'checkpoint.pth')


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[17]:


# TODO: Write a 
def load_checkpoint(filepath='checkpoint.pth'):
    
    """ This function loads a checkpoint and rebuilds the model"""
    
    checkpoint = torch.load(filepath)
    
    model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad=False
        
    model.classifier = checkpoint.get('classifier')
    model.class_to_idx = checkpoint.get('class_to_idx')
    model.load_state_dict(checkpoint.get('state_dict'))
    optimizer = checkpoint.get('optimizer')
    epochs = checkpoint.get('epochs')
    
        
    # Find available device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move to available device
    model.to(device)
    
    return model

model = load_checkpoint('checkpoint.pth')
model


# In[18]:


# Test model's accuracy after loading checkpoint
test_model(model, test_loader)


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[19]:


image_path = random.choice(list(Path(f"{test_dir}/").glob('**/*.jpg')))
print(image_path)
image = Image.open(image_path)
image


# In[20]:


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    
    _mean = np.array([0.485,0.456,0.406])
    _std = np.array([0.229,0.224,0.225])
    image_size = 256
    cropped_size = 224
    
    # Define transforms for resize and center crop
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(cropped_size),
        transforms.ToTensor()])
    
    # Process and load PIL Image
    image_pil = Image.open(image_path)
    
    # Apply transform to resize and crop image
    image_pil = transform(image_pil)
    
    # Convert image into numpy array
    np_image = np.array(image_pil)
    
    # Normalize based on the preset mean and standard deviation
    image = (np.transpose(np_image,(1,2,0)) - _mean)/_std
    
    # Transpose from third dimension to first dimension of color channel
    image = np.transpose(image, (2,0,1))
    
    return image


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[21]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# In[22]:


plt.show(imshow(process_image('flowers/test/46/image_00969.jpg'),None,None))


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[23]:


def predict(image_path, model, topk=5):
    ''' This function predicts the class (or classes) of an image using a trained deep learning model.
    '''
    #processing the image
    image = process_image(image_path)
    
    #converting the numpy image to a tensor
    img = torch.from_numpy(image)
    img = img.float().unsqueeze_(0).to(device)
    
    print(type(model))
    
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient calculation
    with torch.no_grad():
        
        model = model.to(device)
        
        ps=torch.exp(model(img))
        top_prob,top_class = ps.topk(topk, dim=1)
    
    return top_prob, top_class

predict('flowers/test/46/image_00969.jpg', model)


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[26]:


# TODO: Display an image along with the top 5 classes

# Select an image in the test folder for prediction
test_image ='./flowers/test/46/image_00969.jpg'

# Use cat_to_name to convert from class integer to actual flower class names
with open('cat_to_name.json', 'r') as f:
    mapper = json.load(f)

# Isolate the test name and pass to the class labels to identify the actual flower class name
class_name = mapper[test_image.split('/')[-2]]

# Process the test image
image = process_image(test_image)

# Load model from checkpoint
model = load_checkpoint('checkpoint.pth')

# Get the top probabilities and top class using the predict function
top_prob, top_class = predict(test_image, model)

# Create a subplot
figure, (ax1,ax2) = plt.subplots(2,1)
imshow(image, ax1)
ax1.axis("off")

# Create a data frame that combines the class with the flower class name
df = pd.DataFrame({'class': pd.Series(model.class_to_idx), 'class_name':pd.Series(cat_to_name)})

# Set the class index
df = df.set_index('class')

# display the top probabilities and class
df = df.iloc[top_class[0]]
df['probability'] = top_prob[0]

# Create a new subplot
plt.subplot(2,1,2)

# Set a color palette
with sns.color_palette("coolwarm"):
    sns.barplot(x=df['probability'], y=df['class_name'])
    
plt.show()

