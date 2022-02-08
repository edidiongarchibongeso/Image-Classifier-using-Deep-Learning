import argparse
import time
import torch
import torch.nn.functional as F

from collections import OrderedDict
from torch import nn, optim
from torchvision import datasets, models, transforms



def parse_args():
    parser = argparse.ArgumentParser(description='Train the model')

    parser.add_argument('--data_dir', type=str, help='Input a path to dataset', default='flowers')
    parser.add_argument('--save_dir', type=str, help='Save a checkpoint for the trained model', default="checkpoint.pth")
    parser.add_argument('--arch', type=str, help='Input the models architecture', default='vgg16', choices=['vgg16', 'densenet121'])
    parser.add_argument('--learning_rate', type=float, help='Input learning rate', default=0.001)
    parser.add_argument('--hidden_units', type=int, help='Input number of hidden units', default=512)
    parser.add_argument('--epochs', type=int, help='Input number of epochs', default=1)
    parser.add_argument('--device', type=str, help='Make use of GPU if available (True|False)', default='True')

    return parser.parse_args()

def load_model(model, h_layers, choice_of_model, dropout=0.2):
    """ This defines a new, untrained feed-forward network as a classifier, using ReLU activations and dropout"""

  
    # Freezing parameters to prevent backpropagation through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Select model choice
    if choice_of_model == 'densenet121':
        input_size = model.classifier.in_features

        classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(input_size, h_layers)),
                        ('relu1', nn.ReLU()),
                        ('dropout1', nn.Dropout(p=dropout)),
                        ('fc2', nn.Linear(h_layers, 102)),
                        ('output', nn.LogSoftmax(dim=1))]))
    else:
        input_size_vgg = model.classifier[0].in_features

        classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(input_size_vgg, h_layers)),
                        ('relu1', nn.ReLU()),
                        ('dropout1', nn.Dropout(p=dropout)),
                        ('fc2', nn.Linear(h_layers, 256)),
                        ('relu2', nn.ReLU()),
                        ('dropout2', nn.Dropout(p=dropout)),
                        ('fc3', nn.Linear(256, 102)),
                        ('output', nn.LogSoftmax(dim=1))]))
    
    # Replace the default classifier with the new classifier
    model.classifier = classifier
    
    return model


def train_model(model, criterion, optimizer, epochs, trainloader, validationloader, device):
    # Find the device available 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the model to the available device specified above
    model.to(device) 

    counter = 0
    running_loss = 0
    print_every = 40 

    # Time the training process
    start_time = time.time()
    for epoch in range(epochs):
        for data in trainloader:
            # Make the data iterable
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
                    for data in validationloader:
                        images, labels = data
                        
                        # Move to device
                        images, labels = images.to(device), labels.to(device)  # Use GPu if requested and available
                        # Pass in images from the validation set to get the logps
                        output = model(images)
                        # Calculate the loss
                        batch_loss = criterion(output, labels)
                        # Add the batch_loss to the validation set running
                        # to keep track of the validation loss
                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        # Get the probabilities
                        proba = torch.exp(output)
                        # Check for equality with the labels
                        equality = (labels.data == proba.max(dim=1)[1])
                        # With the equality tensor update the accuracy
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                        
                # Print the training and evaluation information
                print("Epoch: {}/{} ".format(epoch+1, epochs),
                 "Training Loss: {:.3f} ".format(running_loss/print_every),
                 "Validation Loss: {:.3f} ".format(validation_loss/len(validationloader)),
                 "Validation Accuracy: {:.3f} ".format(accuracy/len(validationloader)))
                
                # Set running loss back to zero
                running_loss = 0
                # After evaluation, set model to training mode
                model.train()
                
    # Compute total training time
    total_time = time.time() - start_time
    print(
        '\n\n The model trained for: {:.0f}m {:.2f}s'.format(total_time // 60, total_time % 60))


def save_checkpoint(save_path, model, optimizer, classifier, args):
    checkpoint = {'arch': args.arch, 
                  'model': model,
                  'learning_rate': args.learning_rate,
                  'hidden_units': args.hidden_units,
                  'epochs': args.epochs,
                  'classifier' : classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                 }
    return checkpoint

    torch.save(checkpoint, save_path)
    print('Checkpoint saved')

def test_model(model, testloader):
    test_accuracy = 0
    
    # Set to evaluation mode
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for data in testloader:
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
    
    model_accuracy = (test_accuracy/len(testloader)) * 100
    
    return model_accuracy

def main():
    print('Hello and welcome!')

    args = parse_args()

    if args.data_dir:
         data_dir = args.data_dir
    else:
        data_dir = 'flowers'
        
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(), 
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_test_transform = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])


    # Load the datasets with ImageFolder
    training_set = datasets.ImageFolder(train_dir, transform=train_transform)
    validation_set = datasets.ImageFolder(valid_dir, transform=valid_test_transform)
    test_set = datasets.ImageFolder(test_dir, transform=valid_test_transform)

    # Using the image datasets, load image into dataloader
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_set, batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)
    
    pretrained_model = getattr(models, args.arch)(pretrained=True)    
    
    if args.arch == "vgg16":
        choice_of_model = "vgg16"
    else:
        choice_of_model =  "densenet121"
           
    model = load_model(pretrained_model, args.hidden_units, choice_of_model)
    print(model)
    print("Model training is in progress. This may take a while")
    print("-" * 30)
    
    # Set up the loss function to evaluate the error in the model
    criterion = nn.NLLLoss()
    
    # Update the weights and biases to reduce the error
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    epochs = int(args.epochs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_model(model, criterion, optimizer, epochs, trainloader, validationloader, device)
    model_accuracy = test_model(model, testloader)
    model.class_to_idx = training_set.class_to_idx
    save_path = args.save_dir

    save_checkpoint(save_path, model, optimizer, model.classifier, args)
    print("The accuracy of the model is {:.2f}%".format(model_accuracy))

if __name__ == "__main__":  
    main()