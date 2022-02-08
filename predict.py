import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import numpy as np
import pandas as pd

from PIL import Image
import time
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Predict image classification')

    parser.add_argument('--img_path', type=str, help='Path to image prediction')
    parser.add_argument('--checkpoint', help='Path to trained model checkpoint', default='checkpoint.pth')
    parser.add_argument('--topk', help='Number of top predictions', type=int, default=5)
    parser.add_argument('--category_names', type=str, help='File containing category names', default='ImageClassifier/cat_to_name.json')
    parser.add_argument('--gpu', type=str, help='Make use of GPU if available (true|false)')

    return parser.parse_args()

def process_image(image):
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
    image_pil = Image.open(image, mode='r')
    
    # Apply transform to resize and crop image
    image_pil = transform(image_pil)
    
    # Convert image into numpy array
    np_image = np.array(image_pil)
    
    # Normalize based on the preset mean and standard deviation
    image = (np.transpose(np_image,(1,2,0)) - _mean)/_std
    
    # Transpose from third dimension to first dimension of color channel
    image = np.transpose(image, (2,0,1))
    
    return image

def load_checkpoint(model, filepath='checkpoint.pth'):
    """ This function loads a checkpoint and rebuilds the model"""
    
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
    
    if arch == ('vgg16'):
        model = models.vgg16(pretrained=True)
    elif arch == ('densenet'):
        model = models.densenet121(pretrained=True)
 
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.optimizer = checkpoint.get('optimizer')
    model.learning_rate = checkpoint.get('learning_rate')
    model.class_to_idx = checkpoint.get('class_to_idx')
    model.load_state_dict(checkpoint.get('state_dict'))
    
    return model

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img = process_image(image_path)  # Process image before making predictions
    img = torch.from_numpy(img)
    img = img.float().unsqueeze_(0).to(device)
    
    print(type(model))
    
    model.eval()
    
    with torch.no_grad():
        
        model = model.to(device)
        
        ps = torch.exp(model(img))
        top_prob, top_class = ps.topk(topk, dim=1)
        
    return top_prob, top_class


def main():
    args = parse_args()
    
    if args.img_path:
        img_path = args.img_path
    else:
        raise argparse.ArgumentError("File path required for image_path")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_checkpoint(args.checkpoint)
   
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
           
    
    if args.topk:
        topk = args.topk
    else:
        topk = 5
   
    start_time = time.time()
    top_prob, top_class = predict(img_path, model, args.topk, device)
    classes = [cat_to_name[str(idx-1)] for idx in np.array(top_class[0])]
    #classes = [cat_to_name[str(idx)] for idx in top_class]
    #classes = [cat_to_name[str(idx)] for idx in np.array(top_class[0])]
    proba = np.array(top_prob[0])
    
    print(top_prob)
    print(top_class)
    

    i=0
    print('{:*^50s}\n'.format('Prediction Results'))
    while i < topk:
        print('Image: {} \t{:>15}: {:.4f}%'.format(classes[i], 'Probability', proba[i]))
        i += 1

    total_prediction_time = time.time() - start_time

if __name__ == "__main__":
    main()